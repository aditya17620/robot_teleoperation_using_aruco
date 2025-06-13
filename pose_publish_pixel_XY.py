#!/usr/bin/env python

import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from franka_msgs.msg import FrankaState

# ==============================================================================
# === ROS & CONTROL PARAMETERS (TUNE THESE) ===
# ==============================================================================

# The topic to publish control commands to.
CONTROL_TOPIC = '/cartesian_impedance_example_controller/equilibrium_pose'

# This is the most important tuning parameter. It converts pixel error
# (how far the marker is from the center of the screen) into a robot
# movement command in meters. You will need to adjust this value based on
# your camera's resolution and how fast you want the robot to move.
# A larger value means the robot moves more for the same pixel error.
PIXEL_TO_METER_GAIN = 0.0005  # meters per pixel

# Low-pass filter factor for smoothing the robot's motion.
# Value should be between 0.0 and 1.0.
# A smaller value means more smoothing but more lag.
# A larger value means less smoothing but more responsive (and potentially jittery).
SMOOTHING_FACTOR = 0.1

# ==============================================================================
# === PHYSICAL & CAMERA PARAMETERS (SET THESE ONCE) ===
# ==============================================================================

# Your camera's calibration matrix.
# This is from your first script.
CAMERA_MATRIX = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
DIST_COEFFS = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)

# Which camera to use. 0 is typically the default built-in camera.
CAMERA_INDEX = 0

# ==============================================================================
# === CORE LOGIC (GENERALLY DO NOT NEED TO EDIT BELOW THIS LINE) ===
# ==============================================================================

# Global variable to hold the target pose message that we will publish.
# We initialize it once and then modify its contents in the loop.
target_pose_msg = PoseStamped()

# Global variables to store the robot's initial state.
# The Z position and orientation will be held constant.
initial_pose_position = None
initial_pose_orientation = None

# Global variable for smoothing the command.
smoothed_command_m = np.array([0.0, 0.0])

def init_ros_and_get_initial_pose():
    """
    Initializes the ROS node, publisher, and gets the robot's starting pose.
    The starting Z-height and orientation will be maintained throughout the control.
    """
    global target_pose_msg, initial_pose_position, initial_pose_orientation

    rospy.init_node('aruco_pixel_to_pose_controller', anonymous=True)
    pub = rospy.Publisher(CONTROL_TOPIC, PoseStamped, queue_size=1)

    print("Waiting for initial Franka state...")
    # Get the current state of the robot to use as a reference
    current_state = rospy.wait_for_message("franka_state_controller/franka_states", FrankaState)
    print("Initial Franka state received.")

    # Store the initial position [x, y, z]
    initial_pose_position = np.array([
        current_state.O_T_EE[12],
        current_state.O_T_EE[13],
        current_state.O_T_EE[14]
    ])

    # Store the initial orientation as a quaternion [x, y, z, w]
    initial_pose_orientation = tf.transformations.quaternion_from_matrix(
        np.transpose(np.reshape(current_state.O_T_EE, (4, 4)))
    )
    initial_pose_orientation /= np.linalg.norm(initial_pose_orientation)

    # Prepare the constant parts of our target pose message
    target_pose_msg.header.frame_id = "panda_link0"
    target_pose_msg.pose.position.z = initial_pose_position[2]  # Z is CONSTANT
    target_pose_msg.pose.orientation.x = initial_pose_orientation[0]
    target_pose_msg.pose.orientation.y = initial_pose_orientation[1]
    target_pose_msg.pose.orientation.z = initial_pose_orientation[2]
    target_pose_msg.pose.orientation.w = initial_pose_orientation[3]

    print(f"Initialization complete. Robot starting at:\n"
          f"Position: {initial_pose_position}\n"
          f"Orientation: {initial_pose_orientation}")
    print(f"Control will maintain Z = {initial_pose_position[2]:.3f} and this orientation.")

    return pub

def detect_aruco_center(frame):
    """Detects an ArUco marker and returns its pixel center coordinates."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        # Take the first detected marker
        marker_corners = corners[0].reshape((4, 2))
        
        # Calculate the center of the marker
        centerX = int(np.mean(marker_corners[:, 0]))
        centerY = int(np.mean(marker_corners[:, 1]))
        
        # For visualization, draw the outline
        cv2.polylines(frame, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)
        cv2.circle(frame, (centerX, centerY), 4, (0, 0, 255), -1)

        return centerX, centerY
        
    return None, None

def process_frame(frame, pub):
    """
    Main processing pipeline for each frame. Detects the marker, calculates
    the control command, and publishes it.
    """
    global smoothed_command_m

    # Get image dimensions to find the center
    height, width, _ = frame.shape
    image_center_x = width // 2
    image_center_y = height // 2
    
    # Visualize the image center
    cv2.circle(frame, (image_center_x, image_center_y), 5, (255, 0, 0), -1)

    # Detect the marker's center
    marker_center_x, marker_center_y = detect_aruco_center(frame)

    if marker_center_x is not None:
        # --- 1. CONTROL: Calculate the error in pixels ---
        # This is the difference between where the marker IS and where we WANT it to be (the center)
        # NOTE: The sign mapping depends on your camera and robot setup.
        # This assumes:
        # - Robot X+ is "forward" (away from the base).
        # - Robot Y+ is "left".
        # - Camera Y+ is "down" in the image.
        # - Camera X+ is "right" in the image.
        # To reduce a positive error_x (marker is to the right), robot should move right (robot Y-).
        # To reduce a positive error_y (marker is down), robot should move forward (robot X+).
        error_x = marker_center_x - image_center_x
        error_y = marker_center_y - image_center_y

        # --- 2. CONVERSION: Convert pixel error to a movement command in meters ---
        # The command is relative to the robot's starting point.
        # We introduce a negative sign to move the robot to *reduce* the error.
        raw_command_m = np.array([
           -error_y * PIXEL_TO_METER_GAIN, # Maps pixel Y-error to robot X-motion
           -error_x * PIXEL_TO_METER_GAIN  # Maps pixel X-error to robot Y-motion
        ])

        # --- 3. FILTERING: Smooth the command to prevent jerky motion ---
        smoothed_command_m = SMOOTHING_FACTOR * raw_command_m + (1 - SMOOTHING_FACTOR) * smoothed_command_m
        
        # --- 4. PUBLISHING: Update and publish the final pose ---
        # The command is an offset from the initial pose
        target_pose_msg.pose.position.x = initial_pose_position[0] + smoothed_command_m[0]
        target_pose_msg.pose.position.y = initial_pose_position[1] + smoothed_command_m[1]
        # Z and orientation are already set and remain constant

        target_pose_msg.header.stamp = rospy.Time.now()
        pub.publish(target_pose_msg)

        # --- 5. VISUALIZATION ---
        print("\033c", end="") # Clears the console for clean output
        print(f"Image Center (px): ({image_center_x}, {image_center_y})")
        print(f"Marker Center (px): ({marker_center_x}, {marker_center_y})")
        print(f"Pixel Error (px):   ({error_x}, {error_y})")
        print("---------------------------------------------")
        print(f"Initial Robot Pos (m): {np.round(initial_pose_position[:2], 3)}")
        print(f"Command Offset (m):    {np.round(smoothed_command_m, 3)}")
        print(f"Final Target Pos (m):  [{target_pose_msg.pose.position.x:.3f}, {target_pose_msg.pose.position.y:.3f}]")
    
    # Display the final image
    cv2.imshow("ArUco Pixel Control", frame)


if __name__ == '__main__':
    try:
        # Initialize ROS and get the robot's starting pose
        control_pub = init_ros_and_get_initial_pose()
        
        # Start video capture
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            rospy.logerr(f"Cannot open camera at index {CAMERA_INDEX}. Exiting.")
            exit()

        print("\nStarting control loop. Press 'q' in the OpenCV window to quit.")
        
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                rospy.logwarn("Failed to grab frame from camera.")
                continue
            
            # The main logic is here
            process_frame(frame, control_pub)

            # Allow exiting with the 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except rospy.ROSInterruptException:
        pass
    finally:
        # Cleanup
        print("\nShutting down.")
        cap.release()
        cv2.destroyAllWindows()
