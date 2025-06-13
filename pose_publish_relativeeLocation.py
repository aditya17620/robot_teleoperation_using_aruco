import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from collections import deque
from std_msgs.msg import Header
from franka_msgs.msg import FrankaState
import time

msg = PoseStamped()

msg.header.frame_id = "panda_link0" # Use a standard frame_id if you have one

CONTROL_TOPIC = '/cartesian_impedance_example_controller/equilibrium_pose'
XY_GAIN = 3.0
SMOOTHING_FACTOR = 0.05
JUMP_REJECTION_THRESHOLD = 0.5 # meters

rvec_old = None
tvec_old = None
correction_factor = None
last_time = time.time()

# --- Physical and Camera Parameters ---
# The exact size of your ArUco marker's black square in meters.
MARKER_SIZE = 0.075 # meters

# Your camera's calibration matrix.
CAMERA_MATRIX = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
DIST_COEFFS = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)

# ==============================================================================
# === CORE LOGIC (GENERALLY DO NOT NEED TO EDIT BELOW THIS LINE) ===
# ==============================================================================

# Real-world 3D coordinates of the marker's corners.
# The Z-coordinate is 0 because the marker is flat.
OBJECT_POINTS = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0]
], dtype=np.float32)

# Global variables to store the smoothed pose between frames.
# They act as the "memory" for the filtering process.
smoothed_tvec = None
smoothed_rvec = None

def wait_for_initial_pose():
    global msg
    current_state = rospy.wait_for_message("franka_state_controller/franka_states",
                                 FrankaState)  # type: FrankaState

    initial_quaternion = \
        tf.transformations.quaternion_from_matrix(
            np.transpose(np.reshape(current_state.O_T_EE,
                                    (4, 4))))
    initial_quaternion = initial_quaternion / \
        np.linalg.norm(initial_quaternion)
    msg.pose.orientation.x = initial_quaternion[0]
    msg.pose.orientation.y = initial_quaternion[1]
    msg.pose.orientation.z = initial_quaternion[2]
    msg.pose.orientation.w = initial_quaternion[3]
    msg.pose.position.x = current_state.O_T_EE[12]
    msg.pose.position.y = current_state.O_T_EE[13]
    msg.pose.position.z = current_state.O_T_EE[14]

def init_ros_publisher():
    """Initializes the ROS node and a single publisher for the control command."""
    rospy.init_node('aruco_to_pose_controller', anonymous=True)
    pub = rospy.Publisher(CONTROL_TOPIC, PoseStamped, queue_size=10)
    return pub

def publish_control_pose(pub, tvec, rvec):
    global msg
    """Creates and publishes a PoseStamped message from tvec and rvec."""

    msg.header.stamp = rospy.Time.now()

    msg.pose.position.x += tvec[0]
    msg.pose.position.y += tvec[1]
    msg.pose.position.z += tvec[2]

    #The rvec (rotation vector) must be converted to a quaternion for the pose message.
    # #'sxyz' is a common convention for the order of axes.
    # quat = tf.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2], 'sxyz')
    # msg.pose.orientation.x = quat[0]
    # msg.pose.orientation.y = quat[1]
    # msg.pose.orientation.z = quat[2]
    # msg.pose.orientation.w = quat[3]

    pub.publish(msg)

def detect_aruco(frame):
    """Detects ArUco markers in a given camera frame."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids

def process_frame(frame, pub):
    """The main processing pipeline for each frame."""
    global tvec_old, rvec_old, smoothed_tvec, smoothed_rvec, last_time, correction_factor
    corners, ids = detect_aruco(frame)

    if ids is not None:
        # --- 1. PERCEPTION: Get the raw physical pose of the first detected marker ---
        # This pose is in real-world units (meters) relative to the camera.
        image_points = corners[0].reshape((4, 2)).astype(np.float32)
        success, rvec_raw, tvec_raw = cv2.solvePnP(OBJECT_POINTS, image_points, CAMERA_MATRIX, DIST_COEFFS)

        tvec_raw = tvec_raw.flatten()
        rvec_raw = rvec_raw.flatten()
        
        # if correction_factor is None:
        #     correction_factor = tvec_raw

        # tvec_raw -= correction_factor

        if success:
            current_time = time.time()
        
        if not success:
            return # If pose estimation fails, do nothing for this frame.

        # --- 3. FILTERING (Part B): Smooth the pose using a low-pass filter ---
        # This reduces jitter and creates a more stable output.
        if smoothed_tvec is None:
            # On the very first detection, initialize the smoothed values directly.
            smoothed_tvec = tvec_raw
            smoothed_rvec = rvec_raw
        else:
            # Apply exponential moving average for smoothing.

            # delta = np.linalg.norm(tvec_raw - smoothed_tvec)
            # if delta > JUMP_REJECTION_THRESHOLD:
            #     print(f"⚠️  Rejected noisy detection (jump of {delta:.2f} meters)")
            #     return # Ignore this frame's data as it's likely noise.
            
            smoothed_tvec = SMOOTHING_FACTOR * tvec_raw + (1 - SMOOTHING_FACTOR) * smoothed_tvec
            smoothed_rvec = SMOOTHING_FACTOR * rvec_raw + (1 - SMOOTHING_FACTOR) * smoothed_rvec
        
        if current_time - last_time >= 0.2:
            if rvec_old is not None:
                rvec_delta = smoothed_rvec - rvec_old
            else:
                rvec_delta = np.zeros_like(smoothed_rvec)
            
            if tvec_old is not None:
                tvec_delta = smoothed_tvec - tvec_old
            else:
                tvec_delta = np.zeros_like(smoothed_tvec)
            
            rvec_old = smoothed_rvec
            tvec_old = smoothed_tvec

        control_tvec = tvec_delta.copy()
        control_rvec = rvec_delta.copy()
        
        # Apply the gain to the X and Y axes.
        control_tvec[0] *= XY_GAIN
        control_tvec[1] *= XY_GAIN

        # temp = control_tvec[1]
        # control_tvec[1] = control_tvec[2]
        # control_tvec[2] = temp * -1

        control_rvec[0] *= XY_GAIN
        control_rvec[1] *= XY_GAIN
        control_rvec[2] *= XY_GAIN

        publish_control_pose(pub, control_tvec, smoothed_rvec)

        cv2.drawFrameAxes(frame, CAMERA_MATRIX, DIST_COEFFS, rvec_raw, tvec_raw.reshape(3, 1), 0.05)
        print("\033c", end="") # Clears the console for clean output
        # print(f"REAL Smoothed Pose (m): {np.round(smoothed_tvec, 3)}")
        print(f"CONTROL Cmd Pose (Gained): {np.round(control_tvec, 3)}")
        print(f"XY Gain: {XY_GAIN}, Smoothing: {SMOOTHING_FACTOR}")

    # Display the final image
    cv2.imshow("ArUco Pose Control", frame)


if __name__ == '__main__':
    # Initialize ROS publisher
    control_pub = init_ros_publisher()
    # Start video capture
    cap = cv2.VideoCapture(0)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            continue
        
        process_frame(frame, control_pub)

        # Allow exiting with the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
