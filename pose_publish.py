import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from collections import deque

# === Calibration parameters ===
camera_matrix = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist_coeffs = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)

# === Marker size in meters ===
marker_size = 0.075

# 3D marker corners
object_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# === Smoothing and trajectory ===
alpha = 0.1
smoothed_tvec = None
trajectory = deque(maxlen=1000)

# Mapping config
INPUT_X_RANGE = [-1.5, 1.5]
Z_MIN, Z_MAX = -0.855, 0.855
X_TARGET_AT_Z_MIN = [0.7, 0.07]
X_TARGET_AT_Z_MAX = [0.52, -0.84]
FINAL_X_LIMITS = [0.0, 1.3]

def map_value(value, from_min, from_max, to_min, to_max):
    if from_max == from_min: return (to_min + to_max) / 2
    normalized = (value - from_min) / (from_max - from_min)
    return normalized * (to_max - to_min) + to_min

def init_ros():
    rospy.init_node('aruco_pose_publisher', anonymous=True)
    pub = rospy.Publisher('/aruco_smoothed_pose', PoseStamped, queue_size=10)
    return pub

def publish_pose(pub, tvec, rvec):
    msg = PoseStamped()
    msg.header = Header()
    msg.header.stamp = rospy.Time.now()
    msg.header.frame_id = "camera"

    msg.pose.position.x = tvec[0]
    msg.pose.position.y = tvec[1]
    msg.pose.position.z = tvec[2]

    quat = tf.transformations.quaternion_from_euler(rvec[0], rvec[1], rvec[2])
    msg.pose.orientation.x = quat[0]
    msg.pose.orientation.y = quat[1]
    msg.pose.orientation.z = quat[2]
    msg.pose.orientation.w = quat[3]

    pub.publish(msg)

def detect_aruco(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids

def process_frame(frame, pub):
    global smoothed_tvec
    corners, ids = detect_aruco(frame)

    if ids is not None:
        ids = ids.flatten()
        for (corner, ID) in zip(corners, ids):
            reshaped_corners = corner.reshape((4, 2)).astype(np.float32)
            image_points = reshaped_corners

            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if not success:
                continue

            tvec = tvec.reshape(-1)
            tvec *= 3
            tvec[2] -= 5
            tvec[2] = np.clip(tvec[2], -0.855, 0.855)

            x_val_processed = tvec[0]
            z_val_clamped = tvec[2]
            z_normalized = (z_val_clamped - Z_MIN) / (Z_MAX - Z_MIN)
            target_min_x = map_value(z_normalized, 0, 1, X_TARGET_AT_Z_MIN[0], X_TARGET_AT_Z_MAX[0])
            target_max_x = map_value(z_normalized, 0, 1, X_TARGET_AT_Z_MIN[1], X_TARGET_AT_Z_MAX[1])
            mapped_x = map_value(x_val_processed, INPUT_X_RANGE[0], INPUT_X_RANGE[1], target_min_x, target_max_x)
            tvec[0] = np.clip(mapped_x, FINAL_X_LIMITS[0], FINAL_X_LIMITS[1])

            if smoothed_tvec is not None:
                delta = np.linalg.norm(tvec - smoothed_tvec)
                if delta > 3.0:
                    print("⚠️  Rejected noisy tvec:", delta)
                    continue

            smoothed_tvec = tvec if smoothed_tvec is None else alpha * tvec + (1 - alpha) * smoothed_tvec

            trajectory.append(smoothed_tvec.copy())

            print("\033c", end="")
            print(f"ID: {ID}")
            print("Smoothed tvec (m):", smoothed_tvec)
            publish_pose(pub, smoothed_tvec, rvec.reshape(-1))
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec.reshape(3, 1), 0.05)

    cv2.imshow("ArUco Pose", frame)

if __name__ == '__main__':
    pub = init_ros()
    cap = cv2.VideoCapture(0)

    while not rospy.is_shutdown():
        ret, frame = cap.read()
        if not ret:
            continue

        process_frame(frame, pub)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
