import cv2
import numpy as np
import rospy
import tf
from geometry_msgs.msg import PoseStamped
from collections import deque
from std_msgs.msg import Header

# === Calibration parameters (replace with yours) ===
camera_matrix = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
                          [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist_coeffs = np.array([[0.09617148, -0.1634111, 0.00022246, -0.00056651, -0.00507073]], dtype=np.float32)

# === Marker size in meters ===
marker_size = 0.075

# Real-world 3D marker corners
object_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# === Smoothing settings ===
alpha = 0.1
smoothed_tvec = None
trajectory = deque(maxlen=1000)

# ROS publisher
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
            tvec[0] *= 3
            tvec[1] *= 3
            tvec[2] *= 3

            # Reject wild jumps
            if smoothed_tvec is not None:
                delta = np.linalg.norm(tvec - smoothed_tvec)
                if delta > 3.0:
                    print("⚠️  Rejected noisy tvec:", delta)
                    continue

            # Smoothing
            if smoothed_tvec is None:
                smoothed_tvec = tvec
            else:
                smoothed_tvec = alpha * tvec + (1 - alpha) * smoothed_tvec

            trajectory.append(smoothed_tvec.copy())
            print("\033c", end="")
            print(f"ID: {ID}")
            print("Smoothed tvec (m):", smoothed_tvec)
            print("rvec:", rvec.ravel())

            publish_pose(pub, smoothed_tvec, rvec.ravel())
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
