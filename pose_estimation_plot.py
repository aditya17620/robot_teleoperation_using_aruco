import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# === Calibration parameters (replace with yours) ===
camera_matrix = np.array([[1.40650756e+03, 0.00000000e+00, 9.67004759e+02],
 [0.00000000e+00, 1.40809873e+03, 5.55713591e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

dist_coeffs = np.array([[ 0.09617148, -0.1634111,   0.00022246, -0.00056651, -0.00507073]],
                       dtype=np.float32)

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

# For plotting
trajectory = deque(maxlen=1000)


# <<< START: MINIMAL ADDITIONS (CONFIG & HELPER) =============================
# --- MAPPING CONFIGURATION ---
# CRITICAL: This is the expected input range of tvec[0] *AFTER* it has been
# multiplied by 3. If your raw tvec[0] goes from -0.5 to 0.5, then this
# range should be [-1.5, 1.5]. Adjust this to your physical setup.
INPUT_X_RANGE = [-1.5, 1.5]

# Define the dynamic mapping rules based on tvec[2]
Z_MIN, Z_MAX = -0.855, 0.855
X_TARGET_AT_Z_MIN = [0.7, 0.07]   # Target X range when tvec[2] is at its MIN
X_TARGET_AT_Z_MAX = [0.52, -0.84] # Target X range when tvec[2] is at its MAX
FINAL_X_LIMITS = [0.0, 1.3]      # Final absolute clamp for the output

# --- Helper function for linear mapping ---
def map_value(value, from_min, from_max, to_min, to_max):
    """Maps a value from one range to another."""
    if from_max == from_min: return (to_min + to_max) / 2
    normalized = (value - from_min) / (from_max - from_min)
    return normalized * (to_max - to_min) + to_min
# <<< END: MINIMAL ADDITIONS =================================================


def detect_aruco(frame):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(frame)
    return corners, ids

def visualize_aruco(frame, corners, ids):
    global smoothed_tvec

    if ids is not None:
        ids = ids.flatten()
        for (corner, ID) in zip(corners, ids):
            reshaped_corners = corner.reshape((4, 2)).astype(np.float32)
            image_points = reshaped_corners

            # SolvePnP
            success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)
            if not success:
                continue

            # Your original transformations
            tvec = tvec.reshape(-1)
            tvec[0] *= 3
            tvec[1] *= 3
            tvec[2] *= 3
            
            tvec[2] = tvec[2] - 5

            # Clamp to [-0.855, 0.855]
            tvec[2] = max(-0.855, min(0.855, tvec[2]))


            # <<< START: NEW MAPPING LOGIC INJECTED HERE ============================
            # This block calculates the new tvec[0] based on the current tvec[2]
            # It uses the values of tvec *after* your transformations above.
            x_val_processed = tvec[0]
            z_val_clamped = tvec[2]

            # Interpolate the target min and max for X based on Z's position
            z_normalized = (z_val_clamped - Z_MIN) / (Z_MAX - Z_MIN)
            target_min_x = map_value(z_normalized, 0, 1, X_TARGET_AT_Z_MIN[0], X_TARGET_AT_Z_MAX[0])
            target_max_x = map_value(z_normalized, 0, 1, X_TARGET_AT_Z_MIN[1], X_TARGET_AT_Z_MAX[1])
            
            # Map the processed x_val (which was multiplied by 3) to the new dynamic target range
            mapped_x = map_value(x_val_processed, INPUT_X_RANGE[0], INPUT_X_RANGE[1], target_min_x, target_max_x)
            
            # Apply the final absolute clamp and update tvec[0]
            tvec[0] = np.clip(mapped_x, FINAL_X_LIMITS[0], FINAL_X_LIMITS[1])
            # <<< END: NEW MAPPING LOGIC ============================================


            # Your original code continues from here, now using the MODIFIED tvec
            # Reject wild jumps
            if smoothed_tvec is not None:
                delta = np.linalg.norm(tvec - smoothed_tvec)
                if delta > 3.0: # This threshold now applies to the mapped data
                    print("⚠️  Rejected noisy tvec:", delta)
                    continue

            # Smoothing
            if smoothed_tvec is None:
                smoothed_tvec = tvec
            else:
                smoothed_tvec = alpha * tvec + (1 - alpha) * smoothed_tvec
            
            # Save to trajectory
            trajectory.append(smoothed_tvec.copy())

            # Debug print
            print("\033c", end="")
            print(f"ID: {ID}")
            # I am printing the final smoothed_tvec, which is what your original code did implicitly
            print("Smoothed tvec (m):", smoothed_tvec) 
            print('rvec: ', rvec)

            # Draw marker and axis
            pts = reshaped_corners.astype(int)
            for i in range(4):
                cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
            # Your original code used the un-smoothed tvec for drawing axes. I'll respect that.
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec.reshape(3, 1), 0.05)

    cv2.imshow("ArUco Pose", frame)

# This is your original plot_trajectory function, restored exactly
def plot_trajectory():
    if len(trajectory) < 2:
        return

    traj = np.array(trajectory)
    # The values in traj are now the final mapped and smoothed values
    traj[:, 1] *= 2
    traj[:, 2] *= 2
    ax.cla()
    # ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color='blue')
    ax.plot(traj[:, 0], traj[:, 1], color='blue')
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    # ax.set_zlabel("Z (m)")
    ax.set_title("Smoothed Trajectory")
    pad = 0.1
    ax.set_xlim(np.min(traj[:, 0]) - pad, np.max(traj[:, 0]) + pad)
    ax.set_ylim(np.min(traj[:, 1]) - pad, np.max(traj[:, 1]) + pad)
    # ax.set_zlim(np.min(traj[:, 2]) - pad, np.max(traj[:, 2]) + pad)
    plt.pause(0.001)

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    # Your original figure setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, ids = detect_aruco(frame)
        visualize_aruco(frame, corners, ids)
        plot_trajectory()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
