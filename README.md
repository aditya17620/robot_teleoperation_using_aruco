# robot_teleoperation_using_aruco
This repository contains all the development for the process of robot teleoperation (Franka Emika Panda, in this case) using only a camera and a ArUco marker

This repository was cloned from the existing repository: https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

Two solutions are implemented in this repository

1. using built in aruco funcions that calculate tvec and rvec, and usin that to publish pose data to the robot via ros.
  Use calibration.py before running any of the following codes
  - pose_estimation_plot.py
  - pose_estimation_pub.py
    Arguments need to be passed for running the calibration.py to be run. The commands for each of it is:

    python calibration.py --dir <directory_to_folder_of_images> --square_size <square_size>

    You can find more details on other parameters using python calibration.py --help
    
2. using pixel coordinates of the detected marker and use the area of the marker for estimating depth.

More informtion about the working can be found on the original repository
