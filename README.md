# robot_teleoperation_using_aruco
This repository contains all the development for the process of robot teleoperation (Franka Emika Panda, in this case) using only a camera and a ArUco marker

This repository was cloned from the existing repository: https://github.com/GSNCodes/ArUCo-Markers-Pose-Estimation-Generation-Python

Two solutions are implemented in this repository

1. using built in aruco funcions that calculate tvec and rvec, and usin that to publish pose data to the robot via ros.
2. using pixel coordinates of the detected marker and use the area of the marker for estimating depth.
