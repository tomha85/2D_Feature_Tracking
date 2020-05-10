# 2D_Feature_TrackingSFND 2D Feature Tracking


The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load.
Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed.
In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson.
In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures.
See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning.

Dependencies for Running Locally
cmake >= 2.8
All OSes: click here for installation instructions
make >= 4.1 (Linux, Mac), 3.81 (Windows)
Linux: make is installed by default on most Linux distros
Mac: install Xcode command line tools to get make
Windows: Click here for installation instructions
OpenCV >= 4.1
This must be compiled from source using the -D OPENCV_ENABLE_NONFREE=ON cmake flag for testing the SIFT and SURF detectors.
The OpenCV 4.1.0 source code can be found here
gcc/g++ >= 5.4
Linux: gcc / g++ is installed by default on most Linux distros
Mac: same deal as make - install Xcode command line tools
Windows: recommend using MinGW
Basic Build Instructions
Clone this repo.
Make a build directory in the top level directory: mkdir build && cd build
Compile: cmake .. && make
Run it: ./2D_feature_tracking.
