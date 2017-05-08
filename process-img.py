#!/usr/bin/env python
# Image Processing Script for OSVR Review
# Converts images into a format usable by a machine learning system

# Tools Required
#  - Intraface - Windows Only Software (requested from author. Waiting...)
#  - Local Binary Pattern (LBP) - part of OpenCV (http://docs.opencv.org/2.4/modules/contrib/doc/facerec/facerec_tutorial.html#local-binary-patterns-histograms-in-opencv)
#  - Gabor wavelet coefficient (maybe part of OpenCV?)
#
# Process
#  - Use IntraFace to track facial landmark points, output to CSV
#  - Use output CSV to track eyes and line up images so that eyes are in same place every time
#  - Crop & resize image to 100x100px image of just face
#  - For each frame "choose uniform LBP with 8-neighbourhood pixels"
#  - Divide image into 5 equally sized non-overlapping patches
#  - Extract an LBP histogram from each patch, resulting in a 1475th dimensional vector
#  - Extract Gabor features from the same patches, resulting in a 1000th dimensional vector
#  - "Apply PCA to each type of feature separately to keep up to 95% energy"
#  - The final feature vector is the concatenation of PCA results for each feature
#  - ???
#  - Profit


