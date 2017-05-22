#!/usr/bin/env python
# Facial Cropping using OpenCV
# Adapted from http://stackoverflow.com/questions/13211745/detect-face-then-autocrop-pictures

'''
Sources:
http://opencv.willowgarage.com/documentation/python/cookbook.html
http://www.lucaamore.com/?p=638
'''

import cv #Opencv
from PIL import Image #Image from PIL
import glob
import os

def DetectFace(image, faceCascade, returnImage=False):
    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function
    # modified from: http://www.lucaamore.com/?p=638

    #variables    
    min_size = (20,20)
    haar_scale = 1.1
    min_neighbors = 3
    haar_flags = 0

    # Equalize the histogram
    cv.EqualizeHist(image, image)

    # Detect the faces
    faces = cv.HaarDetectObjects(
            image, faceCascade, cv.CreateMemStorage(0),
            haar_scale, min_neighbors, haar_flags, min_size
        )

    # If faces are found
    if faces and returnImage:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if returnImage:
        return image
    else:
        return faces

def pil2cvGrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tobytes(), pil_im.size[0]  )
    return cv_im

def cv2pil(cv_im):
    # Convert the cv image to a PIL image
    return Image.fromstring("L", cv.GetSize(cv_im), cv_im.tostring())

def imgCrop(image, cropBox, boxScale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    xDelta=max(cropBox[2]*(boxScale-1),0)
    yDelta=max(cropBox[3]*(boxScale-1),0)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)

def faceCrop(imagePattern,boxScale=1):
    # Select one of the haarcascade files:
    #   haarcascade_frontalface_alt.xml  <-- Best one?
    #   haarcascade_frontalface_alt2.xml
    #   haarcascade_frontalface_alt_tree.xml
    #   haarcascade_frontalface_default.xml
    #   haarcascade_profileface.xml
    faceCascade = cv.Load("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")

    imgList=glob.glob(imagePattern)
    if len(imgList)<=0:
        print 'No Images Found'
        return

    for img in imgList:
        pil_im=Image.open(img)
        cv_im=pil2cvGrey(pil_im)
        faces=DetectFace(cv_im,faceCascade)
        if faces:
            n=1
            for face in faces:
                croppedImage=imgCrop(pil_im, face[0],boxScale=boxScale)
                fname,ext=os.path.splitext(img)
                croppedImage.save(fname+'_crop'+str(n)+ext)
                n+=1
        else:
            print 'No faces found:', img
            
            
def crop_face_and_landmarks(img, landmark_points):
    """
    Crop an image to the face and adjust the landmark points so that they match 
    the crop
    
    :param img: a PIL image containing a face
    :param landmark_points: np array for eyes [[Lx,Ly],[Rx,Ry]]
    """
    faceCascade = cv.Load("/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml")
    
    pil_im=img
    cv_im=pil2cvGrey(pil_im)
    face_bounds=DetectFace(cv_im,faceCascade)[0][0]
    
    cropped_landmark_points = []
    
    for eye_pt in landmark_points:
        if (eye_pt[0] < face_bounds[0] or eye_pt[0] > face_bounds[0] + face_bounds[2]
            or eye_pt[1] < face_bounds[1] or eye_pt[1] > face_bounds[1] + face_bounds[3]):
            print("WARNING: Eyes outside of facial bounds!")
        
        cropped_landmark_points += [eye_pt[0]-face_bounds[0], eye_pt[1]-face_bounds[1]]
        
    cropped_image=imgCrop(pil_im, face_bounds, boxScale=1.2)
    
    return (cropped_image, cropped_landmark_points)
    
    