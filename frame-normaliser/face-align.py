#!/usr/bin/env python
# Image Alignment
# Takes in an image and AAM facial landmarks (68-pt model, provided with CK+)
# and aligns eyes and reformats image so that faces are aligned

import math
import cv
import cv2
import glob
import os
import numpy as np
from skimage.feature import local_binary_pattern
#from skimage.feature import multiblock_lbp
from PIL import Image
import scipy.io as sio

CKPLUSPATH = "/home/ubuntu/workspace/frame-normaliser/ckplus"
IMAGEPATH = CKPLUSPATH + "/cohn-kanade-images/"
LANDMARKPATH = CKPLUSPATH + "/Landmarks/"
EMOTIONPATH = CKPLUSPATH + "/Emotion/"

def load_frame(subject_id, seq_num, frame_id):
    seq_num_fmt = format(seq_num, '03')
    frame_id_fmt = format(frame_id, '08')
    filename = subject_id + "_" + seq_num_fmt + "_" + frame_id_fmt + ".png"
    path_to_image_file = IMAGEPATH + subject_id + "/" + seq_num_fmt + "/" + filename
    
    try:
        result = Image.open(path_to_image_file).convert('L')
    except:
        result = None
        
    return result

def load_eye_landmarks(subject_id, seq_num, frame_id):
    seq_num_fmt = format(seq_num, '03')
    frame_id_fmt = format(frame_id, '08')
    filename = subject_id + "_" + seq_num_fmt + "_" + frame_id_fmt + "_landmarks.txt"
    path_to_landmark_file = LANDMARKPATH + subject_id + "/" + seq_num_fmt + "/" + filename
    
    # Note that left and right are from the subject's perspective
    left_eye = ["x", "y"]
    right_eye = ["x", "y"]
    
    with open(path_to_landmark_file) as fp:
        for i, line in enumerate(fp):
            if i == 41:
                right_eye = map(float, line.strip().split())
            elif i == 46:
                left_eye = map(float, line.strip().split())
            elif i > 46:
                break
    
    return np.array([left_eye, right_eye], dtype=np.float32)
    
def get_affine_transform_matrix(inPoints, outPoints):
    """
    Get the matrix used for affine transfomation of the image using only two points
    :param inPoints: a numpy matrix of two xy coordinate pairs (dtype=np.float32)
    :param outPoints: a numpy matrix of two xy coordinate pairs (dtype=np.float32)
    """
    # Adapted from http://www.learnopencv.com/average-face-opencv-c-python-tutorial/
    s60 = math.sin(60*math.pi/180);
    c60 = math.cos(60*math.pi/180);  
  
    inPts = np.copy(inPoints).tolist();
    outPts = np.copy(outPoints).tolist();
    
    xin = c60*(inPts[0][0] - inPts[1][0]) - s60*(inPts[0][1] - inPts[1][1]) + inPts[1][0];
    yin = s60*(inPts[0][0] - inPts[1][0]) + c60*(inPts[0][1] - inPts[1][1]) + inPts[1][1];
    
    inPts.append([np.int(xin), np.int(yin)]);
    
    xout = c60*(outPts[0][0] - outPts[1][0]) - s60*(outPts[0][1] - outPts[1][1]) + outPts[1][0];
    yout = s60*(outPts[0][0] - outPts[1][0]) + c60*(outPts[0][1] - outPts[1][1]) + outPts[1][1];
    
    outPts.append([np.int(xout), np.int(yout)]);

    tform = cv2.getAffineTransform(np.array([inPts], dtype=np.float32), np.array([outPts], dtype=np.float32));
    
    return tform;
    
def align_eyes(src_subject_id, src_seq_num, src_frame_id, dst_subject_id, dst_seq_num, dst_frame_id):
    # dst = where we want eyes to be
    
    src_frame = load_frame(src_subject_id, src_seq_num, src_frame_id)
    dst_frame = load_frame(dst_subject_id, dst_seq_num, dst_frame_id)
    
    #Err check
    if src_frame is None or dst_frame is None:
        return None
    
    src_landmarks = load_eye_landmarks(src_subject_id, src_seq_num, src_frame_id)
    dst_landmarks = load_eye_landmarks(dst_subject_id, dst_seq_num, dst_frame_id)
    
    t_parameters = get_affine_transform_matrix(src_landmarks, dst_landmarks).flatten().tolist()
    
    return src_frame.transform(src_frame.size, Image.AFFINE, data = t_parameters)
    
    
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
    xDelta=max(cropBox[2]*(boxScale-1),-10000)
    yDelta=max(cropBox[3]*(boxScale-1),-10000)

    # Convert cv box to PIL box [left, upper, right, lower]
    PIL_box=[cropBox[0]-xDelta, cropBox[1]-yDelta, cropBox[0]+cropBox[2]+xDelta, cropBox[1]+cropBox[3]+yDelta]

    return image.crop(PIL_box)
            
            
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
        ## Cropping boundry check (error condition doesn't actually affect results)
        # if (eye_pt[0] < face_bounds[0] or eye_pt[0] > face_bounds[0] + face_bounds[2]
        #     or eye_pt[1] < face_bounds[1] or eye_pt[1] > face_bounds[1] + face_bounds[3]):
        #     print("WARNING: Eyes outside of facial bounds!")
        cropped_landmark_points.append([eye_pt[0]-face_bounds[0], eye_pt[1]-face_bounds[1]])
        
    cropped_image=imgCrop(pil_im, face_bounds, boxScale=0.9)
    
    return (cropped_image, np.array(cropped_landmark_points, dtype=np.float32))
    
def transform_face_and_landmarks(src_img, landmark_points, transform_matrix):
    """
    Perform an affine transform on an image (using PIL's .transform) and adjust
    the landmark_points so that they match the transform
    
    :param src_img: a PIL image
    :param landmark_points: np array for eyes [[Lx,Ly],[Rx,Ry]]
    :param transform_matrix: np matrix for affine transformation
    """
    t_par = transform_matrix.flatten().tolist()
    
    transformed_image = src_img.transform(src_img.size, Image.AFFINE, data = t_par)
    
    transformed_landmark_points = []
    
    for eye_pt in landmark_points:
        transformed_landmark_points.append([t_par[0]*eye_pt[0]+t_par[1]*eye_pt[1]+t_par[2],
            t_par[3]*eye_pt[0]+t_par[4]*eye_pt[1]+t_par[5]])  
    
    return (transformed_image, np.array(transformed_landmark_points, dtype=np.float32))

def crop_and_align(src, dst, size=None):
    src_subject_id = src[0]
    src_seq_num = src[1]
    src_frame_id = src[2] 
    
    dst_subject_id = dst[0] 
    dst_seq_num = dst[1] 
    dst_frame_id = dst[2]
    
    src_frame_raw = load_frame(src_subject_id, src_seq_num, src_frame_id)
    dst_frame_raw = load_frame(dst_subject_id, dst_seq_num, dst_frame_id)
    
    #Err check
    if src_frame_raw is None or dst_frame_raw is None:
        return None
    
    src_landmarks_raw = load_eye_landmarks(src_subject_id, src_seq_num, src_frame_id)
    dst_landmarks_raw = load_eye_landmarks(dst_subject_id, dst_seq_num, dst_frame_id)
    
    t_matrix = get_affine_transform_matrix(src_landmarks_raw, dst_landmarks_raw)
    
    src_frame, src_landmarks = transform_face_and_landmarks(src_frame_raw, src_landmarks_raw, t_matrix)
    
    final_frame, src_landmarks_crop = crop_face_and_landmarks(src_frame, src_landmarks) 

    if size is not None:
        return final_frame.resize((size,size))
    else:
        return final_frame

f512 = ["S005",1,2]
f517 = ["S005",1,7]
f1011 = ["S010",1,1]
f1012 = ["S010",1,2]
f1055 = ["S010",5,5]

def get_lbp_histogram(img):
    """
    Produces a normalised LBP histogram with 9 bins from a greyscale image. Uses
    standard non-rotation-invariant uniform LBP with 8 neighborhood points in 
    radius 5
    
    :param img: the input image (a 2D array of greyscale values)
    :returns: the normalised histogram
    """
    lbp_image=local_binary_pattern(img,8,2,method='nri_uniform')
    histogram, edges = np.histogram(lbp_image, bins=range(60), density=True)
    return histogram
    
def get_gridded_image(img, rows=5, cols=5):
    """
    Cuts an image into a grid and returns a list containing each cell as a 
    2D array of its greyscale intensities
    """
    cell_width = img.size[0]/rows
    cell_height = img.size[1]/cols
    whole_image = np.array(img.getdata()).reshape(img.size)
    result = []
    for col in range(0,img.size[1],cell_height):
        for row in range(0, img.size[0],cell_width):
            result.append(whole_image[row:row+cell_width,col:col+cell_height])
            
    return result
    
def get_feature_vector(img):
    """
    Generates a 1575-dimension feature vector from an image using the LBP 
    histogram method described in the paper
    """
    image_cell_list = get_gridded_image(img)
    return np.array(map(get_lbp_histogram, image_cell_list)).flatten().tolist()
    
def get_usable_frames(emotion_num):
    """
    Produces a list of usable frame descriptors and their emotion intensities
    """
    result = []
    
    #traverse all dirs
    for subject_dir in os.listdir(EMOTIONPATH):
        #check if subjectdir
        if subject_dir.startswith("S"):
            emotion_label_path = EMOTIONPATH + subject_dir + "/"
            #traverse all seq dirs
            for dirname in os.listdir(emotion_label_path):
                #check if item is sequence directory
                #now check if label file exists for that seq
                if os.path.isdir(emotion_label_path+dirname) and len(os.listdir(emotion_label_path+dirname)) > 0 and os.listdir(emotion_label_path+dirname)[0].endswith("_emotion.txt"):
                    with open(emotion_label_path+dirname+"/"+os.listdir(emotion_label_path+dirname)[0]) as em_file:
                        em_num = int(float(em_file.read().strip(' \t\n\r')))
                        if em_num == emotion_num:
                            #Boom, we got a usable sequence
                            seq_num = int(dirname)
                            end_frame_num = int(os.listdir(emotion_label_path+dirname)[0].split("_")[2])
                            
                            frame_descriptors = []
                            for frame_num in range(1, end_frame_num+1):
                                frame_descriptors.append([subject_dir,seq_num,frame_num])
                            intensities = np.linspace(0, 10, end_frame_num).tolist()
                            
                            result += list(zip(frame_descriptors,intensities))
    
    return result

def get_data_matrix(usable_frames, master_frame):
    """
    Takes in a list of usable frames (from get_usable_frames) and outputs the 
    feature and label matrices, ready to be converted to MATLAB. The master_frame
    is a descriptor determines the alignment of each image prior to processing
    """
    
    data = {}
    
    for frame_descriptor, intensity in usable_frames:
        frame = crop_and_align(frame_descriptor, master_frame, size=100)

        if frame is not None:
            feature = get_feature_vector(frame)
            label = [frame_descriptor[2], intensity]
            
            #If we havent seen this sequence before, set up the data struct
            key = "-".join([frame_descriptor[0], str(frame_descriptor[1])])
            if key not in data:
                data[key] = []
                
            data[key].append([feature, label])
    
    features = [None] * len(data)
    labels =  [None] * len(data)
    
    count = 0
    for key, sequence_data in data.iteritems():
        features[count] = []
        labels[count] = []
        for frame_data in sequence_data:
            features[count].append(frame_data[0]) 
            labels[count].append(frame_data[1])
        count += 1
        
    features_t = []
    for feature in features:
        features_t += [np.array(feature).T]
    
    #Make MATLAB ready
    features_mlr = np.zeros(len(features_t), dtype=object)
    features_mlr[:] = features_t

    return features_mlr, np.array(labels)

#sio.savemat('custom.mat', {'my_train_data_seq' : np.array(dm[0]), 'my_train_label_seq' : np.array(dm[1])})
#