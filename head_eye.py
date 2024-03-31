import cv2
import mediapipe as mp
import numpy as np
import math

#landmarks

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

#iris
RIGHT_IRIS = [474, 475, 476, 477]
LEFT_IRIS = [469, 470, 471, 472]

L_iris_center = [468]
R_iris_center = [473]

CHIN = [167, 393]

THAADI = [200]

NOSE = [4]

LH_LEFT = [33]
LH_RIGHT = [133]
RH_LEFT = [362]
RH_RIGHT = [263]

#colors

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

#HEAD-EYE FUNCTIONS

def compute_ipd(left_iris_landmark, right_iris_landmark):
  """Calculates the interpupillary distance (IPD) between two 3D landmarks.

  Args:
    left_iris_landmark: A `np.ndarray` of shape (3,) containing the 3D coordinates of the left iris landmark.
    right_iris_landmark: A `np.ndarray` of shape (3,) containing the 3D coordinates of the right iris landmark.

  Returns:
    A `float` representing the IPD in millimeters.
  """

  # Calculate the distance between the two landmarks.
  distance = np.linalg.norm(left_iris_landmark - right_iris_landmark)

  # Convert the distance to millimeters.
  ipd_in_mm = distance * 1000

  return ipd_in_mm

def are_points_collinear(point1, point2, point3, tolerance):
    # Extract x and y coordinates of the points
    x1, y1 = point1
    x2, y2 = point2
    x3, y3 = point3
    
    # Calculate slopes between pairs of points
    slope12 = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else float('inf')
    slope13 = (y3 - y1) / (x3 - x1) if (x3 - x1) != 0 else float('inf')
    slope23 = (y3 - y2) / (x3 - x2) if (x3 - x2) != 0 else float('inf')
    
    # Check if the slopes are equal or almost equal within the given tolerance
    if abs(slope12 - slope13) <= tolerance and abs(slope12 - slope23) <= tolerance:
        return True
    else:
        return False

def calculate_percentage(binary_list):
    if not binary_list:
        return 0.0  # Handle the case where the list is empty
    
    count_ones = sum(1 for bit in binary_list if bit == 1)
    total_bits = len(binary_list)
    
    percentage_ones = (count_ones / total_bits) * 100.0
    return percentage_ones

def find_leftmost_rightmost(coordinates):
    leftmost = (float('inf'), float('inf'))
    rightmost = (-float('inf'), -float('inf'))
    
    for x, y in coordinates:
        leftmost = (min(leftmost[0], x), min(leftmost[1], y))
        rightmost = (max(rightmost[0], x), max(rightmost[1], y))
    
    return leftmost, rightmost

def transform_coordinates(coordinates):
    leftmost, rightmost = find_leftmost_rightmost(coordinates)
    
    # Calculate the scaling factor
    scaling_factor = 100 / (rightmost[0] - leftmost[0])
    
    transformed_coordinates = []
    for x, y in coordinates:
        # Translate to (50, 50)
        translated_x = x - leftmost[0] + 50
        translated_y = y - leftmost[1] + 50
        
        # Scale the coordinates
        scaled_x = translated_x * scaling_factor
        scaled_y = translated_y * scaling_factor
        
        transformed_coordinates.append([int(scaled_x), int(scaled_y)])
    
    return [np.array(transformed_coordinates)]

def landmarkdet(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    return mesh_coord

#euclidean dist
def eucli(p1, p2):
    x, y = p1
    x1, y1 = p2
    dist = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return dist

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = eucli(iris_center, right_point.ravel())
    tot_dist = eucli(left_point.ravel(), right_point.ravel())
    ratio = center_to_right_dist/tot_dist
    return ratio

def iris_position2(iris_center, landmarks, indices):
    top1 = landmarks[indices[12]]
    top2 = landmarks[indices[11]]
    top = (int((top1[0] + top2[0])/2), int((top1[1] + top2[1])/2))
    bottom = landmarks[indices[4]]
    center_to_top_dist = eucli(iris_center, top)
    tot_dist = eucli(top, bottom.ravel())
    try:
        ratio = center_to_top_dist/tot_dist
    except:
        ratio = -1
    return ratio

def blinkratio(img, landmarks, right_indices, left_indices):
    #right eyes horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    #right eyes vertical line
    rv_top1 = landmarks[right_indices[12]]
    rv_top2 = landmarks[right_indices[11]]
    rv_top = (int((rv_top1[0] + rv_top2[0])/2), int((rv_top1[1] + rv_top2[1])/2))
    rv_bottom = landmarks[right_indices[4]]

def head_pose_estimate(model_points, landmarks, K):
    #h, w = image_size
    '''
    K = np.float64(
                [[w,   0,      0.5*(w-1)],
                [0,         h, 0.5*(h-1)],
                [0.0,       0.0,    1.0]])
    '''
    dist_coef = np.zeros((4, 1))
    ret, rvec, tvec = cv2.solvePnP(model_points, landmarks, K, dist_coef, flags=cv2.SOLVEPNP_ITERATIVE)

    rot_mat = cv2.Rodrigues(rvec)[0]
    P = np.hstack((rot_mat, np.zeros((3, 1), dtype=np.float64)))
    eulerAngles =  cv2.decomposeProjectionMatrix(P)[6]
    yaw   = int(eulerAngles[1, 0]*360)
    pitch = int(eulerAngles[0, 0]*360)
    roll  = eulerAngles[2, 0]*360
    return roll, yaw, pitch

def newirispos2(transformed_eye_coordinates, image):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p1 = flat_cords[0]
    p4 = flat_cords[8]
    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    iris = flat_cords[17]
    #right = flat_cords[0]
    #left = flat_cords[8]

    #earclosed = ((10) / (eucli(right, left)))
    #earopen = ((100) / (eucli(right, left)))

    p = (p1+p4)/2

    #p = (((p2+p6)/2)+((p3+p5)/2))/2 #center of left eye
    #print(p)
    #print(iris)
    con = p-iris
    con = (abs(con[0]), abs(con[1]))
    #print(p, iris)
    #print(p-iris) #center of left eye - center of left iris

    #print("eucli: ", eucli(p,iris))

    # 8 - 9.5, 1-2.5

    # Convert points to integers
    point1_int = (int(p[0]), int(p[1]))
    point2_int = (int(iris[0]), int(iris[1]))

    # Draw circles for the points on the image
    cv2.circle(image, point1_int, 5, (0, 0, 255), -1)  # Red color for point1
    cv2.circle(image, point2_int, 5, (0, 255, 0), -1)  # Green color for point2

    return con

#blink ratio
def newbratio(transformed_eye_coordinates):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    right = flat_cords[0]
    left = flat_cords[8]

    earclosed = ((5.385164807134504 + 4.47213595499958) / 2*(eucli(right, left)))
    earopen = ((35.12833614050059 + 31.400636936215164) / 2*(eucli(right, left)))

    ear = (eucli(p2,p6) + eucli(p3,p5))/2*(eucli(right, left))

    thresh = (earopen + earclosed)/2

    if ear<=thresh:
        return True
    else:
        return False

def yratio(transformed_eye_coordinates, iris_center):
    flat_cords = [item for sublist in transformed_eye_coordinates for item in sublist]

    p2 = flat_cords[13]
    p6 = flat_cords[3]
    p3 = flat_cords[11]
    p5 = flat_cords[5]
    right = flat_cords[0]
    left = flat_cords[8]

    earclosed = ((5.385164807134504 + 4.47213595499958) / 2*(eucli(right, left)))
    earopen = ((35.12833614050059 + 31.400636936215164) / 2*(eucli(right, left)))

    ear = (eucli(p2,p6) + eucli(p3,p5))/2*(eucli(right, left))

    thresh = (earopen + earclosed)/2

    if ear<=thresh:
        return True
    else:
        return False

def smiledet(point1, point2, point33):
    # Calculate vectors from point1 to point2 and from point1 to point33
    vector1 = (point2[0] - point1[0], point2[1] - point1[1])
    vector2 = (point33[0] - point1[0], point33[1] - point1[1])

    # Calculate the cross product of vector1 and vector2
    cross_product = vector1[0] * vector2[1] - vector1[1] * vector2[0]

    # If the cross product is positive, point2 is below the line
    return cross_product

def smileratio(img, landmarks, LOWER_LIPS, CHIN):
    # Calculate lips width
    lips_width = abs(eucli(landmarks[LOWER_LIPS[0]], landmarks[LOWER_LIPS[16]]))

    # Calculate jaw width
    jaw_width = abs(eucli(landmarks[CHIN[0]], landmarks[CHIN[1]]))

    # Calculate the ratio of lips and jaw widths
    ratio = lips_width/jaw_width

    return ratio

def blinkratio(img, landmarks, right_indices, left_indices):
    #right eyes horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    #right eyes vertical line
    rv_top1 = landmarks[right_indices[12]]
    rv_top2 = landmarks[right_indices[11]]
    rv_top = (int((rv_top1[0] + rv_top2[0])/2), int((rv_top1[1] + rv_top2[1])/2))
    rv_bottom = landmarks[right_indices[4]]
    #left eyes horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    #left eyes vertical line
    lv_top1 = landmarks[left_indices[12]]
    lv_top2 = landmarks[left_indices[13]]
    lv_top = (int((lv_top1[0] + lv_top2[0])/2), int((lv_top1[1] + lv_top2[1])/2))
    lv_bottom = landmarks[left_indices[4]]

    rhdist = eucli(rh_right, rh_left)
    rvdist = eucli(rv_top, rv_bottom)

    lhdist = eucli(lh_right, lh_left)
    lvdist = eucli(lv_top, lv_bottom)

    cv2.line(img, rh_right, rh_left, GREEN,2)
    cv2.line(img, rv_top, rv_bottom, WHITE,2)
    cv2.line(img, lh_right, lh_left, GREEN,2)
    cv2.line(img, lv_top, lv_bottom, WHITE,2)

    try:

        rratio = rhdist/rvdist
        lratio = lhdist/lvdist

        ratio = (rratio + lratio)/2
    except:
        ratio = -1

    return ratio

def do_lines_intersect(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate the slopes of the lines
    m1 = (y2 - y1) / (x2 - x1) if x2 - x1 != 0 else float('inf')
    m2 = (y4 - y3) / (x4 - x3) if x4 - x3 != 0 else float('inf')

    # Calculate the y-intercepts of the lines
    b1 = y1 - m1 * x1 if m1 != float('inf') else None
    b2 = y3 - m2 * x3 if m2 != float('inf') else None

    # Check for parallel lines (no intersection)
    if m1 == m2:
        return False

    # Calculate the intersection point
    if m1 != float('inf') and m2 != float('inf'):
        x_intersect = (b2 - b1) / (m1 - m2)
    elif m1 == float('inf'):
        x_intersect = x1
    else:
        x_intersect = x3

    # Check if the intersection point is within the line segments
    if (
        min(x1, x2) <= x_intersect <= max(x1, x2) and
        min(x3, x4) <= x_intersect <= max(x3, x4)
    ):
        return True
    else:
        return False

#MAIN

import streamlit as st

def draw_bounding_rect(use_brect, image, brect, rect_color):
    if use_brect:
        # Outer rectangle
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), rect_color, 2)

    return image


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_info_text(image, brect, facial_text):
    info_text =''
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if facial_text != "":
        info_text = facial_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return image

def head_eye(vid, loading_bar_smile):
    count = 0

    text=''

    eyecount = 0
    headcount = 0

    straight = 0

    blinkcount = 0
    blinklist = []

    fps=0

    prev = 0
    consecutive_blink = 0
    blink_too_long =0

    output_frames = []

    map_face_mesh = mp.solutions.face_mesh

    rect_color = (0, 255, 0)  # Green

    cap = cv2.VideoCapture(vid)

    loading_bar_smile.progress(10)

    output_file = r'E:\project demo\media\eye-contact.mp4'

    #print(output_file)

    # Create a VideoCapture object
    #cap = cv2.VideoCapture(r'E:\website files\bodylang\myapp\pgms\WhatsApp Video 2023-08-18 at 20.00.43.mp4')

    # Check if the camera or video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    # Define the codec and create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Change to 'avc1' for H.264 codec
    fps = 30.0  # Frames per second (you can adjust this)

    # Define the output video dimensions (use the same as the input if not resizing)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    loading_bar_smile.progress(30)

    progress = 30

    with map_face_mesh.FaceMesh(max_num_faces = 1, refine_landmarks = True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:

            ret, frame = cap.read()
            if not ret:
                break

            # Find the dimensions of the frame
            height, width, _ = frame.shape

            # Determine the scaling factor to make the longest edge 600 pixels
            scaling_factor = 800 / max(height, width)

            # Calculate the new dimensions
            new_height = int(height * scaling_factor)
            new_width = int(width * scaling_factor)

            # Resize the frame
            frame = cv2.resize(frame, (new_width, new_height))

            fps = fps+1

            face_3d = []
            face_2d = []

            results = face_mesh.process(frame)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    count = count + 1

                    if progress<=80:
                        progress + 0.01
                        loading_bar_smile.progress((int(progress)))
                    
                    if fps%1441 == 0:
                        blinklist.append(blinkcount)
                        blinkcount = 0
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1  or idx == 61 or idx ==291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * width, lm.y * height)
                                nose_3d = (lm.x * width, lm.y * height, lm.z * 3000)
                            
                            x,y = int(lm.x * width), int(lm.y * height)

                            face_2d.append([x,y])
                            face_3d.append([x,y,lm.z])
                    
                    face_2d = np.array(face_2d, dtype=np.float64)
                    face_3d = np.array(face_3d, dtype=np.float64)

                    focal_length = 1 * width

                    cam_matrix = np.array([
                        [focal_length, 0, height/2],
                        [0,focal_length, width/2],
                        [0,0,1]  
                    ])

                    #distortion parameters 
                    dist_matrix = np.zeros((4,1), dtype=np.float64)

                    #solve PnP
                    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
                    #X,Y,Z = head_pose_estimate(face_3d, face_2d, cam_matrix)
                    #print(X,Y,Z)

                    rot_matrix, jac = cv2.Rodrigues(rot_vec)

                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rot_matrix)

                    x = angles[0] * 360 #pitch
                    y = angles[1] * 360 #yaw
                    #z = angles[2] * 360

                    #print(x,y,z)

                    #x_deg = angles[0]
                    #y_deg = angles[1]
                    #z_deg = angles[2] 

                    #cv2.putText(frame, str(z), (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                    #print(x,' ', y)

                    #cv2.putText(frame, str(int(y)) + ", " + str(int(x)), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)

                    nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_2d[0])-100, int(nose_2d[1]))
                    p2 = (int(nose_2d[0] + y*10)-100, int(nose_2d[1] - x*10))

                    if not (-5<int(y)<5 and -5<int(x)<5): 
                        straight = 0
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        #print(count, ": Head not straight")
                    else:
                        #cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        #cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        straight = 1
                        #print(count, ": Head straight")
                
                mesh_coords = landmarkdet(frame, results, False)

                mesh_points = np.array(mesh_coords)

                #cv2.line(frame, p1, p2, (255, 0, 0), 3)
                fhead = tuple(mesh_points[151])
                chin = tuple(mesh_points[175])

                #cv2.line(frame, fhead, chin, (0, 255, 0), 2)

                threshold = 10 

                #print(abs(fhead[0] - chin[0]))

                # Check if the slope is almost straight
                if straight == 1:
                    if abs(fhead[0] - chin[0]) < threshold:
                        straight = 1
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Head straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
                        headcount = headcount+1
                    else:
                        straight = 0
                        cv2.rectangle(frame, (25, 80), (200, 110), BLACK, -1)
                        cv2.putText(frame, 'Look straight', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                (l_cx, l_cy), l_radius =  cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius =  cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                #detecting smile

                forehead = tuple(mesh_points[FACE_OVAL][0])
                #print(forehead)

                lip_point1 = tuple(mesh_points[LOWER_LIPS][0])
                lip_point2 = tuple(mesh_points[LOWER_LIPS][16])
                lip_point3 = tuple(mesh_points[UPPER_LIPS][13])
                lip_point4 = tuple(mesh_points[LOWER_LIPS][10])

                #cv2.circle(frame, mesh_points[CHIN][0], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[CHIN][1], 2, (255,0,255), 1, cv2.LINE_AA)

                #cv2.circle(frame, mesh_points[L_iris_center][0], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[R_iris_center][0], 2, (255,0,255), 1, cv2.LINE_AA)

                #cv2.circle(frame, mesh_points[LOWER_LIPS][0], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[LOWER_LIPS][10], 2, (255,0,255), 1, cv2.LINE_AA)
                #cv2.circle(frame, mesh_points[LOWER_LIPS][16], 2, (255,0,255), 1, cv2.LINE_AA)

                x1,y1 = mesh_points[LOWER_LIPS][0]
                x2,y2 = mesh_points[LOWER_LIPS][10]
                x3,y3 = mesh_points[LIPS][25]
                x4,y4 = mesh_points[THAADI][0]

                x5,y5 = mesh_points[CHIN][0]
                x6,y6 = mesh_points[CHIN][1]
                
                eye_coordinates = []
                eye_cont_coordinates = []
                r_eye_cont_coordinates = []

                for i in LEFT_EYE:
                    eye_coordinates.append(tuple(mesh_points[i]))

                LEFT_EYE_and_IRIS =[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398, 468, 473]
                RIGHT_EYE_and_IRIS  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246, 473, 468]

                for i in RIGHT_EYE_and_IRIS:
                    r_eye_cont_coordinates.append(tuple(mesh_points[i]))
                
                for i in LEFT_EYE_and_IRIS:
                    eye_cont_coordinates.append(tuple(mesh_points[i]))
                
                # Transform the coordinates
                transformed_eye_coordinates = transform_coordinates(eye_coordinates)
                transformed_eyecont_coordinates = transform_coordinates(eye_cont_coordinates)
                rtransformed_eyecont_coordinates = transform_coordinates(r_eye_cont_coordinates)

                #cv2.polylines(frame, transformed_eye_coordinates, True, GREEN)

                blink = newbratio(transformed_eye_coordinates)
                if blink:
                    if prev == 1:
                        if consecutive_blink<=72:
                            consecutive_blink = consecutive_blink+1
                        else:
                            blink_too_long =1
                    prev = 1
                else:
                    if prev == 1:
                        blinkcount = blinkcount + 1
                    prev = 0
                    consecutive_blink = 0
                    
                #cont =  newirispos2(transformed_eyecont_coordinates)
                cont =  newirispos2(transformed_eyecont_coordinates, frame)
                rcont = newirispos2(rtransformed_eyecont_coordinates, frame)

                #print((cont[0]+rcont[0])/2, (cont[1]+rcont[1])/2)

                if 0<=((cont[0]+rcont[0])/2)<=2.5 and 0<=((cont[1]+rcont[1])/2)<=3.5:
                    contact = True
                else:
                    contact = False

                #if blink ==True:
                    #cv2.rectangle(frame, (25, 40), (200, 66), BLACK, -1)
                    #cv2.putText(frame, 'Blink', (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)

                l1x, l1y = lip_point1
                l2x, l2y = lip_point2

                vertical_distance = l2y - l1y

                rv_bottom = mesh_coords[RIGHT_EYE[4]] #right eye vertical bottom point

                #print(eucli(tuple(center_right),rv_bottom))
                #cv2.rectangle(frame, (25, 60), (200, 90), BLACK, -1)

                #check if traingle is formed in the form \/

                # Calculate lips width
                
                col = are_points_collinear(lip_point1, lip_point3, lip_point4, 0.8)
                #col2 = are_points_collinear(lip_point1, lip_point2, lip_point4, 0.8)


                ratio1 = iris_position(center_right, mesh_points[RH_RIGHT], mesh_points[RH_LEFT])
                ratio2 = iris_position(center_left, mesh_points[LH_RIGHT], mesh_points[LH_LEFT])
                ratio = (ratio1 + ratio2)/2

                topratio2 = iris_position2(center_right, mesh_points, RIGHT_EYE)
                topratio1 = iris_position2(center_left, mesh_points, LEFT_EYE)
                topratio = (topratio1 + topratio2)/2

                #print(int((1/topratio)*100))
                
                #cv2.putText(frame, str(int(ratio*100)), (100,200), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)
                '''
                    if cont:
                        text = 'Eye Contact'
                        rect_color = (0, 255, 0)  # Green
                        #print(count, ": Eye contact")
                '''
                
                #if straight == 1:
                if not blink:
                    if contact:
                        text = 'Eye Contact'
                        rect_color = (0, 255, 0)  # Green
                        #print(count, ": Eye contact")
                        eyecount = eyecount + 1
                        '''
                    if 45<(int(ratio*100))<55 and 11<=int((1/topratio)*100)<=17:
                            #cv2.rectangle(frame, (25, 40), (200, 66), BLACK, -1)
                            text = 'Eye Contact'
                            rect_color = (0, 255, 0)  # Green
                            #print(count, ": Eye contact")
                            #cv2.putText(frame, str(eucli(tuple(center_right),rv_bottom)), (200,100), cv2.FONT_HERSHEY_COMPLEX, 1.0, GREEN, 2)
                    '''
                    else:
                        text = 'Not Eye contact'
                        rect_color = (0, 0, 255)  # Red
                        #print(count, ": Not Eye contact")
                else:
                    text = 'Blink'
                    rect_color = (0, 0, 255)  # Red
                    #print(count, ": Not Eye contact")
                #else:
                    #text = "Look straight"
                    #rect_color = (0, 0, 255)
                
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate bounding rectangle
                    brect = calc_bounding_rect(frame, face_landmarks)

                    # Draw bounding rectangle around the head
                    frame = draw_bounding_rect(True, frame, brect, rect_color)

                    frame = draw_info_text(
                    frame,
                    brect,
                    text)
            
            #cv2.imshow('Frame', frame)
            output_frames.append(frame)
            #st.image(frame, channels="BGR", caption="Processed Frame")

            if cv2.waitKey(24) & 0xFF == ord('q'):
                break

        # Release the VideoWriter object.

        #cv2.imshow('Image', frame)
        cv2.destroyAllWindows()
    
    try:

        head_score = ((headcount/count)*100)
        eye_score = ((eyecount/count)*100)

        loading_bar_smile.progress(70)

        #print(head_score)

        messagep = 'YOUR POSITIVE AREAS: '
        messagen = 'NEEDS IMPROVEMENT: '

        if head_score<=50:
            messagen += "Your head was not straight most of the time. Keep it straight."
        elif 50<head_score<=90:
            messagen += "Consider maintaining a more consistent straight head posture."
        elif 90<head_score:
            messagep += "Great job maintaining your head straight! It showcases your focus and attentiveness."

        if blink_too_long == 1:
            messagen =  messagen + " Don't close your eyes for too long."

        if eye_score<=25:
            messagen = messagen + " It seems like you are looking away occasionally. Consider practicing maintaining eye contact."
        elif 25<eye_score<=50:
            messagen = messagen + " Limited eye contact detected. Consider practicing maintaining eye contact for longer stretches to increase your confidence and connect with your audience."
        elif 50<eye_score<=75:
            messagen = messagen + " Your eye contact is not bad, but try holding it longer."
        elif 75<eye_score<=90:
            messagep = messagep + " Good Job in maintaining eye contact for most of the time."
        elif 90<eye_score:
            messagep = messagep + " Impressive! Your eye contact is very strong!"
        
        try:
            total_blink = sum(blinklist)/len(blinklist)
            #print(total_blink)
        except:
            print('')

        #print("count = ", fps)

        try:
            if total_blink>20:
                messagen = messagen + " You are blinking too much. On average, most people blink around 15 to 20 times each minute. You were blinking on an average of " + str(int(total_blink)) + ". Too much blinking indicate lack of concentration."
        except:
            print('')

        loading_bar_smile.progress(90)

        if messagep == 'YOUR POSITIVE AREAS: ':
            messagep = ''
        if messagen == 'NEEDS IMPROVEMENT: ':
            messagen = ''

        message = messagep + "\n\n" + messagen

    except:
        head_score, eye_score = 0, 0
        message = 'No face detected.'

    return output_frames, message, head_score, eye_score
