import os

import cv2
import mediapipe as mp
import math
import imutils

def face_landmark_coordinates(face_img):
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,min_detection_confidence=0.5)
    image_rows, image_cols, _ = face_img.shape
    results = face_mesh.process(cv2.cvtColor(face_img , cv2.COLOR_BGR2RGB))
    ls_single_face=results.multi_face_landmarks[0].landmark
    from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates
    ls_single_face=results.multi_face_landmarks[0].landmark
    all_coordinates=[]
    for idx in ls_single_face:
        cord = _normalized_to_pixel_coordinates(idx.x,idx.y,image_cols,image_rows)
        all_coordinates.append(list(cord))
        #cv2.putText(image_input, '.', cord[0:3],cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 2)
    #print(len(all_coordinates))
    for coor in all_coordinates:
        pass
        #cv2.putText(image_input, str(all_coordinates.index(coor)), coor, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    #cv2.imshow('landmarking',face_img)
    #cv2.waitKey(0)
    return all_coordinates

def rotate_face_4_alignment(img,p1,p2):
    dx = p1[0] - p2[0]
    dy = -(p1[1] - p2[1])
    alpha = math.degrees(math.atan2(dy, dx))
    rotation = 180-alpha
    img_2 = imutils.rotate(img, rotation)
    return img_2

def calc_distance(p1,p2):
    return int(abs(math.dist(p1, p2)))

def apply_glasses(inpt_img):
    glass_img = cv2.imread("glass_mask/glass_2.png", -1)
    width_of_glass=calc_distance(all_coor_align_face[127],all_coor_align_face[353])
    height_of_glass=calc_distance(all_coor_align_face[52],all_coor_align_face[118])
    #print('all coor top left [21]: ',height_of_glass)
    #resizing glass
    glass_img=imutils.resize(glass_img,width=width_of_glass,height=height_of_glass)
    #glass_img = cv2.resize(glass_img, (width_of_glass,height_of_glass), interpolation=cv2.INTER_CUBIC)

    coor_point=all_coor_align_face[21]
    adjust_margin=0
    offset_top=calc_distance([coor_point[0],0],coor_point)
    offset_left = calc_distance([0,coor_point[1]],coor_point)-adjust_margin

    y1, y2 = offset_top, offset_top + glass_img.shape[0]
    x1, x2 = offset_left, offset_left + glass_img.shape[1]
    alpha_s = glass_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        inpt_img[y1:y2, x1:x2, c] = (alpha_s * glass_img[:, :, c] + alpha_l * inpt_img[y1:y2, x1:x2, c])
    return inpt_img

def apply_face_mask(face_img_4_mask):
    mask_img_img = cv2.imread("glass_mask/face_mask_2.png", -1)
    width_of_mask = calc_distance(all_coor_align_face[132], all_coor_align_face[361])
    height_of_mask = calc_distance(all_coor_align_face[195], all_coor_align_face[152])
    # print('all coor top left [21]: ',height_of_glass)
    # resizing glass
    #mask_img_img = imutils.resize(mask_img_img, width=width_of_mask, height=height_of_mask)
    mask_img_img = cv2.resize(mask_img_img, (width_of_mask,height_of_mask), interpolation=cv2.INTER_CUBIC)

    coor_point = all_coor_align_face[123]
    adjust_margin = 0
    offset_top = calc_distance([coor_point[0], 0], coor_point)
    offset_left = calc_distance([0, coor_point[1]], coor_point) - adjust_margin

    y1, y2 = offset_top, offset_top + mask_img_img.shape[0]
    x1, x2 = offset_left, offset_left + mask_img_img.shape[1]
    alpha_s = mask_img_img[:, :, 3] / 255.0
    alpha_l = 1.0 - alpha_s
    for c in range(0, 3):
        face_img_4_mask[y1:y2, x1:x2, c] = (alpha_s * mask_img_img[:, :, c] + alpha_l * face_img_4_mask[y1:y2, x1:x2, c])
    return face_img_4_mask

for file in os.listdir('face_img'):
    file='face_img/'+file
    dframe = cv2.imread(file)
    dframe = imutils.resize(dframe, width=800)
    image_input = dframe.copy()

    all_coordinates_org_face=face_landmark_coordinates(image_input)
    aligned_face_img=rotate_face_4_alignment(image_input,all_coordinates_org_face[226],all_coordinates_org_face[446])
    all_coor_align_face=face_landmark_coordinates(aligned_face_img)
    glass_applied_img=apply_glasses(aligned_face_img)
    try:
        pass
        #mask_applied_img=apply_face_mask(aligned_face_img)
    except:
        pass
    cv2.imshow('glass',glass_applied_img)
    #cv2.imshow('mask', glass_applied_img)
    #cv2.imshow('org',dframe)
    cv2.waitKey(0)