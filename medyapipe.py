import cv2
import itertools
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh

face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                         min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
sample_img = cv2.imread('2.jpg')
face_mesh_results = face_mesh_images.process(sample_img[:,:,::-1])
LEFT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))


img_copy = sample_img[:, :, ::-1].copy()

if face_mesh_results.multi_face_landmarks:

    for face in face_mesh_results.multi_face_landmarks:
        for landmark in face.landmarks:
            x = landmark.x
            y = landmark.y

            shape = img_copy.shape
            relative_x = int(x * shape[1])
            relative_y = int(y * shape[0])

            cv2.circle(img_copy, (relative_x, relative_y), radius=1, color=(225, 0, 100), thickness=1)

fig = plt.figure(figsize=[10, 10])
plt.title("Resultant Image");
plt.axis('off');
plt.imshow(img_copy);
plt.show()