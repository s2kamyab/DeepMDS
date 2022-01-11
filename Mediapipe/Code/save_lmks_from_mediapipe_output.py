import cv2
import mediapipe as mp
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from os import listdir,makedirs
from os.path import isfile, join

# INITIALIZING OBJECTS
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
tr_path = '..\\x3d\\'
celebA_path = '..Path_to\\celebA\\img_align_celeba\\'
file_list = [f for f in listdir(celebA_path) if isfile(join(celebA_path, f))]

# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# cap = cv2.VideoCapture(0)
n_tr  = 1#0000
# DETECT THE FACE LANDMARKS
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
  # while True:
# success, image = cap.read()
for i in range(n_tr):
    print(i)
    img= cv2.imread(celebA_path+file_list[i]) #sio.loadmat('E:\\thesis_phd\\MDS\\celebA\\img_align_celeba\\'+file_list[i])['img']
    img = cv2.resize(img , (96,96))
    # Flip the image horizontally and convert the color space from BGR to RGB
    img = np.ravel(img, order='C')
    img = np.reshape(img , [96,96,3])
    img = img.astype(np.uint8)
    img= cv2.flip(img, 1)
    
    # To improve performance
    img.flags.writeable = False
    
    # Detect the face landmarks
    results = face_mesh.process(img)
    
    # To improve performance
    img.flags.writeable = True
    
    # Convert back to the BGR color space
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cnt=0
    landmark_list=[]
    landmark_set= np.zeros((468,3))
    if results.multi_face_landmarks:
      for face_landmarks in results.multi_face_landmarks:
          landmark_list=face_landmarks
          for idx, landmark in enumerate(landmark_list.landmark):
              dd_x=landmark.x
              dd_y=landmark.y
              dd_z=landmark.z
              landmark_set[cnt,:]=[dd_x,dd_y,dd_z]
              cnt = cnt+1
              # print(landmark.x)
            # if ((landmark.HasField('visibility') and
            #      landmark.visibility < _VISIBILITY_THRESHOLD) or
            #     (landmark.HasField('presence') and
            #      landmark.presence < _PRESENCE_THRESHOLD)):
            #    continue
            # landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                   # image_cols, image_rows)
          # cnt=cnt+1
          # mp_drawing.draw_landmarks(
          #       image=img,
          #       landmark_list=face_landmarks,
          #       connections=mp_face_mesh.FACEMESH_TESSELATION,
          #       landmark_drawing_spec=None,
          #       connection_drawing_spec=mp_drawing_styles
          #       .get_default_face_mesh_tesselation_style())

    # Display the image
    # cv2.imshow('MediaPipe FaceMesh', img)
    
    # landmark_list = results.multi_face_landmarks[0]
    # landmark_list = np.array(landmark_list)
        
    # sio.savemat(tr_path+str(i)+'.mat',{'lnd': landmark_list})
   