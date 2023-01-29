import cv2
import mediapipe as mp
import sys
import os
import tqdm
import numpy as np
import pickle

def main():
	# Body Pose Tracking
	mp_holistic = mp.solutions.holistic

	dir_name = sys.argv[1]
	# Output directory
	if not os.path.exists(dir_name + '_hol'):
		os.makedirs(dir_name + '_hol')

	for filename in tqdm.tqdm(os.listdir('./' + dir_name)):
		if filename[-4:]!='.mp4':
			continue
		
		imgs = []
		# Dictionary of lists of lists with holistic landmarks
		# each list of lists contains landmarks for all video frames
		keypoints = {'face': [], 'pose': [], 'posew': [], 'left': [], 'right': []}
		with mp_holistic.Holistic(
				static_image_mode=False,
				min_tracking_confidence=0.9,
				min_detection_confidence=0.5,
				model_complexity=2,
				enable_segmentation=True,
				refine_face_landmarks=True) as holistic:
			capture = cv2.VideoCapture('./' + dir_name + '/' + filename)
			
			# Read frames
			while 1:
				success, image = capture.read()
				if not success:
					break

				image.flags.writeable = False
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
				
				results = holistic.process(image)

				image.flags.writeable = True
				image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

				keypoints['face'].append(results.face_landmarks)
				keypoints['pose'].append(results.pose_landmarks)
				keypoints['posew'].append(results.pose_world_landmarks)
				keypoints['left'].append(results.left_hand_landmarks)
				keypoints['right'].append(results.right_hand_landmarks)

		# Pickle keypoints
		newfilename = filename[:len(filename)-4]
		pklname = './' + dir_name + '_hol/' + newfilename + '_hol.pkl'
		pickle.dump(keypoints, open(pklname, 'wb'))
		
# 		# Load like this 
# 		with open(pklname, 'rb') as pkl:
# 			keypoints = pickle.load(pkl)
# 		keypoints['face'][i].landmark 	# List with frame i face landmarks

if __name__ == main():
    main()
