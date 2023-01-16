# Run command: python cam.py METHOD MODALITY
#		where METHOD=[gradcam, gradcam++, xgradcam, eigencam]
#			  MODALITY=[face, body]


import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
# from torchvision.models import resnet50
from EmoReact.models import TSN
import os
import sys
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as tF

my_model = TSN(8, 1, 'RGB', 
                base_model='resnet50',
                categorical=True, 
                continuous=False)


############### Model Definition ###############
# checkpoint = torch.load('logs/EmoReact/models/TSN_face_mask_seg1/model_best.pth')
checkpoint = torch.load('logs/EmoReact/models/TSN_body_mask_seg1/0928_165533/model_best.pth')
sd = checkpoint['state_dict']
new_sd = {key.replace('module.', '') : sd[key] for key in sd} #.replace('net.', '')
my_model.load_state_dict(new_sd)


############ Target Layer Definition ###########
target_layers = [my_model.base_model[-2][-1].conv1, my_model.base_model[-2][-1].bn1, 
				 my_model.base_model[-2][-1].conv2, my_model.base_model[-2][-1].bn2,
				 my_model.base_model[-2][-1].conv3, my_model.base_model[-2][-1].bn3, 
				 my_model.base_model[-2][-1].relu]


############### Input Tensor ###################
# input_folder = '/gpu-data2/nkeg/EmoReact/Data_mask/Train_mask/SUSHI28_2_mask_frames_full/' #Curiosity
# input_folder = '/gpu-data2/nkeg/EmoReact/Data_mask/Train_mask/Gay_Marriage326_2_mask_frames_full/' #Excitement
# input_folder = '/gpu-data2/nkeg/EmoReact/Data_mask/Train_mask/BULLYING101_2_mask_frames_full/' #Happiness
input_folder = '/gpu-data2/nkeg/EmoReact/Data_mask/Train_mask/SWING_COPTERS105_2_mask_frames_full/' #Frustration


modality = sys.argv[2]

############### Crop ###############

if modality == 'face':
	# df = pd.read_csv('/gpu-data/filby/EmoReact_V_1.0/Data/AllVideos/SUSHI28_2.mp4_openface/SUSHI28_2.csv')
	# df = pd.read_csv('/gpu-data/filby/EmoReact_V_1.0/Data/AllVideos/Gay_Marriage326_2.mp4_openface/Gay_Marriage326_2.csv')
	# df = pd.read_csv('/gpu-data/filby/EmoReact_V_1.0/Data/AllVideos/BULLYING101_2.mp4_openface/BULLYING101_2.csv')
	df = pd.read_csv('/gpu-data/filby/EmoReact_V_1.0/Data/AllVideos/SWING_COPTERS105_2.mp4_openface/SWING_COPTERS105_2.csv')
	landmarks = ["x_"+str(i) for i in range(68)]
	landmarks.extend (["y_"+str(i) for i in range(68)])
	df = df.rename(columns=lambda x: x.strip())[landmarks]
	os.mkdir('SUSHI28_2_face_crops')
	input_list = []
	for i, file in enumerate(sorted(os.listdir(input_folder))):
		frame = Image.open(os.path.join(input_folder, file)).convert("RGB")
		keypoints = df.iloc[i].values.reshape((2,68)).T
		joint_min_x = int(round(np.nanmin(keypoints[:,0])))
		joint_min_y = int(round(np.nanmin(keypoints[:,1])))
		joint_max_x = int(round(np.nanmax(keypoints[:,0])))
		joint_max_y = int(round(np.nanmax(keypoints[:,1])))
		expand_x = int(round(10/100 * (joint_max_x-joint_min_x)))
		expand_y = int(round(10/100 * (joint_max_y-joint_min_y)))
		bottom = min(joint_max_y+expand_y, frame.height)
		right = min(joint_max_x+expand_x,frame.width)
		top = max(0,joint_min_y-expand_y)
		left = max(0,joint_min_x-expand_x)
	# 	print('Crop: ({},{})\n      ({},{})'.format(top, left, bottom, right))
		crop = tF.crop(frame, top, left, bottom-top, right-left)
		crop.save('SUSHI28_2_face_crops'+'/'+str(i)+'.jpg')
		crop = crop.resize((224,224))
		crop = np.asarray(crop).astype(np.float32)/255
		shp = crop.shape
		input_frame = torch.from_numpy(crop).reshape((shp[2],shp[0],shp[1]))
		input_list.append(input_frame)
	os.system('rm -rf SUSHI28_2_face_crops')
	input_tensor = torch.stack(input_list)


elif modality == 'body':
	# vid = 'SUSHI28_2.mp4'
	# vid = 'Gay_Marriage326_2.mp4'
	# vid = 'BULLYING101_2.mp4'
	vid = 'SWING_COPTERS105_2.mp4'
	txt = os.path.join("/gpu-data2/nkeg/EmoReact/Holistic", vid.replace(".mp4","_hol.txt"))
	with open(txt, "r") as f:
		keypoints_file = f.readlines()
	os.mkdir('SUSHI28_2_body_crops')
	input_list = []
	for i, file in enumerate(sorted(os.listdir(input_folder))):
		frame = Image.open(os.path.join(input_folder, file)).convert("RGB")
		keypoints = keypoints_file[i].rstrip('\n').split(' ')
		keypoints = np.array([float(k) for k in keypoints if k!=''])
		if keypoints.size == 0 or len(keypoints) == 0 or len(keypoints) == 1:
			continue
		keypoints = np.array([ [keypoints[i], keypoints[i+1]] for i in range(0, keypoints.shape[0], 2)])
		keypoints[:,0] *= frame.width
		keypoints[:,1] *= frame.height
		joint_min_x = int(np.min(keypoints[:,0])) 
		joint_min_y = int(np.min(keypoints[:,1]))
		joint_max_x = int(np.max(keypoints[:,0]))
		joint_max_y = int(np.max(keypoints[:,1]))
		expand_x = int(round(10/100 * (joint_max_x-joint_min_x)))
		expand_y = int(round(10/100 * (joint_max_y-joint_min_y)))
		joint_min_x = max(0, joint_min_x - expand_x)
		joint_max_x = min(joint_max_x + expand_x, frame.width)
		joint_min_y = max(0, joint_min_y - expand_y)
		joint_max_y = min(joint_max_y + expand_y, frame.height)
		crop = frame.crop(((joint_min_x, joint_min_y, joint_max_x, joint_max_y)))
		crop.save('SUSHI28_2_body_crops'+'/'+str(i)+'.jpg')
		if crop.size == (0,0) or crop.size == 0:
			continue
		crop = crop.resize((224,224))
		crop = np.asarray(crop).astype(np.float32)/255
		shp = crop.shape
		input_frame = torch.from_numpy(crop).reshape((shp[2],shp[0],shp[1]))
		input_list.append(input_frame)
	os.system('rm -rf SUSHI28_2_body_crops')
	input_tensor = torch.stack(input_list)


############ Construct the CAM object ##########
methods = {"gradcam": GradCAM, "gradcam++": GradCAMPlusPlus, "xgradcam": XGradCAM, "eigencam": EigenCAM}
cam_algorithm = methods[sys.argv[1]]
cam = cam_algorithm(model=my_model, target_layers=target_layers)#, use_cuda=True)


# Class Target
label_names = ['Curiosity', 'Uncertainty', 'Excitement', 'Happiness', 'Surprise', 'Disgust', 'Fear', 'Frustration']
class_target = {label: i for i, label in enumerate(label_names)}
targets = [ClassifierOutputTarget(class_target['Frustration'])]


grayscale_cam = cam(input_tensor=input_tensor, targets=targets)#, eigen_smooth=True, aug_smooth=True)

input_tensor = input_tensor.cpu().detach().numpy()
imgs = []
# Mask input with cam
for (input_frame, frame_cam) in zip(input_tensor, grayscale_cam):
	s = input_frame.shape
	input_frame = input_frame.reshape((s[1], s[2], s[0]))
	cam_image = show_cam_on_image(input_frame, frame_cam, use_rgb=True)
	imgs.append(cam_image)


# Video output
fps = 3
width = imgs[0].shape[1]
height = imgs[0].shape[0]
if not os.path.exists('cams/'+ sys.argv[1]):
	os.mkdir('cams/'+ sys.argv[1])
# out = cv2.VideoWriter('cams/' + sys.argv[1] + '/SUSHI28_2_' + modality + '_Curiosity.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
# out = cv2.VideoWriter('cams/' + sys.argv[1] + '/Gay_Marriage326_2_' + modality + '_Excitement.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
# out = cv2.VideoWriter('cams/' + sys.argv[1] + '/BULLYING101_2_' + modality + '_Happiness.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))
out = cv2.VideoWriter('cams/' + sys.argv[1] + '/SWING_COPTERS105_2_' + modality + '_Frustration.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (int(width), int(height)))

for img in imgs:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	out.write(img)
out.release()
