import torch.utils.data as data
import cv2
from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import pandas as pd
import torch
import torchvision.transforms.functional as tF
import torchvision.transforms as transforms


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])
    #
    # @property
    # def num_frames(self):
    #     return int(self._data[3] - self._data[2])

    @property
    def min_frame(self):
        return int(self._data[2])

    @property
    def max_frame(self):
        return int(self._data[3])


class TSNDataSet(data.Dataset):
    def __init__(self, phase, mask, input,
                 num_segments=3, new_length=1, modality='RGB', # image_tmpl='img_{:05d}.jpg',
                 image_tmpl='{:06d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        self.db_path = "/gpu-data/filby/EmoReact_V_1.0/Data"

        self.categorical_emotions = ["Curiosity", "Uncertainty", "Excitement", "Happiness", "Surprise", "Disgust", "Fear", "Frustration"]
        
        self.df = pd.read_csv(os.path.join("EmoReact/{}.csv".format(phase)))
        
        self.video_list = self.df["video"]
        self.phase = phase
        self.input = input
        self.mask = '_mask' if mask else ''


    def get_bounding_box(self, image, keypoints, mode, format="cv2"):
        if mode=='face':
            keypoints = keypoints.values
            keypoints = keypoints.reshape((2,68)).T
            joint_min_x = int(round(np.nanmin(keypoints[:,0])))
            joint_min_y = int(round(np.nanmin(keypoints[:,1])))
            joint_max_x = int(round(np.nanmax(keypoints[:,0])))
            joint_max_y = int(round(np.nanmax(keypoints[:,1])))

        elif mode in ['body','fullbody','fusion']:
            keypoints = np.array([[keypoints[i],keypoints[i+1]] for i in range(0, keypoints.shape[0], 2)])
            keypoints[:,0] *= image.width
            keypoints[:,1] *= image.height
            joint_min_x = int(round(np.nanmin(keypoints[:,0])))
            joint_min_y = int(round(np.nanmin(keypoints[:,1])))
            joint_max_x = int(round(np.nanmax(keypoints[:,0])))
            joint_max_y = int(round(np.nanmax(keypoints[:,1])))

        if format == "cv2":
            return image[max(0,joint_min_y-expand_y):min(joint_max_y+expand_y, image.shape[0]), max(0,joint_min_x-expand_x):min(joint_max_x+expand_x,image.shape[1])]
        elif format == "PIL":

            # Expand 10%
            expand_x = int(round(10/100 * (joint_max_x-joint_min_x)))
            expand_y = int(round(10/100 * (joint_max_y-joint_min_y)))
            
            if mode=='face':
                bottom = min(joint_max_y+expand_y, image.height)
                right = min(joint_max_x+expand_x,image.width)
                top = max(0,joint_min_y-expand_y)
                left = max(0,joint_min_x-expand_x)
                # print(top, left, bottom, right)
                return [bottom, right, top, left]
                # return tF.crop(image, top, left, bottom-top, right-left)
                
            elif mode in ['body','fullbody','fusion']:
                # Legal boundaries
                joint_min_x = max(0, joint_min_x - expand_x)
                joint_max_x = min(joint_max_x + expand_x, image.width)
                joint_min_y = max(0, joint_min_y - expand_y)
                joint_max_y = min(joint_max_y + expand_y, image.height)
                cropped = image.crop(((joint_min_x, joint_min_y, joint_max_x, joint_max_y)))
#                 print(joint_min_x, joint_max_x, joint_min_y, joint_max_y)
                return cropped


    def keypoints(self, index, mode):
        sample = self.df.iloc[index] # for one sample/video
        if mode == 'face':
            # e.g. /gpu-data/filby/EmoReact_V_1.0/Data/AllVideos/Gay_Marriage142_2.mp4_openface/Gay_Marriage142_2.csv
            csv = os.path.join("/gpu-data/filby/EmoReact_V_1.0/Data/", "AllVideos", sample["video"]+"_openface", sample["video"].replace("mp4","csv"))
            df = pd.read_csv(csv)
            df = df.rename(columns=lambda x: x.strip())
            landmarks = ['x_'+str(i) for i in range(68)]
            landmarks.extend(['y_'+str(i) for i in range(68)])
            return df[landmarks]
        elif mode in ['body','fullbody','fusion']:
            # e.g. /gpu-data2/nkeg/EmoReact/Holistic/Gay_Marriage142_2_hol.txt
            txt = os.path.join("/gpu-data2/nkeg/EmoReact/Holistic", sample["video"].replace(".mp4","_hol.txt"))
            with open(txt, "r") as f:
                ls = f.readlines()
            return ls



    def _load_image(self, directory, idx, index, mode):
        keypoints = self.keypoints(index, mode) # for one video
        
        if idx >= len(keypoints):
            idx = len(keypoints)-1
        try:
            if mode == 'face':
                keypoints = keypoints.iloc[idx]
            elif mode in ['body','fullbody','fusion']:
                keypoints = keypoints[idx].rstrip('\n').split(' ')
                keypoints = np.array([float(k) for k in keypoints if k!=''])

        except IndexError as e:
            print("Keypoints error", keypoints, idx)
            raise
            
        frame = Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert("RGB")
        frame_copy = frame.copy()
        fusion_face = frame.copy()
		
        if mode in ['fusion','body']:
            face_keypoints = self.keypoints(index, 'face')
            if idx >= len(face_keypoints):
                idx = len(face_keypoints)-1
            face_keypoints = face_keypoints.iloc[idx]
            
            bottom, right, top, left = self.get_bounding_box(frame, face_keypoints, 'face', format="PIL")
            frame = np.asarray(frame)
            frame[top:bottom, left:right, :] = 0 # black out face region
            frame = Image.fromarray(frame)

            if face_keypoints.isnull().values.any():
                fusion_face = frame_copy
            
        if keypoints.size == 0 or len(keypoints) == 0 or len(keypoints) == 1:
            crop = frame.copy()
            pass #just do the whole frame
        elif mode == 'face' and keypoints.isnull().values.any():
            crop = frame
        else:
            crop = self.get_bounding_box(frame, keypoints, mode, format="PIL")
            if mode == 'face':
                bottom, right, top, left = crop
                crop = tF.crop(frame, top, left, bottom-top, right-left)
            if crop.size == (0,0) or crop.size == 0:
                crop = frame.copy()

        if mode == 'fusion':
            face_crop = self.get_bounding_box(frame_copy, face_keypoints, 'face', format="PIL")
            bottom, right, top, left = face_crop
            fusion_face = tF.crop(frame_copy, top, left, bottom-top, right-left)
            if fusion_face.size == (0,0) or fusion_face.size == 0:
                fusion_face = frame_copy.copy()
                
        ############################# Show input image results #############################                
#             fusion_face.save('crop_tests/'+ 'fus' + self.image_tmpl.format(idx))
#         if not os.path.exists('crop_tests/'):
#             os.mkdir('crop_tests/')
#         crop.save('crop_tests/'+ self.image_tmpl.format(idx))
#         quit()

        return [crop], [fusion_face]


    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments) # + (record.min_frame+1)
            # print(record.num_frames, record.min_frame, record.max_frame)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        sample = self.df.iloc[index]

        fname = os.path.join(self.db_path,"AllVideos",self.df.iloc[index]["video"])
        # print(fname)

        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))-1
        # print(frame_count)
        capture.release()

        if self.phase == 'train':
            record_path = os.path.join('/gpu-data2/nkeg/EmoReact/Data'+self.mask+'/Train'+self.mask,sample["video"].replace(".mp4",self.mask+"_frames_full"))
        elif self.phase == 'val':
            record_path = os.path.join('/gpu-data2/nkeg/EmoReact/Data'+self.mask+'/Validation'+self.mask,sample["video"].replace(".mp4",self.mask+"_frames_full"))
        elif self.phase == 'test':
            record_path = os.path.join('/gpu-data2/nkeg/EmoReact/Data'+self.mask+'/Test'+self.mask,sample["video"].replace(".mp4",self.mask+"_frames_full"))

        record = VideoRecord([record_path, frame_count])

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        # segment_indices = [100]
        return self.get(record, segment_indices, index)

    def get(self, record, indices, index):

        images = list()
        face_images = list()

        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs, face_seg_imgs = self._load_image(record.path, p, index, mode = self.input)
                images.extend(seg_imgs)
                face_images.extend(face_seg_imgs)
                
                if p < record.num_frames:
                    p += 1

        categorical = self.df.iloc[index][self.categorical_emotions]
		
        if self.transform is None:
            process_data = images
            if self.input=='fusion':
                face_process_data = face_images

        else:
            process_data = self.transform(images)
            if self.input=='fusion':
                face_process_data = self.transform(face_images)
            else:
                face_process_data = self.transform(images)

        return process_data, torch.tensor(categorical).float(), self.df.iloc[index]["video"], face_process_data
   
    def __len__(self):
        return len(self.df)

