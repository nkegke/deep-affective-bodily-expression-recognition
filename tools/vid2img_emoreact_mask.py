import os
import threading
import glob
import tqdm
import sys

NUM_THREADS = 10

def split(l, n):
	"""Yield successive n-sized chunks from l."""
	for i in range(0, len(l), n):
		yield l[i:i + n]

									  #-vf scale=-1:256
def extract(video, frame_folder, tmpl='%06d.jpg'):
	cmd = 'ffmpeg -i \"{}\" -threads 1 -q:v 0 \"{}/%06d.jpg\" -loglevel quiet'.format(video, frame_folder)
	os.system(cmd)


def target(video_list):
	for video in video_list:
		frame_folder = video[:-4]+'_frames_full/'
		if not os.path.exists(frame_folder):
			os.makedirs(frame_folder)
		extract(video, frame_folder)


def main(VIDEO_ROOT):
	if not os.path.exists(VIDEO_ROOT):
		raise ValueError('No directory: ' + VIDEO_ROOT)
	video_list = glob.glob(VIDEO_ROOT + '*.mp4')
	splits = list(split(video_list, NUM_THREADS))
	threads = []
	for i, spl in enumerate(splits):
		thread = threading.Thread(target=target, args=(spl,))
		thread.start()
		threads.append(thread)

	for thread in threads:
		thread.join()


if __name__ == '__main__':

	train_root = "/gpu-data2/nkeg/EmoReact/Data_mask/Train_mask/"
	val_root = "/gpu-data2/nkeg/EmoReact/Data_mask/Validation_mask/"
	test_root = "/gpu-data2/nkeg/EmoReact/Data_mask/Test_mask/"
	
	print("Extracting training set video frames...")
	main(train_root)
	print("Extracting validation set video frames...")
	main(val_root)
	print("Extracting test set video frames...")
	main(test_root)
		