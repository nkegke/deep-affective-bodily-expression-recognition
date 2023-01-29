# Saves txt file like this /gpu-data2/nkeg/EmoReact/Holistic/Gay_Marriage142_2_hol.txt
# and contains NUM_FRAMES lines with coords of visible (>=0.5) joints: (x,y, x,y, ...)

import tqdm
import pickle
from PIL import Image
import numpy as np
import os
import glob

def prepare(dir):
	pkls = glob.glob(os.path.join(dir,'*.pkl'))
	for pkl in tqdm.tqdm(pkls):
		with open(pkl, 'rb') as f:
			landmarks = pickle.load(f)

		with open(os.path.join('/gpu-data2/nkeg/EmoReact/Holistic/', pkl.replace(dir+'/','').replace('pkl','txt')), 'w') as f:
			
			for key in landmarks['pose']:
				if key==None:
					f.write('\n')
					continue

				keypoints=[]
				for t in key.landmark:
					 if t.visibility>=0.5:
					 	# keypoints.append(str(round(-t.x+1, 2)))
					 	# keypoints.append(str(round(-t.y+1, 2)))
					 	keypoints.append(str(round(t.x, 2)))
					 	keypoints.append(str(round(t.y, 2)))
				keypoints = ' '.join(keypoints)+'\n'
				f.write(keypoints)

			########### include hands
			areas = [landmarks['left'], landmarks['right']]
			for area in areas:
				if area != None:
					for key in area:
						if key==None:
							f.write('\n')
							continue

						keypoints = []
						for t in key.landmark:
							keypoints.append(str(round(t.x, 2)))
							keypoints.append(str(round(t.y, 2)))
						keypoints = ' '.join(keypoints)+'\n'
						f.write(keypoints)
			###########

def main():
	base_dir = '/gpu-data2/nkeg/EmoReact/Data'
	dirs = ['Train', 'Validation', 'Test']
	for dir in dirs:
		prepare(os.path.join(base_dir, dir))

if __name__ == main():
    main()

