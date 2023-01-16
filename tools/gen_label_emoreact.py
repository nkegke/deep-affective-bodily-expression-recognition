import os

def main():
	base = '/gpu-data2/nkeg/EmoReact/'
	
	filenames_ = [base + 'Labels/Val_names.txt', base + 'Labels/Train_names.txt']
	videos_ = [base + 'Data/Validation/', base + 'Data/Train/']
	labels_ = [base + 'Labels/val_labels.text', base + 'Labels/train_labels.text']
	files_output = [base + 'TSM/val_videofolder_nomask.txt', base + 'TSM/train_videofolder_nomask.txt']
	for (filenames, videos, labels, file_output) in zip(filenames_, videos_, labels_, files_output):
		# Appending video frames folder abs path on folders
		with open(filenames) as f:
			lines = f.readlines()
		folders = []
		for line in lines[:-1]:
			folder = videos + line.replace("'","")
			folders.append(folder[:-5] + '_frames') #get rid of ".mp4\n"
		
		# Appending string of annotations 'int, int, ..., int, float' on categories
		with open(labels) as f:
			lines = f.readlines()
		categories = []
		for line in lines:
# 			l = line.rstrip().split(',')
# 			categories_list = list(map(int, l[:-1])) + [float(l[-1])]
# 			categories.append(categories_list)
			categories.append(line.rstrip())

		output = []
		for i in range(len(folders)):
			curFolder = folders[i]
			curCategs = categories[i]
			# Counting the number of frames in each video folders
			dir_files = os.listdir(curFolder)
			output.append('%s %d %s' % (curFolder, len(dir_files), curCategs))
			print('%d/%d' % (i, len(folders)))
		with open(file_output, 'w') as f:
			f.write('\n'.join(output))

if __name__ == '__main__':
	main()