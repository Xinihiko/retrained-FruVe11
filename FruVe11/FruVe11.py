from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import os

class FruVe11:
	def __init__(self, test_ratio = .1, rand_state=41):
		self.__get_path__()
		self.__load_labels__()
		if self.__check__():
			dataset = np.load(self.cur_dir+'/'+'FruVe11.npz')
			img = dataset['images']
			lbl = dataset['labels']
		else:
			img, lbl = self.__load_images__()
		self.__weight__(img, lbl)
		self.__split__(img, lbl, test_ratio, rand_state)

	def __get_path__(self):
		self.cur_dir = os.path.realpath(__file__).split('\\')
		self.cur_dir = "\\".join(self.cur_dir[:-1])

	def __load_labels__(self):
		files_dirs = os.listdir(self.cur_dir)
		labels = []
		if 'labels.txt' in files_dirs:
			with open(self.cur_dir+'/'+'labels.txt', 'r') as f:
				labels.append(f.read())

			self.classList = labels[0].split('\n')[:-1]
		else:
			print("Error loading labels")

	def __check__(self):
		files_dirs = os.listdir(self.cur_dir)
		if 'FruVe11.npz' in files_dirs:
			return True
		else:
			return False

	def __load_images__(self):
		for folder in os.listdir(data_path):
			for file in os.listdir(data_path+'/'+folder):
				img.append(np.array(Image.open(data_path+'/'+folder+'/'+file).convert('RGB').resize(image_size))*1./255.)
				lbl.append(self.classList.index(folder))

		img = np.array(img)
		lbl = np.array(lbl)

		le = LabelBinarizer()
		lbl = le.fit_transform(lbl)

		np.savez_compressed('FruVe11.npz', images=img, labels=lbl)

		return img, lbl

	def __weight__(self, img, lbl):
		imagePerClass = lbl.sum(axis=0)
		self.classWeight = {}
		for i in range(len(imagePerClass)):
			self.classWeight[i] = imagePerClass.max() / imagePerClass[i]

	def __split__(self, img, lbl, test_ratio, rand_state):
		(self.trainX, self.testX, self.trainY, self.testY) = train_test_split(img, lbl,
															test_size = test_ratio,
															stratify = lbl,
															random_state = rand_state)

if __name__ == '__main__':
	dataset = FruVe11()