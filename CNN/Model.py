from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay 
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Model:
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, load_path=""):
		self.input_size = input_size
		self.output_size = output_size
		self.epoch = epoch
		self.batch = batch
		self.model = self.create_model()
		if load_path != "":
			self.model.load_weights(load_path)
		else:
			opt = tf.keras.optimizers.SGD(lr=init_lr, decay=init_lr/epoch)
			self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
		self.name=""

	def summary(self):
		self.model.summary()

	def create_model(self):
		(width, height) = self.input_size 
		inputs = tf.keras.layers.Input(shape=[width, height, 3])
		output = self.__layers__(inputs, self.output_size)

		model = tf.keras.Model(inputs=inputs, outputs=output)

		return model

	def __division_layer__(self, x, strides=2, factorize = False):
		# 2 output
		if factorize:
			x_1 = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=(1,3), padding='same')(x)
			x_1 = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=(3,1), padding='same')(x_1)
		else:
			x_1 = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3, padding='same')(x)
		x_1 = tf.keras.layers.Conv2D(24, activation='relu', kernel_size=3)(x_1)

		x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=1)(x)
		x = tf.keras.layers.Conv2D(24, activation='relu', kernel_size=3)(x)

		# 3 output
		if factorize:
			x_2 = tf.keras.layers.Conv2D(24, activation='relu', kernel_size=(1,3), padding='same')(x_1)
			x_2 = tf.keras.layers.Conv2D(24, activation='relu', kernel_size=(3,1), padding='same')(x_2)
		else:
			x_2 = tf.keras.layers.Conv2D(24, activation='relu', kernel_size=3, padding='same')(x_1)
		x_2 = tf.keras.layers.Conv2D(40, activation='relu', kernel_size=3, strides=strides)(x_2)

		x_1 = tf.keras.layers.add([x_1,x])
		x_1 = tf.keras.layers.Conv2D(40, activation='relu', kernel_size=3, strides=strides)(x_1)

		x = tf.keras.layers.Conv2D(24, activation='relu', kernel_size=1)(x)
		x = tf.keras.layers.Conv2D(40, activation='relu', kernel_size=3, strides=strides)(x)
		return x_2, x_1, x

	def __maxpool__(self, x_2, x_1, x):
		x_2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x_2)
		x_1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x_1)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
		return x_2, x_1, x

	def train(self, trainImg, trainLbl, testImg, testLbl,  classNames, aug=True, plot=True, classWeight=None):
		eS = EarlyStopping(monitor='val_loss', patience=int(.3*self.epoch), verbose=0, mode='min')
		mChk = ModelCheckpoint(self.name+'.h5', save_best_only=True, monitor='val_loss', mode='min')
		rLR = ReduceLROnPlateau(monitor='val_loss', factor=.01, patience=int(.3*self.epoch), verbose=1, min_lr=1e-6, mode='min')

		h = ""

		if aug:
			aug = ImageDataGenerator(rotation_range=90, zoom_range=.05,
									width_shift_range=.1, height_shift_range=.1,
									shear_range=.15, horizontal_flip=True,
									vertical_flip=True, fill_mode='nearest')
			h = self.model.fit(aug.flow(trainImg, trainLbl, batch_size=self.batch),
							validation_data=(testImg, testLbl),
							steps_per_epoch=len(trainImg)//self.batch,
							epochs=self.epoch, callbacks=[eS, mChk, rLR],
							class_weight=classWeight, verbose=2)
		else:
			h = self.model.fit(	trainImg, trainLbl, 
							batch_size=self.batch,
							validation_data=(testImg, testLbl),
							steps_per_epoch=len(trainImg)//self.batch,
							epochs=self.epoch, callbacks=[eS, mChk, rLR],
							class_weight=classWeight, verbose=2)

		if plot:
			self.plotHistory(h)
		pass

	def plotHistory(self, h):
		n = np.array(range(len(h.history['loss'])))
		plt.style.use('ggplot')
		plt.figure()
		plt.plot(n, h.history['loss'], label='Train_loss')
		plt.plot(n, h.history['val_loss'], label='Val_loss')
		plt.title=('Training Accuracy and Loss')
		plt.xlabel('Epoch')
		plt.ylabel('Loss/Accuracy')
		plt.legend(loc='lower left')
		plt.savefig('Result.png')

	def test(self, img, lbl, classNames=None, f1=False, savefig=True):
		prediction = self.model.predict(img, batch_size=self.batch)
		print(lbl.argmax(axis=1), prediction.argmax(axis=1))
		if f1:
			print(classification_report(lbl.argmax(axis=1), prediction.argmax(axis=1), target_names = classNames))
		if savefig:
			cm = confusion_matrix(lbl.argmax(axis=1), prediction.argmax(axis=1))
			cm_display = ConfusionMatrixDisplay(cm).plot()
			cm_display.figure_.savefig('Confusion Matrix {}.png'.format(self.name))
