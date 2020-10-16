import tensorflow as tf
try:
    from Model import Model
except ImportError:
    from .Model import Model

class CNN_14(Model):
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, load_path=""):
		super().__init__(input_size, output_size, init_lr, epoch, batch, load_path)
		self.name='CNN_14'

	def __residual_block__(self, x, n, pad):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		x = tf.keras.layers.Conv2D(n, kernel_size=1)(x)
		if pad:
			x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, padding='same')(x)
		else:
			x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		if pad:
			x_s = tf.keras.layers.Conv2D(n, kernel_size=3, strides=1, padding='same')(x_s)
		else:
			x_s = tf.keras.layers.Conv2D(n, kernel_size=3, strides=1)(x_s)
		x_s = tf.keras.layers.BatchNormalization()(x_s)
		x_s = tf.keras.layers.LeakyReLU()(x_s)
		x = tf.keras.layers.add([x_s,x])
		return x

	def __build_mid__(self, x):
		x_2, x_1, x = self.__division_layer__(x)

		neck = [[64, False, False, False], 
				['MAXPOOL'], 
				[104, False, True, True], 
				['MAXPOOL'], 
				[168, True, True, True], 
				['MAXPOOL']]

		for z in range(len(neck)):
			if len(neck[z]) == 1 and neck[z][0] == 'MAXPOOL':
				x_2, x_1, x = self.__maxpool__(x_2, x_1, x)
			elif len(neck[z]) == 4:
				x_2 = self.__residual_block__(x_2, neck[z][0], neck[z][1])
				x_1 = self.__residual_block__(x_1, neck[z][0], neck[z][2])
				x = self.__residual_block__(x, neck[z][0], neck[z][3])

		x = tf.keras.layers.concatenate([x_2,x_1,x])
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)

		return x

	def __layers__(self, input_tensor, output_size):
		x = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=5)(input_tensor)
		x = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

		x = self.__build_mid__(x)

		#goes to pool then FC
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2)(x)
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		x = tf.keras.layers.Dense(256, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.5)(x)
		x = tf.keras.layers.Flatten()(x)
		softmax = tf.keras.layers.Dense(output_size, activation='softmax', name='softmax')(x)
		return softmax

if __name__ == '__main__':
	cnn = CNN_14((112,112),11)
	cnn.summary()