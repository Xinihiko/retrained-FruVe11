import tensorflow as tf
try:
    from Model import Model
except ImportError:
    from .Model import Model

class CNN_Extra(Model):
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, load_path=""):
		super().__init__(input_size, output_size, init_lr, epoch, batch, load_path)
		self.name='CNN_Extra'

	def __inv_bottleneck__(self, x, n, factor):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		x = tf.keras.layers.Conv2D(n/factor, kernel_size=1)(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.Conv2D(n/factor, kernel_size=1)(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		x_s = tf.keras.layers.Conv2D(n/factor, kernel_size=1)(x_s)
		x_s = tf.keras.layers.Conv2D(n, kernel_size=3, padding='same')(x_s)
		x_s = tf.keras.layers.BatchNormalization()(x_s)
		x_s = tf.keras.layers.LeakyReLU()(x_s)
		x = tf.keras.layers.add([x,x_s])
		return x

	def __bottleneck__(self, x, n):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		x = tf.keras.layers.Conv2D(n/2, kernel_size=1)(x)
		x = tf.keras.layers.Conv2D(n/2, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.Conv2D(n, kernel_size=1)(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		x_s = tf.keras.layers.Conv2D(n, kernel_size=3, padding='same')(x_s)
		x_s = tf.keras.layers.BatchNormalization()(x_s)
		x_s = tf.keras.layers.LeakyReLU()(x_s)
		x = tf.keras.layers.add([x_s,x])
		return x

	def __inception__(self, x, n, sub_n):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		x = tf.keras.layers.Conv2D(n, kernel_size=1)(x)
		x_a = tf.keras.layers.Conv2D(sub_n, activation='relu', kernel_size=(1,3), padding='same')(x)
		x_a = tf.keras.layers.Conv2D(sub_n, activation='relu', kernel_size=(3,1), padding='same')(x_a)
		x = tf.keras.layers.Conv2D(n-sub_n, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.concatenate([x_a,x])
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		x_s = tf.keras.layers.Conv2D(n, kernel_size=3, padding='same')(x_s)
		x_s = tf.keras.layers.BatchNormalization()(x_s)
		x_s = tf.keras.layers.LeakyReLU()(x_s)
		x = tf.keras.layers.add([x_s,x])
		return x

	def __inception_2__(self, x, n, sub_n, sub_m):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		x = tf.keras.layers.Conv2D(n, kernel_size=1)(x)
		x_a = tf.keras.layers.Conv2D(sub_n, activation='relu', kernel_size=(1,3), padding='same')(x)
		x_a = tf.keras.layers.Conv2D(sub_n, activation='relu', kernel_size=(3,1), padding='same')(x_a)
		x_b = tf.keras.layers.Conv2D(sub_m, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.Conv2D(n-sub_n-sub_m, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.concatenate([x_a,x_b,x])
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		x_s = tf.keras.layers.Conv2D(n, kernel_size=3, padding='same')(x_s)
		x_s = tf.keras.layers.BatchNormalization()(x_s)
		x_s = tf.keras.layers.LeakyReLU()(x_s)
		x = tf.keras.layers.add([x_s,x])
		return x

	def __mini_residual__(self, x, n, k=1, strides=2):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		if k == 1:
			x = tf.keras.layers.Conv2D(n, kernel_size=1)(x)
		else:
			x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=k, padding='same')(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, strides=strides)(x)
		x_s = tf.keras.layers.Conv2D(n, kernel_size=3, strides=strides)(x_s)
		x = tf.keras.layers.add([x_s,x])
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)

		return x

	def __division_layer__(self, x):
		# 2 output
		x_1 = self.__mini_residual__(x, 24, k=3, strides=1)
		x = self.__mini_residual__(x, 24, strides=1)

		# 3 output
		x_2 = self.__mini_residual__(x_1, 40, k=3)
		x_1 = tf.keras.layers.add([x_1,x])
		x_1 = self.__mini_residual__(x_1, 40)
		x = self.__mini_residual__(x, 40)
		return x_2, x_1, x

	def __build_mid__(self, x):
		x_2, x_1, x = self.__division_layer__(x)
				#  n, f, i1, i2
		neck = [[ 64, 8, 32],
				["MAXPOOL"],
				[104, 4, 64, 16],
				["MAXPOOL"],
				[168, 7, 64],
				["MAXPOOL"]]

		for z in range(len(neck)):
			if len(neck[z]) == 1 and neck[z][0] == 'MAXPOOL':
				x_2, x_1, x = self.__maxpool__(x_2, x_1, x)
			elif len(neck[z]) == 3:
				x_2 = self.__inv_bottleneck__(x_2, neck[z][0], neck[z][1])
				x_1 = self.__inception__(x_1, neck[z][0], neck[z][2])
				x = self.__bottleneck__(x, neck[z][0])
			elif len(neck[z]) == 4:
				x_2 = self.__inv_bottleneck__(x_2, neck[z][0], neck[z][1])
				x_1 = self.__inception_2__(x_1, neck[z][0], neck[z][2], neck[z][3])
				x = self.__bottleneck__(x, neck[z][0])

		x = tf.keras.layers.concatenate([x_2,x_1,x])
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)

		return x

	def __layers__(self, input_tensor, output_size):
		x = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=(5,1))(input_tensor)
		x = tf.keras.layers.Conv2D(8, activation='relu', kernel_size=(1,5))(x)
		x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

		x = self.__build_mid__(x)

		#goes to pool then FC
		x = tf.keras.layers.GlobalAveragePooling2D()(x)
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.3)(x)
		softmax = tf.keras.layers.Dense(output_size, activation='softmax', name='softmax')(x)
		return softmax

if __name__ == '__main__':
	cnn = CNN_Extra((112,112),11)
	cnn.summary()