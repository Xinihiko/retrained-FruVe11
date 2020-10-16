import tensorflow as tf
try:
    from Model import Model
except ImportError:
    from .Model import Model

class CNN_30(Model):
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, load_path=""):
		super().__init__(input_size, output_size, init_lr, epoch, batch, load_path)
		self.name='CNN_30'

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

	def __build_mid__(self, x):
		x_2, x_1, x = self.__division_layer__(x, strides=1)
				#  n, f, i1, i2
		neck = [["MAXPOOL"],
				[ 64, 8, 32],
				["MAXPOOL"],
				[104, 4, 64],
				["MAXPOOL"],
				[168, 7, 64, 32],
				[272, 8, 64, 80],
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
		x = tf.keras.layers.Dropout(0.4)(x)
		x = tf.keras.layers.Dense(256, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.4)(x)
		softmax = tf.keras.layers.Dense(output_size, activation='softmax', name='softmax')(x)
		return softmax

if __name__ == '__main__':
	cnn = CNN_30((112,112),11)
	cnn.summary()