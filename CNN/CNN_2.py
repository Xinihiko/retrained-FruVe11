import tensorflow as tf
try:
    from Model import Model
except ImportError:
    from .Model import Model

class CNN_2(Model):
	def __init__(self, input_size, output_size, init_lr=1e-1, epoch=50, batch=32, load_path=""):
		super().__init__(input_size, output_size, init_lr, epoch, batch, load_path)
		self.name='CNN_2'

	def __residual_block__(self, x, n):
		x_s = tf.keras.layers.Lambda(lambda x: 1 * x)(x)
		x = tf.keras.layers.Conv2D(n, kernel_size=1)(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.Conv2D(n, activation='relu', kernel_size=3, padding='same')(x)
		x = tf.keras.layers.BatchNormalization()(x)
		x = tf.keras.layers.LeakyReLU()(x)
		x_s = tf.keras.layers.Conv2D(n, kernel_size=3, strides=1)(x_s)
		x_s = tf.keras.layers.BatchNormalization()(x_s)
		x_s = tf.keras.layers.LeakyReLU()(x_s)
		x = tf.keras.layers.add([x_s,x])
		return x

	def __build_mid__(self, x):

		neck = [64,	"MAXPOOL", 128, "MAXPOOL", 256, "MAXPOOL", 512]

		for z in range(len(neck)):
			if neck[z] == 'MAXPOOL':
				x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)
			else:
				x = self.__residual_block__(x, neck[z])

		return x

	def __layers__(self, input_tensor, output_size):
		x = tf.keras.layers.Conv2D(4, activation='relu', kernel_size=5)(input_tensor)
		x = tf.keras.layers.Conv2D(16, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.Conv2D(32, activation='relu', kernel_size=3)(x)
		x = tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=2)(x)

		x = self.__build_mid__(x)

		#goes to pool then FC
		x = tf.keras.layers.AveragePooling2D(pool_size=(2,2), strides=2)(x)
		x = tf.keras.layers.Dense(1024, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.5)(x)
		x = tf.keras.layers.Dense(256, activation='relu')(x)
		x = tf.keras.layers.Dropout(0.4)(x)
		x = tf.keras.layers.Flatten()(x)
		softmax = tf.keras.layers.Dense(output_size, activation='softmax', name='softmax')(x)
		return softmax

if __name__ == '__main__':
	cnn = CNN_2((112,112),11)
	cnn.summary()
	# print(cnn.name)