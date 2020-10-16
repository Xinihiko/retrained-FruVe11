from CNN.CNN_2 import CNN_2
from CNN.CNN_3 import CNN_3
from CNN.CNN_5 import CNN_5
from CNN.CNN_11 import CNN_11
from CNN.CNN_14 import CNN_14
from CNN.CNN_29 import CNN_29
from CNN.CNN_30 import CNN_30
from CNN.CNN_Extra import CNN_Extra

from FruVe11.FruVe11 import FruVe11

dataset = FruVe11(test_ratio=.11)
print(dataset.trainX.shape)
print(dataset.trainY.shape)
print(dataset.testX.shape)
print(dataset.testY.shape)
print(dataset.classWeight)

# model = CNN_2(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_2.h5')
# # model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_3(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_3.h5')
# # model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_5(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_5.h5')
# # model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_11(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_11.h5')
# # model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_14(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_14.h5')
# # model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_29(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_29.h5')
# # model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_30(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01)
# model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model = CNN_30(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_30.h5')
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)

# model = CNN_Extra(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01)
# model.train(dataset.trainX, dataset.trainY, dataset.testX, dataset.testY, dataset.classList, classWeight=dataset.classWeight)
# model = CNN_Extra(dataset.trainX.shape[1:3], dataset.trainY.shape[-1], epoch=250, init_lr=0.01, load_path='CNN_Extra.h5')
# model.test(dataset.testX, dataset.testY, classNames=dataset.classList, f1=True, savefig=True)