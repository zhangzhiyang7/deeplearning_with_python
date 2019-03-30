import os, shutil
# # 创建数据集与目录
# original_dataset_dir = 'C:/Users/13980/Desktop/dataset/dogs_vs_cats/all/train'
# base_dir = 'C:/Users/13980/Desktop/dataset/dogs_vs_cats/cats_vs_dogs_small'
# os.mkdir(base_dir)
# # 划分训练，验证，测试集
# train_dir = os.path.join(base_dir, 'train')
# os.mkdir(train_dir)
# validation_dir = os.path.join(base_dir, 'validation')
# os.mkdir(validation_dir)
# test_dir = os.path.join(base_dir, 'test')
# os.mkdir(test_dir)
# # 创建训练图像目录
# train_cats_dir = os.path.join(train_dir, 'cats')
# os.mkdir(train_cats_dir)
# train_dogs_dir = os.path.join(train_dir, 'dogs')
# os.mkdir(train_dogs_dir)
# # 创建验证图像目录
# validation_cats_dir = os.path.join(validation_dir, 'cats')
# os.mkdir(validation_cats_dir)
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')
# os.mkdir(validation_dogs_dir)
# # 创建测试图像目录
# test_cats_dir = os.path.join(test_dir, 'cats')
# os.mkdir(test_cats_dir)
# test_dogs_dir = os.path.join(test_dir, 'dogs')
# os.mkdir(test_dogs_dir)
# # 向目录里添加图像
# fNames = ['cat.{}.jpg'.format(i) for i in range(1000)]
# for fName in fNames:
#     src = os.path.join(original_dataset_dir, fName)
#     dst = os.path.join(train_cats_dir, fName)
#     shutil.copyfile(src, dst)
# fNames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fName in fNames:
#     src = os.path.join(original_dataset_dir, fName)
#     dst = os.path.join(validation_cats_dir, fName)
#     shutil.copyfile(src, dst)
# fNames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fName in fNames:
#     src = os.path.join(original_dataset_dir, fName)
#     dst = os.path.join(test_cats_dir, fName)
#     shutil.copyfile(src, dst)
# # --------------------------------------------------------------
# fNames = ['dog.{}.jpg'.format(i) for i in range(1000)]
# for fName in fNames:
#     src = os.path.join(original_dataset_dir, fName)
#     dst = os.path.join(train_dogs_dir, fName)
#     shutil.copyfile(src, dst)
# fNames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
# for fName in fNames:
#     src = os.path.join(original_dataset_dir, fName)
#     dst = os.path.join(validation_dogs_dir, fName)
#     shutil.copyfile(src, dst)
# fNames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
# for fName in fNames:
#     src = os.path.join(original_dataset_dir, fName)
#     dst = os.path.join(test_dogs_dir, fName)
#     shutil.copyfile(src, dst)
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications import VGG16

import matplotlib.pyplot as plt
original_dataset_dir = 'C:/Users/13980/Desktop/dataset/dogs_vs_cats/all/train'
base_dir = 'C:/Users/13980/Desktop/dataset/dogs_vs_cats/cats_vs_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# 增强训练数据
train_dataGen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# 不能增强验证数据
test_dataGen = ImageDataGenerator(rescale=1./255)

train_generator = train_dataGen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
)
validation_generator = test_dataGen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
)

model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2, 2)))
# model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2, 2)))
model.add(conv_base)
model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# 微调模型
conv_base.trainable = False
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
# # model.summary()
# # _________________________________________________________________
# # Layer (type)                 Output Shape              Param #
# # =================================================================
# # conv2d_1 (Conv2D)            (None, 148, 148, 32)      896
# # _________________________________________________________________
# # max_pooling2d_1 (MaxPooling2 (None, 74, 74, 32)        0
# # _________________________________________________________________
# # conv2d_2 (Conv2D)            (None, 72, 72, 64)        18496
# # _________________________________________________________________
# # max_pooling2d_2 (MaxPooling2 (None, 36, 36, 64)        0
# # _________________________________________________________________
# # conv2d_3 (Conv2D)            (None, 34, 34, 128)       73856
# # _________________________________________________________________
# # max_pooling2d_3 (MaxPooling2 (None, 17, 17, 128)       0
# # _________________________________________________________________
# # conv2d_4 (Conv2D)            (None, 15, 15, 128)       147584
# # _________________________________________________________________
# # max_pooling2d_4 (MaxPooling2 (None, 7, 7, 128)         0
# # _________________________________________________________________
# # flatten_1 (Flatten)          (None, 6272)              0
# # _________________________________________________________________
# # dense_1 (Dense)              (None, 512)               3211776
# # _________________________________________________________________
# # dense_2 (Dense)              (None, 1)                 513
# # =================================================================
# # Total params: 3,453,121
# # Trainable params: 3,453,121
# # Non-trainable params: 0
# # _________________________________________________________________
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)
# val_loss: 0.2486 - val_acc: 0.8970
model.save('cats_vs_dogs_small_1.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.figure()
#
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


plt.plot(epochs,
         smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,
         smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs,
         smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,
         smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

