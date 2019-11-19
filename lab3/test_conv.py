import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

from train_conv import train

data_path = '../datasets/flowers_split/'
resize = 64
batch_size = 128
epochs = 30

model = train(resize=resize, data_path=data_path, batch_size=batch_size, epochs=epochs, augmentation=True)

print('Trained...')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(os.path.join(data_path, 'test'),
                                                  target_size=(resize, resize),
                                                  batch_size=batch_size,
                                                  class_mode='categorical',
                                                  shuffle=False)

Y_pred = model.predict_generator(test_generator, test_generator.samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')
target_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
print(classification_report(test_generator.classes, y_pred, target_names=target_names))
