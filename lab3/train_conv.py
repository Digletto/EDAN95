from keras import layers, models
import matplotlib.pyplot as plt
from keras import layers, models
from keras.preprocessing.image import ImageDataGenerator

# keras.__version__


def train(resize, train_path, batch_size, epochs):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(resize, resize, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.summary()

    model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    # ------------------------------------------------------------------------------------------------

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
            train_path,  # This is the source directory for training images
            target_size=(64, 64),  # All images will be resized to target size
            batch_size=batch_size,
            # Specify the classes explicitly
            classes=['daisy', 'dandelion', 'rose', 'sunflower', 'tulip'],
            # Since we use categorical_crossentropy loss, we need categorical labels
            class_mode='categorical')

    # ------------------------------------------------------------------------------------------------

    total_sample = train_generator.n
    history = model.fit_generator(
            train_generator,
            steps_per_epoch=int(total_sample/batch_size),
            epochs=epochs,
            verbose=1)

    # ------------------------------------------------------------------------------------------------

    plt.figure(figsize=(7, 4))
    plt.plot([i+1 for i in range(epochs)],history.history['acc'], '-o', c='k', lw=2, markersize=9)
    plt.grid(True)
    plt.title("Training accuracy with epochs\n",fontsize=18)
    plt.xlabel("Training epochs", fontsize=15)
    plt.ylabel("Training accuracy", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()

    return history


train(64, '../datasets/flowers_split/train', 128, 20)
