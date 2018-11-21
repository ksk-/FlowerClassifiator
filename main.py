import cv2
import itertools
import keras.callbacks
import numpy as np

import matplotlib
matplotlib.use("Qt5Agg", force=True)

import matplotlib.pyplot as plt

from enum import Enum
from pathlib import Path

from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import model_to_dot

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class Flower(object):
    class Category(Enum):
        DAISY = 0
        DANDELION = 1
        ROSE = 2
        SUNFLOWER = 3
        TULIP = 4

    _categories = {
        'daisy': Category.DAISY,
        'dandelion': Category.DANDELION,
        'rose': Category.ROSE,
        'sunflower': Category.SUNFLOWER,
        'tulip': Category.TULIP,
    }

    @classmethod
    def is_flower(cls, name: str) -> bool:
        return name in cls._categories

    def __init__(self, path: Path, category: str):
        self._path = path
        self._category = self._categories[category]

    @property
    def path_to_image(self) -> str:
        return str(self._path.absolute())

    @property
    def category(self) -> Category:
        return self._category


class ModelBuilder(object):
    def __init__(self, model):
        self._model = model

    def layer(self, layer):
        self._model.add(layer)
        return self

    def build(self, params: dict, summary=False, plot=False):
        if summary:
            self._model.summary()

        if plot:
            model_to_dot(self._model).write(
                path='model.svg', prog='dot', format='svg'
            )

        self._model.compile(**params)

        return self._model


class Callback(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self._train_loss = list()
        self._val_loss = list()

    @property
    def train_loss(self) -> list:
        return self._train_loss

    @property
    def val_loss(self) -> list:
        return self._val_loss

    def on_epoch_end(self, epoch, logs=None):
        self._train_loss.append(logs.get('loss'))
        self._val_loss.append(logs.get('val_loss'))


def _flowers(path: Path):
    for category in path.iterdir():
        if category.is_dir() and Flower.is_flower(category.name):
            yield from (
                Flower(f, category.name)
                for f in category.iterdir()
                if f.is_file() and f.suffix == '.jpg'
            )


def _get_images(path: Path) -> tuple:
    encoder = OneHotEncoder(sparse=False, categories='auto')

    images = list()
    labels = list()

    for flower in _flowers(path):
        image = cv2.imread(flower.path_to_image, flags=cv2.IMREAD_COLOR)
        image = cv2.resize(image, (120, 120), interpolation=cv2.INTER_AREA)

        images.append(image)
        labels.append(flower.category.value)

    images = np.asarray(images)
    labels = np.asarray(labels)
    labels = encoder.fit_transform(labels.reshape(-1, 1))

    return images, labels


def _get_model(input_shape, class_count):
    params = {
        'loss': categorical_crossentropy,
        'optimizer': 'adam',
        'metrics': ['accuracy'],
    }

    return ModelBuilder(model=Sequential()).layer(
        Conv2D(
            filters=32,
            kernel_size=(5, 5),
            activation='relu',
            input_shape=input_shape
        )
    ).layer(
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
    ).layer(
        MaxPooling2D(pool_size=(2, 2))
    ).layer(
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
    ).layer(
        MaxPooling2D(pool_size=(2, 2))
    ).layer(
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
    ).layer(
        MaxPooling2D(pool_size=(2, 2))
    ).layer(
        Dropout(rate=0.25)
    ).layer(
        Flatten()
    ).layer(
        Dense(units=500, activation='relu')
    ).layer(
        Dropout(rate=0.5)
    ).layer(
        Dense(units=class_count, activation='softmax')
    ).build(params, summary=True, plot=True)


def _show_loss(epochs: int, train_loss: list, val_loss: list):
    x_axis = np.linspace(1, epochs, num=epochs)

    plt.figure(figsize=(10, 7))

    plt.plot(
        x_axis, train_loss, color='b', lw=3, alpha=0.7, label='Train Loss'
    )

    plt.plot(
        x_axis, val_loss, color='r', lw=3, alpha=0.7, label='Val Loss'
    )

    plt.title('Loss graph')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


def _show_confusion_matrix(cm, categories: list):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Conclusion matrix')
    plt.colorbar()

    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories)
    plt.yticks(tick_marks, categories)

    thresh = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], 'd'),
            horizontalalignment='center',
            color='white' if cm[i, j] > thresh else 'black'
        )

    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    images, labels = _get_images(Path('./train/'))
    test_images, test_labels = _get_images(Path('./test/'))

    train_images, valid_images, train_labels, valid_labels = train_test_split(
        images, labels,
        stratify=labels,
        test_size=0.15,
        shuffle=True,
        random_state=333
    )

    train_generator = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
    )

    train_generator.fit(train_images)

    valid_generator = ImageDataGenerator(rescale=2.0 / 255)
    valid_generator.fit(valid_images)

    model = _get_model(train_images[0].shape, Flower.Category.__len__())

    callback = Callback()

    epochs = 30
    batch_size = 32

    model.fit_generator(
        epochs=epochs,
        generator=train_generator.flow(
            train_images, train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=42,
        ),
        steps_per_epoch=(train_images.shape[0] / batch_size),
        validation_data=valid_generator.flow(
            valid_images, valid_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=19,
        ),
        validation_steps=(valid_images.shape[0] / batch_size),
        callbacks=[callback]
    )

    score = model.evaluate(test_images, test_labels, verbose=0)
    print('Loss: ', score[0])
    print('Acc:', score[1])

    predictions = model.predict(test_images)
    test_labels_labeled = np.argmax(test_labels, axis=1)
    predictions_labeled = np.argmax(predictions, axis=1)

    print('Acc:', accuracy_score(test_labels_labeled, predictions_labeled))

    _show_loss(epochs, callback.train_loss, callback.val_loss)

    _show_confusion_matrix(
        confusion_matrix(test_labels_labeled, predictions_labeled),
        list(Flower._categories.keys())
    )
