import json
import numpy as np
import pandas as pd
import cv2
import os
import sklearn.model_selection
import autokeras as ak
import keras

from keras.api.utils import to_categorical
from keras.api.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from auto_ml.autokeras.AutoModelImageClassifier import *
from data_utils.data_utils import *


class ModelTrainer:
    def __init__(self, data_path, classes, img_size, model, model_dir, model_name):
        self.y_data_val = None
        self.x_data_val = None
        self.model = model
        self.y_data_test = None
        self.y_data_train = None
        self.x_data_test = None
        self.x_data_train = None
        self.data_path = data_path
        self.classes = classes
        self.img_size = img_size
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_path = os.path.join(model_dir, model_name)
        self.history = None
        self.data_encoder = DataEncoder(classes)
        self.image_data = []
        self.labels = []

    def load_data_and_labels(self, min_sample_index=0, max_sample_index=-1):
        # Загружаем данные
        for class_name in self.classes:
            print(f"Loading {class_name}...")

            path = os.path.join(self.data_path, class_name)

            for count, img in enumerate(os.listdir(path)):
                if count < min_sample_index:
                    continue
                if max_sample_index != -1 and count >= max_sample_index:
                    break

                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                if img_arr is None:
                    continue
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                self.image_data.append(cv2.resize(img_arr, (self.img_size, self.img_size)))
                self.labels.append(class_name)

        # Преобразуем в nparray
        self.image_data = np.array(self.image_data)
        self.labels = np.array(self.labels)

    def normalize_data(self):
        self.image_data = normalize(self.image_data)

    def encode_labels(self):
        self.labels = self.data_encoder.to_encoded(self.labels)

    def split_data(self, test_size=0.2, random_state=None):
        # Разделение на тестовую и тренировочную выборку
        self.x_data_train, self.x_data_test, self.y_data_train, self.y_data_test = train_test_split(self.image_data, self.labels,
                                                                                test_size=test_size, stratify=self.labels,
                                                                                shuffle=True, random_state=random_state)
        self.x_data_train, self.x_data_val, self.y_data_train, self.y_data_val = train_test_split(self.x_data_train, self.y_data_train,
                                                                                test_size=test_size, stratify=self.y_data_train,
                                                                                shuffle=True, random_state=random_state)
        print("Data shape:")
        print("X train: ", self.x_data_train.shape)
        print("Y train: ", self.y_data_train.shape)
        print("X test: ", self.x_data_test.shape)
        print("Y test: ", self.y_data_test.shape)
        print("X val: ", self.x_data_val.shape)
        print("Y val: ", self.y_data_val.shape)

    def fit_model(self, epochs=None):
        try:
            self.history = self.model.fit(x=self.x_data_train, y=self.y_data_train, epochs=epochs, validation_data=(self.x_data_val, self.y_data_val)).history
        except Exception as e:
            print(e)
        try:
            os.makedirs(f"{self.model_dir}", exist_ok=True)
            with open(f"{self.model_path}_history.json", "w", encoding="utf-8") as f:
                json.dump(self.history, f)
        except Exception as e:
            print(e)

    def show_graphics(self):
        if not self.history and os.path.exists(f"{self.model_path}_history.json"):
            with open(f'{self.model_path}_history.json', 'r', encoding="utf-8") as f:
                self.history = json.load(f)
        if not self.history:
            return
        history_df = pd.DataFrame(self.history)
        # График loss
        plt.figure()
        history_df[['loss', 'val_loss']].plot()
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
        plt.grid(True)
        plt.savefig(f'{self.model_path}_loss.png')
        plt.close()

        # График accuracy
        plt.figure()
        history_df[['accuracy', 'val_accuracy']].plot()
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train Accuracy', 'Validation Accuracy'], loc='lower right')
        plt.grid(True)
        plt.savefig(f'{self.model_path}_accuracy.png')
        plt.close()

    def save_model(self, extension=".keras"):
        os.makedirs(f"{self.model_dir}", exist_ok=True)
        self.model.save_model(self.model_path + extension)

    def describe_model(self):
        with open(f"{self.model_path}.txt", "a", encoding="utf-8") as f:
            self.model.summary(print_fn=lambda x: f.write(x + "\n"))

        try:
            plot_model(self.model, show_shapes=True, expand_nested=False, to_file=f"{self.model_path}.png")
        except ImportError:
            print("Graphviz not installed - skipping model visualization")
        except Exception:
            plot_model(self.model.model.export_model(), show_shapes=True, expand_nested=False, to_file=f"{self.model_path}.png")

    def calculate_accuracy(self, x_test, y_test):
        y_test_onehot = to_categorical(y_test, num_classes=len(self.classes))
        keras_acc = self.model.evaluate(x_test, y_test_onehot)
        sklearn_acc = accuracy_score(y_test, np.argmax(self.model.predict(x_test), axis=1))
        print("Accuracy (keras):", keras_acc)
        print("Accuracy (sklearn):", sklearn_acc)
        with open(f"{self.model_path}.txt", "w", encoding="utf-8") as f:
            f.write(f"Accuracy (keras): {keras_acc}\n")
            f.write(f"Accuracy (sklearn): {sklearn_acc}\n")

if __name__ == '__main__':
    DATA_PATH = "data"
    CLASSES = os.listdir(DATA_PATH)
    IMG_SIZE = 50
    MAX_IMAGE_NUMBER_PER_CLASS = 3000
    MODEL_NAME = "model_autokeras_automodel_xception"
    MODEL_DIR = os.path.join("models", MODEL_NAME)
    MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(
        block_type="xception",
        normalize=False,
        augment=False,
    )(input_node)
    output_node = ak.ClassificationHead()(output_node)

    current_model = AutoModelImageClassifier(
        inputs=input_node,
        outputs=output_node,
        max_trials=2,
        overwrite=False,
        project_name="autokeras\\autokeras_automodel_xception"
    )

    # current_model = keras.models.load_model(f"{MODEL_PATH}.keras", custom_objects=ak.CUSTOM_OBJECTS)

    model_trainer = ModelTrainer(data_path=DATA_PATH, classes=CLASSES, img_size=IMG_SIZE, model=current_model, model_dir=MODEL_DIR, model_name=MODEL_NAME)

    # Загрузка данных
    model_trainer.load_data_and_labels(min_sample_index=1*MAX_IMAGE_NUMBER_PER_CLASS, max_sample_index=2*MAX_IMAGE_NUMBER_PER_CLASS)

    # Нормализация
    model_trainer.normalize_data()

    # Кодирование меток классов
    model_trainer.encode_labels()

    # Разделение данных
    model_trainer.split_data(test_size=0.2, random_state=42)

    # Обучение модели
    model_trainer.fit_model(epochs=15)

    # Сохранение модели
    model_trainer.save_model(extension=".keras")

    # Проверка точности
    model_trainer.calculate_accuracy(x_test=model_trainer.x_data_test, y_test=model_trainer.y_data_test)

    # Построение графиков
    model_trainer.show_graphics()

    # Описание модели
    model_trainer.describe_model()
