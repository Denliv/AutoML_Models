import numpy as np
import pandas as pd
import cv2
import os
import sklearn.model_selection
import autokeras as ak

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data_utils.data_utils import *



DATA_PATH = "data"
CLASSES = os.listdir(DATA_PATH)
IMG_SIZE = 50


if __name__ == '__main__':

    data_encoder = DataEncoder(CLASSES)

    model = ak.ImageClassifier(
        num_classes=len(CLASSES),
        overwrite=True,
        project_name="image_classification"
    )

    image_data = []
    labels = []

    for class_name in CLASSES:
        print(class_name)

        path = os.path.join(DATA_PATH, class_name)

        for img in os.listdir(path):
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if img_arr is not None:
                image_data.append(cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)))
                labels.append(class_name)

    #Преобразуем в nparray
    image_data = np.array(image_data)
    labels = np.array(labels)

    #Подготавливаем данные в нужном формате
    image_data = add_color_channel(image_data, Data.Keras)

    #Нормализация
    input_data = normalize(image_data)

    #Кодирование меток классов
    labels = data_encoder.to_encoded(labels)

    #Разделение на тестовую и тренировочную выборку
    X_train, X_test, Y_train, Y_test = train_test_split(input_data, labels, test_size=0.2)

    #Обучение модели
    model.fit(x=X_train, y=Y_train)

    #Проверка точности
    print("Accuracy:", accuracy_score(Y_test, model.predict(X_test)))