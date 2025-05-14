import numpy as np
import pandas as pd
import cv2
import os
import sklearn.model_selection
import autokeras as ak
import keras
import graphviz

from keras.api.utils import to_categorical
from keras.api.utils import plot_model
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from auto_ml.autokeras.OriginalImageClassifier import *
from auto_ml.autokeras.AutoModelImageClassifier import *
from data_utils.data_utils import *

def show_graphics(model_history):
    history_df = pd.DataFrame(model_history.history)
    history_df[['loss', 'val_loss']].plot()
    history_df[['accuracy', 'val_accuracy']].plot()
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

DATA_PATH = "data"
CLASSES = os.listdir(DATA_PATH)
IMG_SIZE = 50
MAX_IMAGE_NUMBER_PER_CLASS = 3000
MODEL_DIR = "models"
MODEL_NAME = "model_autokeras_automodel_resnet"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME


if __name__ == '__main__':

    data_encoder = DataEncoder(CLASSES)

    # model = OriginalImageClassifier(
    #     num_classes=len(CLASSES),
    #     max_trials=5,
    #     overwrite=False,
    #     project_name="garbage_image_classification"
    # )

    input_node = ak.ImageInput()
    output_node = ak.ImageBlock(
        block_type="resnet",
        normalize=False,
        augment=False,
    )(input_node)
    output_node = ak.ClassificationHead()(output_node)

    model = AutoModelImageClassifier(
        inputs=input_node,
        outputs=output_node,
        max_trials=3,
        overwrite=False,
        project_name="autokeras_automodel_resnet"
    )

    image_data = []
    labels = []

    for class_name in CLASSES:
        print(class_name)

        path = os.path.join(DATA_PATH, class_name)

        for count, img in enumerate(os.listdir(path)):
            if count >= MAX_IMAGE_NUMBER_PER_CLASS:
                break
            # if count <= MAX_IMAGE_NUMBER_PER_CLASS:
            #     continue
            # if count >= 2 * MAX_IMAGE_NUMBER_PER_CLASS:
            #     break

            # img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            if img_arr is None:
                continue
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            image_data.append(cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)))
            labels.append(class_name)

    #Преобразуем в nparray
    image_data = np.array(image_data)
    labels = np.array(labels)

    #Нормализация
    image_data = normalize(image_data)

    #Кодирование меток классов
    labels = data_encoder.to_encoded(labels)

    #Разделение на тестовую и тренировочную выборку

    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(image_data, labels, test_size=0.2, stratify=labels, shuffle=True, random_state=42)

    print("Data shape:")
    print("X train: ", x_data_train.shape)
    print("Y train: ", y_data_train.shape)
    print("X test: ", x_data_test.shape)
    print("Y test: ", y_data_test.shape)

    #Обучение модели
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)
    # model.fit(x=x_data_train, y=y_data_train, epochs=20, callbacks=[early_stopping])

    history = model.fit(x=x_data_train, y=y_data_train, epochs=15)

    model = keras.models.load_model(f"{MODEL_PATH}.keras", custom_objects=ak.CUSTOM_OBJECTS)

    # Сохранение модели
    # model.save_model(f"{MODEL_PATH}.keras")
    # model.save_model(f"{MODEL_PATH}")
    # model.save_model(f"{MODEL_PATH}.h5")

    #Построение графиков
    show_graphics(history)

    #Проверка точности
    y_data_test_onehot = to_categorical(y_data_test, num_classes=len(CLASSES))
    print("Accuracy (keras):", model.evaluate(x_data_test, y_data_test_onehot))
    print("Accuracy (sklearn):", accuracy_score(y_data_test, np.argmax(model.predict(x_data_test), axis=1)))

    # Отрисовка модели
    model.summary()
    plot_model(model, show_shapes=True, expand_nested=False, to_file=f"{MODEL_PATH}.png")