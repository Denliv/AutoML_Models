import numpy as np
import pandas as pd
import cv2
import os
import sklearn.model_selection
from matplotlib import pyplot as plt
import autokeras as ak

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.src.callbacks import EarlyStopping

from auto_ml.autokeras.OriginalImageClassifier import *
from auto_ml.autokeras.AutoModelImageClassifier import *
from data_utils.data_utils import *



DATA_PATH = "data"
CLASSES = os.listdir(DATA_PATH)
IMG_SIZE = 50
MAX_IMAGE_NUMBER_PER_CLASS = 3000


if __name__ == '__main__':

    data_encoder = DataEncoder(CLASSES)

    # model = OriginalImageClassifier(
    #     num_classes=len(CLASSES),
    #     max_trials=5,
    #     overwrite=False,
    #     project_name="image_classification"
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

    x_data_train, x_data_test, y_data_train, y_data_test = [], [], [], []

    for class_name in CLASSES:
        print(class_name)

        image_data = []
        labels = []

        path = os.path.join(DATA_PATH, class_name)

        for count, img in enumerate(os.listdir(path)):
            if count > MAX_IMAGE_NUMBER_PER_CLASS:
                break
            # img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_arr = np.repeat(img_arr, 3, axis=-1)
            if img_arr is not None:
                image_data.append(cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)))
                labels.append(class_name)

        #Преобразуем в nparray
        image_data = np.array(image_data)
        labels = np.array(labels)

        #Подготавливаем данные в нужном формате
        # image_data = add_color_channel(image_data, Data.Keras)

        #Нормализация
        input_data = normalize(image_data)

        #Кодирование меток классов
        labels = data_encoder.to_encoded(labels)

        #Разделение на тестовую и тренировочную выборку
        X_train, X_test, Y_train, Y_test = train_test_split(input_data, labels, test_size=0.2)

        x_data_train.append(X_train)
        x_data_test.append(X_test)
        y_data_train.append(Y_train)
        y_data_test.append(Y_test)

    x_data_train = np.concatenate(x_data_train, axis=0)
    x_data_test = np.concatenate(x_data_test, axis=0)
    y_data_train = np.concatenate(y_data_train, axis=0)
    y_data_test = np.concatenate(y_data_test, axis=0)

    print(x_data_train.shape)
    print(y_data_train.shape)
    print(x_data_test.shape)
    print(y_data_test.shape)

    #Обучение модели
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto', restore_best_weights=True)
    # model.fit(x=x_data_train, y=y_data_train, epochs=20, callbacks=[early_stopping])
    history = model.fit(x=x_data_train, y=y_data_train, epochs=15)

    #Проверка точности
    print("Accuracy:", accuracy_score(y_data_test, model.model.export_model().predict(x_data_test)))

    #Сохранение модели
    model.save_model("model_autokeras_automodel_resnet.keras")

    print(model.model.evaluate(x_data_test, y_data_test))

    keras.utils.plot_model(model, show_shapes=True, expand_treeed=True, to_file="name.png")

    history_df = pd.DataFrame(history.history)
    history_df[['loss', 'val_loss']].plot()
    history_df[['accuracy', 'val_accuracy']].plot()
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()