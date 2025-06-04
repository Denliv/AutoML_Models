from matplotlib import pyplot as plt
from ultralytics import YOLO


if __name__ == "__main__":
    # Параметры
    # model_name = "yolo11n-cls.pt"
    # model_name = "runs/classify/train/weights/last.pt"
    model_name = "yolo11_trained.pt"
    data = ".yolo_data"
    epochs = 100
    imgsz = 50
    patience = 10

    # Отключение аугментации
    augment_params = {
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "erasing": 0.0
    }

    # Загрузка модели
    model = YOLO(model_name)

    # Обучение
    # results = model.train(
    #     data=data,
    #     epochs=epochs,
    #     imgsz=imgsz,
    #     patience=patience,
    #     **augment_params
    # )

    metrics = model.val()

    # Сохранение модели
    # model.save("yolo11_trained.pt")

    model.export(format="saved_model", keras=True, imgsz = 50)
