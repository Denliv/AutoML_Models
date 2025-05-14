import keras

from autokeras import ImageClassifier
# from tensorflow.python import keras

class OriginalImageClassifier:
    def __init__(
            self,
            num_classes = None,
            multi_label = False,
            project_name = "image_classifier",
            max_trials = 100,
            directory = None,
            objective = "val_loss",
            overwrite = False,
            seed = None,
            max_model_size = None
            ):
        self._model = ImageClassifier(
            num_classes=num_classes,
            multi_label=multi_label,
            project_name=project_name,
            max_trials=max_trials,
            directory=directory,
            objective=objective,
            overwrite=overwrite,
            seed=seed,
            max_model_size=max_model_size
        )

    @classmethod
    def load_model(cls, model_path, compile_param = False):
        cls.model = keras.models.load_model(model_path, compile=compile_param)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def fit(self, x, y, epochs = None, callbacks = None):
        self._model.fit(x=x, y=y, epochs=epochs, callbacks=callbacks)

    def predict(self, x):
        return self._model.export_model().predict(x)

    def save_model(self, model_path):
        self._model.export_model().save(model_path)

    def evaluate(self, x, y):
        return self._model.evaluate(x=x, y=y)

    def summary(self):
        return self._model.export_model().summary()