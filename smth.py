import os
import cv2
import keras

from ultralytics import YOLO
from ultralytics.utils.plotting import *

if __name__ == "__main__":
    # plot_results(dir='path/to/results.csv')
    video_captor = cv2.VideoCapture(0)
    hand_gestures = [
        'call', 'dislike', 'fist', 'four', 'like', 'mute', 'no_gesture', 'ok', 'one', 'palm',
        'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three',
        'three2', 'two_up', 'two_up_inverted'
    ]
    # model = keras.models.load_model("models/yolo11")
    model = YOLO("yolo11_trained.pt")
    img_arr = cv2.imread("data/one/fff68641-d921-48f0-8427-9d8462613518.jpeg", cv2.IMREAD_COLOR)
    # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
    img_arr = cv2.resize(img_arr, (50, 50))
    _, frame = video_captor.read()
    print(hand_gestures[model.predict(frame, save=True)[0].probs.top1])
    print("-----------------------------------------")
    print(model.predict(img_arr)[0].probs)