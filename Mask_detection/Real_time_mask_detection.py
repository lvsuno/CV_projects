#%%
from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2

model = YOLO("/Users/elvist/PycharmProjects/CV Projects/Mask_detection/runs/detect/train2/weights/best.onnx")  # load a pretrained model (recommended for training)
# %%
model.predict(source="0",imgsz=512, show=True, stream=True)
# %%
