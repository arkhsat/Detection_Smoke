from ultralytics import YOLO
from roboflow import Roboflow

model = YOLO('yolo8n.pt')

rf = Roboflow(api_key="drop your api_key here")
project = rf.workspace("train-qjr0z").project("detected-b7rrr")
version = project.version(1)
dataset = version.download("yolov8")
