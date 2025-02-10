from ultralytics import YOLO
from roboflow import Roboflow

model = YOLO('yolo8n.pt')

rf = Roboflow(api_key="8XE96Iv5aXEqAqX5pqTF")
project = rf.workspace("train-qjr0z").project("detected-b7rrr")
version = project.version(1)
dataset = version.download("yolov8")
