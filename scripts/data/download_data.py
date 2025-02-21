from roboflow import Roboflow
rf = Roboflow(api_key="6YMhS2gNiKaJrMUXzZ3J")
project = rf.workspace("roboflow-gw7yv").project("self-driving-car")
version = project.version(3)
dataset = version.download("coco", location=r"/home/ubuntu/AI/DATA_SOURCE/Self_Driving_Car.v3-fixed-small.coco/export/coco_json")