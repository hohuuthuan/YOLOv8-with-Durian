from ultralytics import YOLO

def train_model():
    # Build a new model from YAML and load weights
    model = YOLO('../models/yolov8.yaml').load('../models/yolov8n.pt')

    # import shutil

    # Train the model
    results = model.train(data='../data/data.yaml', epochs=10, imgsz=640)

    # Move the results to the desired directory
    # shutil.move('C:/Users/hohuu/runs/detect', '../results')

if __name__ == "__main__":
    train_model()
