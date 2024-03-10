from ultralytics import YOLO
import cv2
import torch
from yolov5.utils.general import non_max_suppression, scale_coords

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
names = ['Algal Leaf Spot', 'Leaf Blight', 'Leaf Spot', 'No Disease']

def predict_image(image_path):
    # Load the trained model
    model = YOLO('models/yolov8.yaml').load('models/yolov8n.pt')

    # Load the image
    im0 = cv2.imread(image_path)
    img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img = torch.from_numpy(img).float().to(device)  # Convert to torch tensor and send to device
    img = img.permute(2, 0, 1)  # Change shape from BHWC to BCHW
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img)[0]

    # Apply NMS
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results to the console
            for *xyxy, conf, cls in reversed(det):
                print(f'{names[int(cls)]} {conf:.2f}')

if __name__ == "__main__":
    image_path = 'kq/25.jpg'  # replace with your image path
    predict_image(image_path)
