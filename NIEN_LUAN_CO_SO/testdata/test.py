import torch
from PIL import Image
from torchvision import transforms
from models.experimental import attempt_load
from utils.general import non_max_suppression

# Load model
model = attempt_load('path_to_your_model_weights.pt', map_location='cuda' if torch.cuda.is_available() else 'cpu')

# Prepare image
image = Image.open('path_to_your_image.jpg')
transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
image = transform(image).unsqueeze(0)

# Run model
with torch.no_grad():
    result = model(image)[0]

# Apply NMS
result = non_max_suppression(result, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False)

# Print result
for i, det in enumerate(result):  # detections per image
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(image.shape[2:], det[:, :4], im0.shape).round()

        # Print results to screen
        for *xyxy, conf, cls in reversed(det):
            print(f'{conf:.2f} {cls}')
