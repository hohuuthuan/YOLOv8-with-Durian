# from ultralytics import YOLO
# import cv2
# import matplotlib.pyplot as plt

# def detect_and_display(image_path):
#     # Load the model
#     model = YOLO('models/yolov8.yaml').load('models/yolov8n.pt')

#     # Perform detection
#     results = model(image_path)

#     # Save the image with bounding boxes and labels
#     results.save()  # Save the results

#     # Display the image
#     img = cv2.imread(results.files[0])  # Read the saved image
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert color space
#     plt.imshow(img)  # Display the image
#     plt.show()

# if __name__ == "__main__":
#     detect_and_display('kq/anhkiemthu.jpg')


from ultralytics import YOLO

def detect_and_display(image_path):
    # Load the model
    model = YOLO('models/yolov8.yaml').load('models/yolov8x.pt')

    # Perform detection
    results = model(image_path)

    # Print the labels of the detected objects
    for result in results:
        for box in result.boxes:
            print(f"Detected object: {model.names[int(box.cls)]}")

if __name__ == "__main__":
    detect_and_display('kq/anhkiemthu.jpg')


# from ultralytics import YOLO

# def detect_and_display(image_path):
#     # Load the model
#     model = YOLO('models/yolov8.yaml').load('models/yolov8n.pt')

#     # Perform detection
#     results = model(image_path)

#     # Print the labels of the detected objects
#     for result in results:
#         for box in result.boxes:
#             print(f"Detected object: {model.names[int(box.cls)]}")

# if __name__ == "__main__":
#     detect_and_display('kq/.jpg')
