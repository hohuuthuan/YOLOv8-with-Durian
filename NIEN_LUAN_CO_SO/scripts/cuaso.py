import tkinter as tk
from tkinter import filedialog
import cv2

# Định nghĩa hàm để tải lên hình ảnh và thực hiện mô hình YOLOv8
def upload_image_and_detect_objects():
    # Mở cửa sổ chọn tập tin để người dùng tải lên hình ảnh
    file_path = filedialog.askopenfilename()
    
    # Tải mô hình YOLOv8 đã được huấn luyện
    net = cv2.dnn.readNet('yolov8.weights', 'yolov8.cfg')
    
    # Đọc hình ảnh
    img = cv2.imread(file_path)
    
    # Thực hiện mô hình YOLOv8 trên hình ảnh
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    # Xử lý kết quả
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Lấy thông tin về hộp giới hạn và lớp của đối tượng
                class_ids.append(class_id)
    
    # In tên lớp của các đối tượng được phát hiện
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    for class_id in class_ids:
        print(classes[class_id])

# Tạo GUI
root = tk.Tk()
upload_button = tk.Button(root, text="Upload Image", command=upload_image_and_detect_objects)
upload_button.pack()
root.mainloop()
