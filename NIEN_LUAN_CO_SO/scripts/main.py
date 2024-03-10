from ultralytics import YOLO
from PIL import Image
model = YOLO('../models/yolov8.yaml').load('../models/yolov8n.pt')

results = model('E:/NIEN_LUAN_CO_SO/kq/anhkiemthu.jpg')

for r in results:
    print(r.boxes)
    im_array = r.plot()
    im = Image.fromarray(im_array[..., ::-1])
    im.show()
    im.save('ketqua.jpg')
    
