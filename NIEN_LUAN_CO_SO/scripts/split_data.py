import os
import shutil
import numpy as np

# Đường dẫn đến thư mục chứa dữ liệu
data_dir = 'E:/NIEN_LUAN_CO_SO/data/test'

# Tạo thư mục cho dữ liệu kiểm thử
os.makedirs('E:/NIEN_LUAN_CO_SO/data/valid/images', exist_ok=True)
os.makedirs('E:/NIEN_LUAN_CO_SO/data/valid/labels', exist_ok=True)

# Lấy danh sách tất cả các tệp trong thư mục dữ liệu
files = os.listdir(os.path.join(data_dir, 'images'))

# Chọn ngẫu nhiên 20% tệp để sử dụng làm dữ liệu kiểm thử
test_files = np.random.choice(files, size=int(len(files) * 0.5), replace=False)

# Di chuyển các tệp đã chọn sang thư mục kiểm thử
for file in test_files:
    # Di chuyển tệp hình ảnh
    shutil.move(os.path.join(data_dir, 'images', file), 'E:/NIEN_LUAN_CO_SO/data/valid/images')

    # Thay đổi phần mở rộng của tệp từ .jpg sang .txt
    label_file = os.path.splitext(file)[0] + '.txt'

    # Di chuyển tệp nhãn
    shutil.move(os.path.join(data_dir, 'labels', label_file), 'E:/NIEN_LUAN_CO_SO/data/valid/labels')
