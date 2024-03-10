import os

def check_access(path):
    # Kiểm tra xem thư mục có tồn tại không
    if not os.path.exists(path):
        return "Thư mục không tồn tại."

    # Kiểm tra quyền đọc
    if not os.access(path, os.R_OK):
        return "Không có quyền đọc thư mục."

    # Kiểm tra quyền ghi
    if not os.access(path, os.W_OK):
        return "Không có quyền ghi vào thư mục."

    return "Có đủ quyền truy cập thư mục."

print(check_access('./kq'))
