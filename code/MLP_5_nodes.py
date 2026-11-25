import numpy as np

E = 2.718281828459045

# ----------------------------------------------------------
# Bước 1: Khởi tạo đầu vào dự đoán của app và đứa bạn:
X = np.array([
    [0.9, 0.1],
    [0.8, 0.2],
    [0.6, 0.4],
    [0.7, 0.3],
    [0.2, 0.8],
    [0.3, 0.7],
    [1.0, 1.0],
    [0.0, 0.0],
], dtype=np.float64)

# ----------------------------------------------------------
# Bước 2: Khởi tạo đầu ra lựa chọn mong muốn.
Yd = np.array([
    [0.0],
    [0.0],
    [1.0],
    [1.0],
    [0.0],
    [1.0],
    [1.0],
    [1.0],
], dtype=np.float64)

# ----------------------------------------------------------
# Bước 3: Định nghĩa hàm kích hoạt: 1) Sigmoid; 2) Tuyến tính.
def Sigmoid_activation(z):
    if z < -500.0:
        z = -500.0
    elif z > 500.0:
        z = 500.0
    return 1.0 / (1.0 + (E ** (-z)))

def tuyen_tinh_activation(z):
    return z

# ----------------------------------------------------------
# Bước 4: Định nghĩa đạo hàm kích hoạt:
def Sigmoid_activation_dao_ham(a):
    return a * (1.0 - a)

def tuyen_tinh_activation_dao_ham(n):
    return 1.0

# ----------------------------------------------------------
# Bước 5: Khởi tạo trọng số ban đầu và tham số học.
# Lớp ẩn F1 (nhận từ X1 và X2)
w_x1_f1 = 0.2
w_x2_f1 = -0.3

# Lớp ẩn F2 (nhận từ X1 và X2)
w_x1_f2 = 0.4
w_x2_f2 = 0.1

# Lớp đầu ra F-out (nhận từ F1 và F2)
w_f1_out = -0.5
w_f2_out = 0.6

# Khởi tạo tham số học
learning_rate = 0.1   # Tốc độ học (eta): độ lớn mỗi bước cập nhật trọng số
epochs = 20000         # Số lần duyệt qua toàn bộ 8 mẫu dữ liệu học

# ----------------------------------------------------------
# Bước 6: Định nghĩa hàm tính toán của node trong lan truyền thuận:
# Node ẩn F1 (Sigmoid): chỉ nhận x1, x2 và trọng số đi vào F1
def calc_f1(x1, x2, w_x1_f1, w_x2_f1):
    z_f1 = w_x1_f1 * x1 + w_x2_f1 * x2
    a_f1 = Sigmoid_activation(z_f1)
    return z_f1, a_f1

# Node ẩn F2 (Sigmoid): hàm copy riêng cho F2, chỉ nhận x1, x2 và trọng số đi vào F2
def calc_f2(x1, x2, w_x1_f2, w_x2_f2):
    z_f2 = w_x1_f2 * x1 + w_x2_f2 * x2
    a_f2 = Sigmoid_activation(z_f2)
    return z_f2, a_f2

# Node đầu ra F-out (Tuyến tính): chỉ nhận a_f1, a_f2 và trọng số nối từ lớp ẩn tới F-out.
def calc_f_out(a_f1, a_f2, w_f1_out, w_f2_out):
    z_out = w_f1_out * a_f1 + w_f2_out * a_f2
    yo = tuyen_tinh_activation(z_out)
    return z_out, yo

# ----------------------------------------------------------
# Bước 7: Định nghĩa hàm lan truyền thuận:
def lan_truyen_thuan(x1, x2, w_x1_f1, w_x2_f1, w_x1_f2, w_x2_f2, w_f1_out, w_f2_out):
    z_f1, a_f1 = calc_f1(x1, x2, w_x1_f1, w_x2_f1)
    z_f2, a_f2 = calc_f2(x1, x2, w_x1_f2, w_x2_f2)
    z_out, yo = calc_f_out(a_f1, a_f2, w_f1_out, w_f2_out)
    return z_f1, a_f1, z_f2, a_f2, z_out, yo

# ----------------------------------------------------------
# Bước 8a: Hàm tính mất mát L = 0.5 * (Yd - Yo)^2
def tinh_mat_mat(yd, yo):
    sai_so = yd - yo
    loss = 0.5 * (sai_so ** 2)
    return loss


# ----------------------------------------------------------
# Bước 8b: Lan truyền ngược cho node đầu ra F-out (Tuyến tính)
# Trả về đạo hàm theo w_f1_out, w_f2_out và delta_out để truyền ngược về F1, F2
def lan_truyen_nguoc_f_out(yd, yo, a_f1, a_f2):
    dao_ham_out = tuyen_tinh_activation_dao_ham(yo)
    delta_out = -(yd - yo) * dao_ham_out

    gradientient_w_f1_out = delta_out * a_f1
    gradientient_w_f2_out = delta_out * a_f2
    return delta_out, gradientient_w_f1_out, gradientient_w_f2_out


# ----------------------------------------------------------
# Bước 8c: Lan truyền ngược cho nhánh ẩn F1 (Sigmoid)
def lan_truyen_nguoc_f1(x1, x2, a_f1, w_f1_out, delta_out):
    dao_ham_f1 = Sigmoid_activation_dao_ham(a_f1)
    delta_f1 = (delta_out * w_f1_out) * dao_ham_f1

    gradient_w_x1_f1 = delta_f1 * x1
    gradient_w_x2_f1 = delta_f1 * x2
    return gradient_w_x1_f1, gradient_w_x2_f1


# ----------------------------------------------------------
# Bước 8d: Lan truyền ngược cho nhánh ẩn F2 (Sigmoid)
def lan_truyen_nguoc_f2(x1, x2, a_f2, w_f2_out, delta_out):
    dao_ham_f2 = Sigmoid_activation_dao_ham(a_f2)
    delta_f2 = (delta_out * w_f2_out) * dao_ham_f2

    gradient_w_x1_f2 = delta_f2 * x1
    gradient_w_x2_f2 = delta_f2 * x2
    return gradient_w_x1_f2, gradient_w_x2_f2


# ----------------------------------------------------------
# Bước 8e: Cập nhật toàn bộ 6 trọng số theo gradientient Descent
def cap_nhat_trong_so(
    w_x1_f1, w_x2_f1,
    w_x1_f2, w_x2_f2,
    w_f1_out, w_f2_out,
    gradient_w_x1_f1, gradient_w_x2_f1,
    gradient_w_x1_f2, gradient_w_x2_f2,
    gradient_w_f1_out, gradient_w_f2_out,
    lr
):
    w_x1_f1 = w_x1_f1 - lr * gradient_w_x1_f1
    w_x2_f1 = w_x2_f1 - lr * gradient_w_x2_f1

    w_x1_f2 = w_x1_f2 - lr * gradient_w_x1_f2
    w_x2_f2 = w_x2_f2 - lr * gradient_w_x2_f2

    w_f1_out = w_f1_out - lr * gradient_w_f1_out
    w_f2_out = w_f2_out - lr * gradient_w_f2_out

    return w_x1_f1, w_x2_f1, w_x1_f2, w_x2_f2, w_f1_out, w_f2_out

# ----------------------------------------------------------
# Bước 9: Huấn luyện mạng qua các epochs
for epoch in range(epochs):
    tong_loss = 0.0

    for i in range(len(X)):
        x1 = X[i][0]
        x2 = X[i][1]
        yd = Yd[i][0]

        # 1. Lan truyền thuận
        z_f1, a_f1, z_f2, a_f2, z_out, yo = lan_truyen_thuan(
            x1, x2,
            w_x1_f1, w_x2_f1,
            w_x1_f2, w_x2_f2,
            w_f1_out, w_f2_out
        )

        # 2. Tính mất mát
        loss = tinh_mat_mat(yd, yo)
        tong_loss = tong_loss + loss

        # 3. Lan truyền ngược từng nhánh
        delta_out, grad_w_f1_out, grad_w_f2_out = lan_truyen_nguoc_f_out(yd, yo, a_f1, a_f2)
        grad_w_x1_f1, grad_w_x2_f1 = lan_truyen_nguoc_f1(x1, x2, a_f1, w_f1_out, delta_out)
        grad_w_x1_f2, grad_w_x2_f2 = lan_truyen_nguoc_f2(x1, x2, a_f2, w_f2_out, delta_out)

        # 4. Cập nhật trọng số
        w_x1_f1, w_x2_f1, w_x1_f2, w_x2_f2, w_f1_out, w_f2_out = cap_nhat_trong_so(
            w_x1_f1, w_x2_f1,
            w_x1_f2, w_x2_f2,
            w_f1_out, w_f2_out,
            grad_w_x1_f1, grad_w_x2_f1,
            grad_w_x1_f2, grad_w_x2_f2,
            grad_w_f1_out, grad_w_f2_out,
            learning_rate
        )

    if (epoch + 1) % 100 == 0 or epoch == 0:
        trung_binh_loss = tong_loss / len(X)
        print(f"Epoch {epoch + 1:4d}/{epochs} | Loss trung bình: {trung_binh_loss:.6f}")

# ----------------------------------------------------------
# Kiểm tra đầu ra dự đoán Yo so với Yd sau khi huấn luyện
print("\n--- Kết quả sau huấn luyện ---")
for i in range(len(X)):
    x1 = X[i][0]
    x2 = X[i][1]
    yd = Yd[i][0]
    _, _, _, _, _, yo = lan_truyen_thuan(
        x1, x2,
        w_x1_f1, w_x2_f1,
        w_x1_f2, w_x2_f2,
        w_f1_out, w_f2_out
    )
    print(f"Mẫu {i+1}: X=({x1}, {x2}) | Yd={yd:.2f} | Yo={yo:.4f}")
