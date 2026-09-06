# 🚦 Hệ thống Nhận diện Biển báo Giao thông Việt Nam (Traffic Sign Recognition System)

Hệ thống ứng dụng Trí tuệ nhân tạo (AI/Deep Learning) kết hợp với **Spring Boot** và **ONNX Runtime** để tự động nhận diện và phân loại các biển báo giao thông Việt Nam từ hình ảnh và video thực tế.

---

## 🌟 Tính năng nổi bật

- **Huấn luyện Mô hình AI (Deep Learning):** Sử dụng mạng nơ-ron tích chập (CNN) xây dựng trên PyTorch, tự động tăng cường dữ liệu (Data Augmentation) từ ảnh chụp thực tế.
- **Tự động Triển khai (Auto-Deployment):** Tự động xuất mô hình sang định dạng **ONNX** và đồng bộ trực tiếp sang tài nguyên của Backend Spring Boot.
- **Nhận diện Thời gian thực (Real-time Inference):** Tích hợp ONNX Runtime Java cho tốc độ phản hồi API cực nhanh (< 10ms).
- **Giao diện Đổi mới & Động (Dynamic Web UI):** Giao diện Modern Dark Mode tinh tế, tự động đồng bộ danh mục biển báo động từ Backend API.
- **Quản lý & Lưu vết:** Hỗ trợ lưu trữ nhật ký nhận diện và thống kê kết quả.

---

## 🛠️ Công nghệ sử dụng

| Phân hệ | Công nghệ / Thư viện |
| :--- | :--- |
| **AI / Machine Learning** | Python 3.10+, PyTorch, Torchvision, ONNX, ONNX Runtime, OpenCV, Pillow |
| **Backend API** | Java 21, Spring Boot 3.4.1, Maven, ONNX Runtime Java, H2 Database / MySQL |
| **Frontend Web UI** | HTML5, Vanilla JS, Modern CSS (Glassmorphism, Dark Theme) |

---

## 📂 Cấu trúc Dự án

```text
Traffic-Sign-Recognition/
├── ml-training/                # Phân hệ Huấn luyện AI & Xuất Model ONNX
│   ├── dataset_real/           # Thư mục chứa các bộ ảnh thực tế (00_ đến 14_)
│   ├── train_real.py           # Script train mô hình với ảnh thực tế
│   ├── class_mapping_vn.json   # Danh mục định nghĩa 15 biển báo tiếng Việt
│   └── requirements.txt        # Danh sách thư viện Python cần thiết
├── backend/                    # Phân hệ Backend Spring Boot (REST API)
│   ├── src/main/java/          # Source code Java (Controller, Service, Inference)
│   ├── src/main/resources/
│   │   ├── model/              # Nơi chứa file traffic_sign_model.onnx & class_mapping.json
│   │   └── application.yml     # Cấu hình ứng dụng
│   └── mvnw.cmd / mvnw         # Maven Wrapper
└── frontend/                   # Phân hệ Giao diện Người dùng Web
    ├── index.html              # Trang chủ giao diện
    ├── css/                    # Stylesheet giao diện
    └── js/app.js               # Logic tương tác API & hiển thị kết quả
```

---

## 📋 Danh sách 15 Biển báo Giao thông Hỗ trợ

| Class ID | Mã biển báo | Tên biển báo tiếng Việt | Loại biển |
| :---: | :---: | :--- | :---: |
| `00` | **W.201a** | Chỗ ngoặt nguy hiểm vòng bên trái | Biển cảnh báo |
| `01` | **W.201b** | Chỗ ngoặt nguy hiểm vòng bên phải | Biển cảnh báo |
| `02` | **W.207a** | Giao nhau với đường không ưu tiên | Biển cảnh báo |
| `03` | **W.208** | Giao nhau với đường ưu tiên | Biển cảnh báo |
| `04` | **W.224** | Đường người đi bộ cắt ngang | Biển cảnh báo |
| `05` | **W.245a** | Đi chậm | Biển cảnh báo |
| `06` | **P.102** | Cấm đi ngược chiều | Biển cấm |
| `07` | **P.103a** | Cấm xe ô tô | Biển cấm |
| `08` | **P.104** | Cấm xe mô tô, xe máy | Biển cấm |
| `09` | **P.123a** | Cấm rẽ trái | Biển cấm |
| `10` | **P.127-40** | Tốc độ tối đa 40 km/h | Biển cấm |
| `11` | **P.127-60** | Tốc độ tối đa 60 km/h | Biển cấm |
| `12` | **R.301a** | Hướng đi phải theo: Chỉ được đi thẳng | Biển hiệu lệnh |
| `13` | **R.302a** | Hướng đi phải theo: Đi vòng sang phía phải | Biển hiệu lệnh |
| `14` | **R.303** | Nơi giao nhau chạy theo vòng xuyến | Biển hiệu lệnh |

---

## 🚀 Hướng dẫn Chạy Demo (Step-by-Step Run Demo)

### 📌 Bước 1: Yêu cầu Môi trường (Prerequisites)
- **Java Development Kit (JDK):** JDK 21 trở lên.
- **Python:** Python 3.10 trở lên.
- **Git** (nếu cần clone dự án).

---

### 📌 Bước 2: Huấn luyện Mô hình AI (AI Training)

1. Mở cửa sổ Terminal/PowerShell và chuyển vào thư mục `ml-training`:
   ```powershell
   cd ml-training
   ```

2. Cài đặt các thư viện Python phụ thuộc:
   ```powershell
   python -m pip install -r requirements.txt
   ```

3. *(Tùy chọn)* Thêm các hình ảnh thực tế bạn tự chụp hoặc tải về vào các thư mục tương ứng trong `dataset_real/` (`00_...` đến `14_...`).

4. Chạy lệnh huấn luyện mô hình và tự động xuất mô hình ONNX:
   ```powershell
   python train_real.py
   ```
   > 💡 **Tự động hóa:** Script sẽ train mô hình 25 epochs, tạo file `traffic_sign_model.onnx` và tự động copy đè mô hình cùng file danh mục sang thư mục Backend.

---

### 📌 Bước 3: Khởi chạy Server Backend Spring Boot

1. Mở cửa sổ Terminal mới (hoặc `cd ..` rồi `cd backend`):
   ```powershell
   cd backend
   ```

2. Khởi chạy ứng dụng bằng Maven Wrapper:
   - **Trên Windows (PowerShell / Command Prompt):**
     ```powershell
     .\mvnw.cmd spring-boot:run
     ```
   - **Trên Linux / macOS / Git Bash:**
     ```bash
     ./mvnw spring-boot:run
     ```

3. Đợi cho đến khi màn hình Terminal xuất hiện dòng thông báo:
   ```text
   Started TrafficSignBackendApplication in X.XXX seconds
   ```
   *(Server đã sẵn sàng lắng nghe tại cổng `http://localhost:8080`)*

---

### 📌 Bước 4: Mở Giao diện Web & Trải nghiệm Demo

1. Mở trình duyệt web (Edge, Chrome, Firefox...) và truy cập vào địa chỉ:
   ```text
   http://localhost:8080
   ```

2. **Trải nghiệm các tính năng:**
   - 🖼️ **Nhận diện Ảnh:** Kéo thả hoặc tải lên ảnh biển báo giao thông để xem kết quả phân loại & độ tin cậy.
   - 🎥 **Nhận diện Video:** Tải lên video hành trình để AI nhận diện qua từng khung hình.
   - 📚 **Danh mục 15 Biển báo:** Xem toàn bộ danh sách biển báo được đồng bộ động từ Backend.
   - 📊 **Lịch sử & Thống kê:** Xem danh sách nhật ký nhận diện đã lưu.

---

## 📌 Lưu ý khi Test Ảnh Mới
- **Kích thước & Bố cục:** Nên cắt (crop) ảnh sát vào vùng chứa biển báo (biển báo chiếm khoảng 70-90% diện tích ảnh) để đạt độ chính xác cao nhất khi nén về độ phân giải $32 \times 32$.

---

## 📝 Giấy phép (License)
Dự án phục vụ mục đích học tập, nghiên cứu và làm demo đồ án môn học.
