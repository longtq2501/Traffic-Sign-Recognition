# Phase Plan: Hệ thống Nhận diện Biển báo Giao thông (CNN + Spring Boot)

> Tài liệu này dùng để feed cho AI Agent (Claude Code, Cursor, v.v.) thực hiện từng phase.
> Kiến trúc: **Monolith Spring Boot**, model CNN train bằng Python/Keras, export ONNX, inference trong Java qua ONNX Runtime.

---

## Phase 0 — Khởi tạo môi trường & cấu trúc project

**Mục tiêu:** Có project chạy được (empty state), toolchain sẵn sàng cho cả phần Python (train) và Java (backend).

**Việc cần làm:**
- [ ] Cài JDK 17+, Maven, MySQL (qua XAMPP hoặc cài riêng)
- [ ] Cài Python 3.9–3.10, tạo virtualenv riêng cho phần training
- [ ] Khởi tạo Spring Boot project (Spring Initializr) với dependencies: `spring-boot-starter-web`, `spring-boot-starter-data-jpa`, `mysql-connector-j`, `spring-boot-starter-websocket`
- [ ] Khởi tạo repo Git, cấu trúc thư mục:
  ```
  project-root/
  ├── ml-training/          # Python: train + export model
  │   ├── data/
  │   ├── train.py
  │   └── export_onnx.py
  ├── backend/               # Spring Boot app
  │   ├── src/main/java/...
  │   └── src/main/resources/model/   # nơi đặt file .onnx
  └── frontend/               # HTML/CSS/JS (hoặc Thymeleaf trong backend)
  ```
- [ ] Tạo file `.gitignore` (loại trừ `target/`, `*.onnx` lớn nếu cần, `venv/`, dataset raw)

**Deliverable:** Repo khởi tạo, `mvn spring-boot:run` chạy được trang mặc định, Python env kích hoạt được.

---

## Phase 1 — Chuẩn bị dữ liệu & huấn luyện mô hình CNN (Python)

**Mục tiêu:** Có model CNN đã train, đạt độ chính xác chấp nhận được, export sang ONNX.

**Việc cần làm:**
- [ ] Tải Kaggle Traffic Sign Dataset (GTSRB hoặc tương đương), giải nén vào `ml-training/data/`
- [ ] Viết script tiền xử lý: resize ảnh về kích thước cố định (ví dụ 32x32 hoặc 64x64), normalize pixel [0,1], chia train/val/test
- [ ] Data augmentation: xoay, lật, thay đổi độ sáng (dùng `ImageDataGenerator` hoặc `albumentations`)
- [ ] Thiết kế kiến trúc CNN (Conv2D → MaxPooling → Dropout → Dense → Softmax)
- [ ] Train model, theo dõi accuracy/loss qua các epoch, lưu checkpoint tốt nhất
- [ ] Đánh giá trên tập test, in confusion matrix / classification report
- [ ] Export model sang ONNX (`tf2onnx.convert` nếu dùng TensorFlow/Keras)
- [ ] Ghi lại: tên input/output tensor, kích thước input, thứ tự class (mapping index → tên biển báo) vào file `class_mapping.json`

**Deliverable:** File `traffic_sign_model.onnx` + `class_mapping.json`, báo cáo accuracy trên test set (≥ ngưỡng đặt ra, ví dụ 90%).

**Acceptance criteria:** Model load được bằng `onnxruntime` (Python) và cho output hợp lý trên vài ảnh test thủ công.

---

## Phase 2 — Tích hợp Model Inference vào Spring Boot

**Mục tiêu:** Backend load được ONNX model và chạy inference đúng, không lệch so với kết quả Python.

**Việc cần làm:**
- [ ] Thêm dependency `onnxruntime` vào `pom.xml`
- [ ] Copy `traffic_sign_model.onnx` và `class_mapping.json` vào `src/main/resources/model/`
- [ ] Viết `ModelInferenceService`: load session lúc khởi động (`@PostConstruct`), method `predict(float[][][][] input)` trả về class + confidence
- [ ] Viết `ClassMappingService`: đọc `class_mapping.json`, map index → tên biển báo
- [ ] Viết unit test: đưa 1 ảnh mẫu đã biết trước kết quả, so sánh output Java với output Python (đảm bảo preprocessing khớp nhau về thứ tự channel RGB/BGR, normalize)

**Deliverable:** Test case chứng minh inference Java cho kết quả khớp (hoặc gần khớp, sai số chấp nhận được) với Python.

**Lưu ý cho agent:** Đây là phase dễ lỗi ngầm nhất — sai thứ tự kênh màu (BGR vs RGB) hoặc sai kích thước input sẽ khiến model chạy nhưng dự đoán sai mà không báo lỗi.

---

## Phase 3 — Xử lý ảnh & video (Backend)

**Mục tiêu:** Backend nhận file ảnh/video, tiền xử lý đúng chuẩn đầu vào model.

**Việc cần làm:**
- [ ] Thêm dependency xử lý ảnh (`org.openpnp:opencv` hoặc `javacv-platform` nếu cần xử lý video)
- [ ] Viết `ImageProcessingService`: resize, normalize ảnh về đúng format model yêu cầu
- [ ] Viết `VideoProcessingService`: dùng JavaCV (`FFmpegFrameGrabber`) tách video thành frame rời rạc (có thể sample mỗi N frame để giảm tải, không cần xử lý toàn bộ)
- [ ] Viết `RecognitionOrchestratorService`: điều phối luồng ảnh/video → preprocessing → inference → format kết quả trả về

**Deliverable:** Có thể đưa 1 file ảnh và 1 file video test qua service, nhận về danh sách kết quả nhận diện hợp lệ.

---

## Phase 4 — Database & Persistence

**Mục tiêu:** Lưu lịch sử nhận diện và dữ liệu người dùng vào MySQL.

**Việc cần làm:**
- [ ] Thiết kế schema: bảng `users`, bảng `recognition_logs` (id, user_id, input_type [image/video/realtime], detected_sign, confidence_score, timestamp, file_reference)
- [ ] Tạo Entity + Repository (Spring Data JPA) tương ứng
- [ ] Cấu hình `application.properties`/`yml` kết nối MySQL (datasource URL, username, password, `spring.jpa.hibernate.ddl-auto`)
- [ ] Viết `RecognitionLogService`: lưu kết quả mỗi lần nhận diện

**Deliverable:** Chạy app, thực hiện 1 request nhận diện thử, kiểm tra record xuất hiện đúng trong MySQL.

---

## Phase 5 — REST API Endpoints

**Mục tiêu:** Expose các API cho 3 chế độ nhận diện.

**Việc cần làm:**
- [ ] `POST /api/recognize/image` — nhận file ảnh (multipart), trả JSON `{sign_name, confidence}`
- [ ] `POST /api/recognize/video` — nhận file video, trả danh sách kết quả theo từng frame/timestamp
- [ ] WebSocket endpoint `/ws/realtime` — nhận frame ảnh (base64/binary) liên tục từ client, trả kết quả từng frame theo thời gian thực
- [ ] `GET /api/logs` — trả lịch sử nhận diện (có thể lọc theo user, thời gian)
- [ ] Xử lý lỗi: file không hợp lệ, không tìm thấy biển báo trong ảnh, file quá lớn

**Deliverable:** Test toàn bộ API bằng Postman/curl, có collection lưu lại để demo.

---

## Phase 6 — Frontend (HTML/CSS/JS)

**Mục tiêu:** Giao diện cho phép người dùng thao tác cả 3 chế độ.

**Việc cần làm:**
- [ ] Trang upload ảnh: chọn file → gọi API `/api/recognize/image` → hiển thị kết quả + confidence
- [ ] Trang upload video: chọn file → gọi API `/api/recognize/video` → hiển thị danh sách kết quả theo thời gian
- [ ] Trang real-time: dùng `getUserMedia` truy cập webcam, capture frame định kỳ (ví dụ mỗi 200–500ms), gửi qua WebSocket, hiển thị kết quả overlay trực tiếp
- [ ] Trang lịch sử: hiển thị danh sách log từ `/api/logs`
- [ ] UI đồng nhất, thông báo lỗi rõ ràng khi không có biển báo trong ảnh

**Deliverable:** Demo end-to-end chạy được trên trình duyệt cho cả 3 chế độ.

---

## Phase 7 — Kiểm thử tổng thể

**Mục tiêu:** Đảm bảo hệ thống ổn định trước khi báo cáo/demo.

**Việc cần làm:**
- [ ] Test với ảnh/video đa dạng góc chụp, ánh sáng, chất lượng
- [ ] Test edge case: file không phải ảnh/video, ảnh không chứa biển báo, file quá lớn, mất kết nối webcam
- [ ] Đo thời gian phản hồi API (đặc biệt real-time — cần đủ nhanh để trải nghiệm mượt)
- [ ] Kiểm tra độ chính xác thực tế so với accuracy đã báo cáo ở Phase 1

**Deliverable:** Bảng kết quả test (pass/fail từng case), ghi nhận các hạn chế đã biết trước (theo tài liệu gốc: yêu cầu ảnh phải có biển báo hợp lệ, phụ thuộc camera thiết bị, không hỗ trợ chạy nền).

---

## Phase 8 — Đóng gói & Tài liệu báo cáo

**Mục tiêu:** Sẵn sàng nộp bài / demo trước lớp.

**Việc cần làm:**
- [ ] Viết README: hướng dẫn cài đặt, chạy project (backend + database)
- [ ] Chuẩn bị script/video demo cho từng chế độ
- [ ] Viết báo cáo theo cấu trúc SDLC Waterfall (đối chiếu với tài liệu gốc: bối cảnh, tech stack, kiến trúc, đánh giá ưu/nhược điểm)
- [ ] Chuẩn bị slide thuyết trình (nếu cần)

**Deliverable:** Bộ nộp bài hoàn chỉnh: source code, README, báo cáo, slide.

---

## Ghi chú chung cho AI Agent

- Thực hiện tuần tự theo phase (đúng tinh thần Waterfall của đồ án), nhưng **Phase 1 (train model) nên chốt sớm và kỹ** vì các phase sau phụ thuộc vào input/output shape của model.
- Ở mỗi phase, sau khi hoàn thành, agent nên tự kiểm tra lại "Deliverable" trước khi chuyển phase tiếp theo.
- Nếu phát sinh thay đổi ở Phase 1 sau khi đã làm Phase 2+ (ví dụ đổi kiến trúc CNN, đổi input size), cần cập nhật lại `ModelInferenceService` và `ImageProcessingService` tương ứng — đây là điểm yếu đã biết trước của mô hình Waterfall theo tài liệu gốc.