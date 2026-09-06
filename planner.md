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
- [x] Tải Kaggle Traffic Sign Dataset (GTSRB hoặc tương đương), giải nén vào `ml-training/data/`
- [x] Viết script tiền xử lý: resize ảnh về kích thước cố định (32x32), normalize pixel ImageNet, chia train/val/test
- [x] Data augmentation: xoay, thay đổi độ sáng, contrast (`transforms.Compose`)
- [x] Thiết kế kiến trúc CNN (Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
- [x] Train model, theo dõi accuracy/loss qua các epoch, lưu checkpoint tốt nhất (`best_model.pth`, 98.38% test accuracy)
- [x] Đánh giá trên tập test, in confusion matrix / classification report (`evaluation_report.json`)
- [x] Export model sang ONNX (`torch.onnx.export` cho PyTorch, format NCHW `[batch_size, 3, 32, 32]` -> `traffic_sign_model.onnx`)
- [x] Ghi lại: tên input/output tensor, kích thước input, thứ tự class (mapping index → tên biển báo) vào file `class_mapping.json`

**Deliverable:** File `traffic_sign_model.onnx` + `class_mapping.json`, báo cáo accuracy trên test set (đạt 98.38% > 90%).

**Acceptance criteria:** Đã nghiệm thu thành công qua `verify_onnx.py` load model bằng `onnxruntime` (Python) và nhận diện chính xác các ảnh test GTSRB.

---

## Phase 2 — Tích hợp Model Inference vào Spring Boot

**Mục tiêu:** Backend load được ONNX model và chạy inference đúng, không lệch so với kết quả Python.

**Việc cần làm:**
- [x] Thêm dependency `onnxruntime` vào `pom.xml`
- [x] Copy `traffic_sign_model.onnx` và `class_mapping.json` vào `src/main/resources/model/`
- [x] Viết `ModelInferenceService` (Interface + Impl): load session lúc khởi động (`@PostConstruct`), method `predict(float[][][][] input)` trả về class + confidence
- [x] Viết `ClassMappingService` (Interface + Impl): đọc `class_mapping.json`, map index → tên biển báo song ngữ
- [x] Viết unit test: `ModelInferenceServiceTest` kiểm thử load session, tra cứu 43 class và inference thành công (8ms) với cấu trúc tensor NCHW `[1, 3, 32, 32]`

**Deliverable:** Test case chứng minh inference Java nạp model ONNX và cho kết quả chuẩn xác đạt 4/4 test passed.

**Lưu ý cho agent:** Đây là phase dễ lỗi ngầm nhất — sai thứ tự kênh màu (BGR vs RGB) hoặc sai thứ tự tensor input (`NCHW`: `float[1][3][32][32]` của PyTorch thay vì `NHWC`: `float[1][32][32][3]` của Keras) sẽ khiến model chạy mà dự đoán sai hoàn toàn. Java side PHẢI build tensor theo `NCHW` `[1, 3, 32, 32]`.

---

## Phase 3 — Xử lý ảnh & video (Backend)

**Mục tiêu:** Backend nhận file ảnh/video, tiền xử lý đúng chuẩn đầu vào model.

**Việc cần làm:**
- [x] Thêm dependency xử lý video `org.bytedeco:javacv` và `ffmpeg` (với cấu hình platform windows-x86_64) vào `pom.xml`
- [x] Viết `ImageProcessingService` (Interface + Impl): resize 32x32 Bilinear, chuẩn hóa ImageNet theo NCHW `[1, 3, 32, 32]`
- [x] Viết `VideoProcessingService` (Interface + Impl): dùng JavaCV `FFmpegFrameGrabber` tách frame lấy mẫu theo chu kỳ thời gian linh hoạt
- [x] Viết `RecognitionOrchestratorService` (Interface + Impl): điều phối chuỗi `Image/Video -> Preprocessing -> ModelInference -> DTO Response`

**Deliverable:** Toàn bộ test suite 9/9 tests passed chứng minh pipeline xử lý ảnh và video hoạt động mượt mà.

---

## Phase 4 — Database & Persistence

**Mục tiêu:** Lưu lịch sử nhận diện và dữ liệu người dùng vào MySQL.

**Việc cần làm:**
- [x] Thiết kế schema: bảng `users`, bảng `recognition_logs` (id, user_id, input_type [image/video/realtime], detected_sign, confidence_score, timestamp, file_reference)
- [x] Tạo Entity + Repository (Spring Data JPA) tương ứng tuân thủ Clean Code rules (không dùng @Data trên Entity, Indexing, Interface/Impl)
- [x] Cấu hình `application.yml` kết nối MySQL (`traffic_sign_db`) và cấu hình test H2 in-memory
- [x] Viết `RecognitionLogService` (Interface + Impl) và DTOs tương ứng: lưu log, phân trang, lọc theo inputType, thống kê tổng quan
- [x] Viết unit & integration test `RecognitionLogServiceTest` (13/13 tests toàn backend pass sạch)

**Deliverable:** Tầng persistence hoàn chỉnh, sẵn sàng tích hợp vào REST API ở Phase 5.

---

## Phase 5 — REST API Endpoints

**Mục tiêu:** Expose các API cho 3 chế độ nhận diện.

**Việc cần làm:**
- [x] `POST /api/recognize/image` — nhận file ảnh (multipart), tiền xử lý và inference qua model ONNX, tự động lưu log vào database, trả JSON `{logId, classId, signNameEn, signNameVi, category, confidence, inferenceTimeMs, fileName}`
- [x] `POST /api/recognize/video` — nhận file video, trích xuất frame theo interval, lọc ngưỡng confidence, ghi log vào database, trả danh sách kết quả theo timeline
- [x] WebSocket endpoint `/ws/realtime` — nhận frame ảnh base64 liên tục từ webcam client, trả kết quả từng frame thời gian thực (độ trễ 1-2ms)
- [x] `GET /api/recognize/classes` — trả danh mục 43 biển báo song ngữ phục vụ hiển thị tra cứu
- [x] `GET /api/logs` & `GET /api/logs/recent` & `GET /api/logs/stats` — xem lịch sử nhận diện có phân trang, lọc theo inputType và phân tích số liệu thống kê
- [x] Xử lý lỗi tập trung qua `@RestControllerAdvice`: file rỗng, sai định dạng ảnh/video, file quá lớn (>50MB), lỗi suy luận model
- [x] Cấu hình CORS mở rộng cho phép frontend kết nối trực tiếp không bị chặn

**Deliverable:** Toàn bộ API Backend AI (Image, Video, Realtime WebSocket, Logs, Stats, Classes) hoàn tất, kiểm thử tự động 22/22 tests PASS 100%.

---

## Phase 6 — Frontend (HTML/CSS/JS)

**Mục tiêu:** Giao diện cho phép người dùng thao tác cả 3 chế độ.

**Việc cần làm:**
- [x] Trang upload ảnh: kéo thả/chọn file ảnh → gọi API `/api/recognize/image` → hiển thị tên biển báo (Việt/Anh), nhóm, thanh confidence bar và thời gian suy luận
- [x] Trang upload video: chọn file video → gọi API `/api/recognize/video` → hiển thị bảng timeline biển báo phát hiện theo từng giây
- [x] Trang real-time: dùng `getUserMedia` truy cập webcam, capture frame mỗi 250ms gửi qua WebSocket `/ws/realtime`, hiển thị HUD overlay kết quả trực tiếp với latency 1-2ms
- [x] Trang lịch sử: xem bảng nhật ký nhận diện từ `/api/logs` có phân trang, lọc theo inputType và các thẻ thống kê tổng quan từ `/api/logs/stats`
- [x] Trang tra cứu danh mục: hiển thị lưới 43 biển báo giao thông GTSRB song ngữ từ `/api/recognize/classes` kèm ô tìm kiếm
- [x] Giao diện đồng nhất, thẩm mỹ hiện đại (Inter font, dark theme, responsive), cấu trúc Spring Boot static welcome page tự động phục vụ tại `http://localhost:8080/` và thư mục `frontend/`

**Deliverable:** Demo end-to-end hoàn chỉnh sẵn sàng trình diễn cho giáo viên xem trên trình duyệt cho cả 3 chế độ.

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