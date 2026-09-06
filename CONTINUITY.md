# Continuity Ledger

## Current Phase: Hoàn thành Huấn luyện 15 Biển báo Việt Nam & Tự động Triển khai Model ONNX

### Completed Work
- Cập nhật bộ 15 biển báo giao thông Việt Nam thực tế trong `ml-training/dataset_real` và `class_mapping_vn.json`.
- Cập nhật script `train_real.py`: Tự động Augmentation dữ liệu, khắc phục lỗi DataLoader `drop_last=True`, xuất mô hình ONNX và tự động copy đè sang tài nguyên Backend Spring Boot.
- Đồng bộ hóa động dữ liệu 15 biển báo từ Spring Boot API `/api/recognize/classes` sang giao diện Frontend Web UI (`app.js` & `index.html`).
- Tạo file [README.md](file:///d:/long-personal-project/Traffic-Sign-Recognition/README.md) đầy đủ thông tin giới thiệu dự án, bảng 15 biển báo và hướng dẫn từng bước chạy Demo (train AI & chạy Spring Boot).

### Next Immediate Steps
1. Tiếp tục thu thập & bổ sung thêm ảnh thực tế đa dạng cho các lớp biển báo.
2. Nâng cấp kĩ thuật Random Margin/Padding Augmentation trong `train_real.py` để tối ưu nhận diện ảnh chưa qua crop sát.
3. Hoàn thiện Phase 4: Persistence nhật ký nhận diện vào MySQL/H2 Database.
