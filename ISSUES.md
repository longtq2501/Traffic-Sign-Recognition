# Traffic Sign Recognition - Issues and Optimization

## Current Issues and Next Actions

- [ ] [P1-High] MySQL Database & Entity persistence (Phase 4)
    - Description: Setup MySQL connection and Spring Data JPA entities for recognition logs and history.
    - Target: RecognitionLog entity, repository, and service with unit tests.
    - Metrics: Successful DB migration and CRUD operations.

- [ ] [P1-High] REST API Endpoints & Multipart Upload (Phase 5)
    - Description: Expose REST endpoints for image and video upload, plus WebSocket for realtime.
    - Target: Controllers for /api/recognize/image and /api/recognize/video.
    - Metrics: Functional API endpoints responding in JSON format.

---

## Completed Work (Archive)
- [x] [P1-High] Image & Video Preprocessing Pipeline (Phase 3)
    - Achieved: ImageProcessingService, VideoProcessingService (JavaCV FFmpeg), and RecognitionOrchestratorService passing 9/9 JUnit tests.
- [x] [P1-High] Spring Boot ONNX Runtime Inference Service (Phase 2)
    - Achieved: Loaded traffic_sign_model.onnx and class_mapping.json in Java, 8ms latency per tensor.
- [x] [P0-Critical] Maven environment and Spring Boot project initialization (Phase 0)
    - Achieved: Spring Boot 3.3.5 with Maven Wrapper initialized.
- [x] [P0-Critical] Python environment setup and ONNX export workflow (Phase 1)
    - Achieved: Exported traffic_sign_model.onnx with dynamic batch dimension, verified via onnx.checker and ONNX Runtime.
- [x] [P1-High] CNN Accuracy on Traffic Sign Benchmark (Phase 1)
    - Achieved: 98.38% test accuracy on 12,630 GTSRB test images (best_model.pth).
- [x] [P2-Medium] Ignore large dataset and virtual environment files in .gitignore
    - Achieved: .gitignore configured for venv, data, temporary artifacts while tracking essential model assets.
