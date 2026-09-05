# Continuity Ledger

## Current Phase: Sẵn sàng chuyển giao sang Phase 2 (Tích hợp Model Inference vào Spring Boot)

### Completed Work (Phase 1)
- Environment setup: Python 3.11 virtual environment with PyTorch, Torchvision, ONNX, ONNX Runtime.
- Dataset preparation: GTSRB (43 classes) downloaded and preprocessed with ImageNet normalization and data augmentation.
- Model architecture: 4-layer Conv2D + BatchNorm + ReLU + MaxPool + Dropout (`TrafficSignCNN`), shape NCHW [batch, 3, 32, 32].
- Model training: Checkpoint saved at `best_model.pth`.
- Benchmark evaluation: Achieved 98.38% test accuracy (12,425/12,630 correct) on the entire GTSRB test split. Full report generated in `evaluation_report.json`.
- ONNX export: Successfully exported to `traffic_sign_model.onnx` (8.97 MB) with dynamic batch axis. Validated structurally with ONNX checker and numerically against PyTorch (max diff < 3e-6).
- Class mapping: Generated `class_mapping.json` for all 43 classes with English and Vietnamese names and sign categories.
- Acceptance criteria verification: Tested via `verify_onnx.py` using ONNX Runtime.

### Next Immediate Steps (Phase 2 & Phase 0 completion)
1. Install/configure Maven toolchain on PATH.
2. Initialize Spring Boot project (`backend/`) with dependencies: Web, JPA, MySQL, ONNX Runtime.
3. Copy `traffic_sign_model.onnx` and `class_mapping.json` into `backend/src/main/resources/model/`.
4. Implement `ModelInferenceService` and `ClassMappingService` in Java.
5. Write Java unit test validating inference outputs match Python test results.
