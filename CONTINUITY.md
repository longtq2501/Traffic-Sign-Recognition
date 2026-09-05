# Continuity Ledger

## Current Phase: Hoan thanh Phase 2 - San sang chuyen giao sang Phase 3 (Xu ly anh & video Backend)

### Completed Work (Phase 2 & Phase 0 Initialization)
- Spring Boot 3.3.5 project initialized in `backend/` with Java 17 and Maven Wrapper (`mvnw.cmd`).
- ONNX Runtime Java 1.18.0 configured in `pom.xml`.
- Embedded `traffic_sign_model.onnx` and `class_mapping.json` in `backend/src/main/resources/model/`.
- Implemented `ClassMappingService` and `ClassMappingServiceImpl` according to Spring clean code guidelines (Interface + Impl separation, `@RequiredArgsConstructor`, logging).
- Implemented `ModelInferenceService` and `ModelInferenceServiceImpl` using Microsoft ONNX Runtime Java (`OrtEnvironment`, `OrtSession`, Softmax probabilities, inference timing).
- Created response DTOs (`SignPredictionResponse`, `SignClassInfo`) and custom domain exception (`ModelInferenceException`).
- Verified via `ModelInferenceServiceTest`: 4/4 tests passed with 8ms inference time per tensor.

### Next Immediate Steps (Phase 3: Xu ly anh & video Backend)
1. Add image processing dependency (`org.openpnp:opencv` or Java standard `BufferedImage`).
2. Implement `ImageProcessingService` to convert uploaded MultipartFile to NCHW `float[1][3][32][32]` tensor matching Python ImageNet normalization (`mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`).
3. Implement `VideoProcessingService` using JavaCV / FFmpegFrameGrabber for frame sampling.
4. Implement `RecognitionOrchestratorService` orchestrating Image/Video -> Preprocessing -> Inference -> Response.
