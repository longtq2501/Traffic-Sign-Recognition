# Continuity Ledger

## Current Phase: Hoan thanh Phase 3 - San sang chuyen giao sang Phase 4 (Database & Persistence)

### Completed Work (Phase 3: Xu ly anh & video Backend)
- Configured `javacv` 1.5.11 and `ffmpeg` 7.1-1.5.11 with Windows classifier in `pom.xml`.
- Created DTOs `FrameDetectionResult` and `VideoRecognitionResponse`.
- Implemented `ImageProcessingService` and `ImageProcessingServiceImpl` converting any `BufferedImage`, byte array, or `InputStream` into normalized NCHW `float[1][3][32][32]` tensor matching PyTorch ImageNet normalization.
- Implemented `VideoProcessingService` and `VideoProcessingServiceImpl` extracting sampled frames with timestamps using `FFmpegFrameGrabber`.
- Implemented `RecognitionOrchestratorService` and `RecognitionOrchestratorServiceImpl` coordinating the end-to-end flow: Image/Video -> Preprocessing -> ONNX Inference -> DTO Response.
- Added comprehensive unit and integration tests (`ImageProcessingServiceTest`, `RecognitionOrchestratorServiceTest`): Total 9/9 tests passed!

### Next Immediate Steps (Phase 4: Database & Persistence)
1. Add Spring Data JPA and MySQL Connector dependencies to `pom.xml`.
2. Configure MySQL datasource and JPA properties in `application.yml`.
3. Design Entities: `User` and `RecognitionLog` (storing input type, detected sign, confidence score, timestamp, file reference).
4. Implement Repositories and `RecognitionLogService` (Interface + Impl) for history persistence.
