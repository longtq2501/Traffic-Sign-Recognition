package com.trafficsign.module.recognition.controller;

import com.trafficsign.common.dto.request.CreateRecognitionLogRequest;
import com.trafficsign.common.dto.response.*;
import com.trafficsign.common.exception.ModelInferenceException;
import com.trafficsign.module.persistence.model.entity.InputType;
import com.trafficsign.module.persistence.service.RecognitionLogService;
import com.trafficsign.module.recognition.service.ClassMappingService;
import com.trafficsign.module.recognition.service.RecognitionOrchestratorService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.util.Comparator;
import java.util.Map;

@Slf4j
@RestController
@RequestMapping("/api/recognize")
@RequiredArgsConstructor
public class RecognitionController {

    private final RecognitionOrchestratorService recognitionOrchestratorService;
    private final RecognitionLogService recognitionLogService;
    private final ClassMappingService classMappingService;

    @PostMapping(value = "/image", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<ImageRecognitionResponse>> recognizeImage(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "userId", required = false) Long userId) {

        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded image file cannot be empty");
        }

        String contentType = file.getContentType();
        if (contentType != null && !contentType.startsWith("image/")) {
            throw new IllegalArgumentException("Invalid file format. Please upload an image file (JPEG, PNG, etc.)");
        }

        log.info("Received image recognition request: {} (size: {} bytes)", file.getOriginalFilename(), file.getSize());

        try (InputStream inputStream = file.getInputStream()) {
            SignPredictionResponse prediction = recognitionOrchestratorService.recognizeImage(inputStream);

            // Persist recognition result to database
            CreateRecognitionLogRequest logRequest = CreateRecognitionLogRequest.builder()
                    .userId(userId)
                    .inputType(InputType.IMAGE)
                    .signClassId(prediction.getClassId())
                    .detectedSignEn(prediction.getSignNameEn())
                    .detectedSignVi(prediction.getSignNameVi())
                    .category(prediction.getCategory())
                    .confidenceScore(prediction.getConfidence())
                    .inferenceTimeMs(prediction.getInferenceTimeMs())
                    .fileReference(file.getOriginalFilename())
                    .build();

            RecognitionLogResponse savedLog = recognitionLogService.saveLog(logRequest);

            ImageRecognitionResponse response = ImageRecognitionResponse.builder()
                    .logId(savedLog.getId())
                    .classId(prediction.getClassId())
                    .signNameEn(prediction.getSignNameEn())
                    .signNameVi(prediction.getSignNameVi())
                    .category(prediction.getCategory())
                    .confidence(prediction.getConfidence())
                    .inferenceTimeMs(prediction.getInferenceTimeMs())
                    .fileName(file.getOriginalFilename())
                    .build();

            return ResponseEntity.ok(ApiResponse.success("Image recognition completed successfully", response));

        } catch (IOException e) {
            log.error("Failed to read uploaded image stream", e);
            throw new ModelInferenceException("Failed to read image file: " + e.getMessage(), e);
        }
    }

    @PostMapping(value = "/video", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<ApiResponse<VideoRecognitionResponse>> recognizeVideo(
            @RequestParam("file") MultipartFile file,
            @RequestParam(value = "sampleIntervalSeconds", defaultValue = "0.5") Double sampleIntervalSeconds,
            @RequestParam(value = "minConfidenceThreshold", defaultValue = "0.40") Double minConfidenceThreshold,
            @RequestParam(value = "userId", required = false) Long userId) {

        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Uploaded video file cannot be empty");
        }

        log.info("Received video recognition request: {} (size: {} bytes, interval: {}s)",
                file.getOriginalFilename(), file.getSize(), sampleIntervalSeconds);

        try (InputStream inputStream = file.getInputStream()) {
            VideoRecognitionResponse videoResult = recognitionOrchestratorService.recognizeVideo(
                    inputStream, sampleIntervalSeconds, minConfidenceThreshold);

            // Persist video recognition summary to database if any sign detected
            if (videoResult.getDetections() != null && !videoResult.getDetections().isEmpty()) {
                FrameDetectionResult topDetection = videoResult.getDetections().stream()
                        .filter(d -> d.getPrediction() != null)
                        .max(Comparator.comparingDouble(d -> d.getPrediction().getConfidence()))
                        .orElse(null);

                if (topDetection != null && topDetection.getPrediction() != null) {
                    SignPredictionResponse pred = topDetection.getPrediction();
                    CreateRecognitionLogRequest logRequest = CreateRecognitionLogRequest.builder()
                            .userId(userId)
                            .inputType(InputType.VIDEO)
                            .signClassId(pred.getClassId())
                            .detectedSignEn(pred.getSignNameEn())
                            .detectedSignVi(pred.getSignNameVi())
                            .category(pred.getCategory())
                            .confidenceScore(pred.getConfidence())
                            .inferenceTimeMs(videoResult.getProcessingTimeMs())
                            .fileReference(file.getOriginalFilename())
                            .build();
                    recognitionLogService.saveLog(logRequest);
                }
            }

            return ResponseEntity.ok(ApiResponse.success("Video recognition completed successfully", videoResult));

        } catch (IOException e) {
            log.error("Failed to read uploaded video stream", e);
            throw new ModelInferenceException("Failed to read video file: " + e.getMessage(), e);
        }
    }

    @GetMapping("/classes")
    public ResponseEntity<ApiResponse<Map<Integer, SignClassInfo>>> getAllTrafficSignClasses() {
        Map<Integer, SignClassInfo> classes = classMappingService.getAllClasses();
        return ResponseEntity.ok(ApiResponse.success("Traffic sign classes retrieved successfully", classes));
    }
}
