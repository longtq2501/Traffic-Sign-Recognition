package com.trafficsign.module.recognition.service.impl;

import com.trafficsign.common.dto.response.FrameDetectionResult;
import com.trafficsign.common.dto.response.SignPredictionResponse;
import com.trafficsign.common.dto.response.VideoRecognitionResponse;
import com.trafficsign.common.exception.ModelInferenceException;
import com.trafficsign.module.recognition.service.ImageProcessingService;
import com.trafficsign.module.recognition.service.ModelInferenceService;
import com.trafficsign.module.recognition.service.RecognitionOrchestratorService;
import com.trafficsign.module.recognition.service.VideoProcessingService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.awt.image.BufferedImage;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class RecognitionOrchestratorServiceImpl implements RecognitionOrchestratorService {

    private final ImageProcessingService imageProcessingService;
    private final VideoProcessingService videoProcessingService;
    private final ModelInferenceService modelInferenceService;

    @Override
    public SignPredictionResponse recognizeImage(InputStream imageStream) {
        log.debug("Initiating image recognition pipeline from InputStream");
        float[][][][] tensor = imageProcessingService.preprocessImage(imageStream);
        return modelInferenceService.predict(tensor);
    }

    @Override
    public SignPredictionResponse recognizeImage(byte[] imageBytes) {
        log.debug("Initiating image recognition pipeline from byte array (length: {})", imageBytes != null ? imageBytes.length : 0);
        float[][][][] tensor = imageProcessingService.preprocessImage(imageBytes);
        return modelInferenceService.predict(tensor);
    }

    @Override
    public SignPredictionResponse recognizeImage(BufferedImage image) {
        log.debug("Initiating image recognition pipeline from BufferedImage ({}x{})", image.getWidth(), image.getHeight());
        float[][][][] tensor = imageProcessingService.preprocessImage(image);
        return modelInferenceService.predict(tensor);
    }

    @Override
    public VideoRecognitionResponse recognizeVideo(InputStream videoStream, double sampleIntervalSeconds, double minConfidenceThreshold) {
        if (minConfidenceThreshold <= 0.0) {
            minConfidenceThreshold = 40.0; // Default 40% confidence threshold
        }
        long startProcessingTime = System.currentTimeMillis();
        log.info("Starting video recognition pipeline. Sample interval: {}s, Min threshold: {}%",
                sampleIntervalSeconds, minConfidenceThreshold);

        List<VideoProcessingService.SampledVideoFrame> frames =
                videoProcessingService.extractSampledFrames(videoStream, sampleIntervalSeconds);

        List<FrameDetectionResult> detections = new ArrayList<>();
        double maxTimestamp = 0.0;

        for (VideoProcessingService.SampledVideoFrame frame : frames) {
            try {
                float[][][][] tensor = imageProcessingService.preprocessImage(frame.image());
                SignPredictionResponse prediction = modelInferenceService.predict(tensor);

                if (frame.timestampSeconds() > maxTimestamp) {
                    maxTimestamp = frame.timestampSeconds();
                }

                if (prediction.getConfidence() >= minConfidenceThreshold) {
                    FrameDetectionResult detection = FrameDetectionResult.builder()
                            .frameNumber(frame.frameNumber())
                            .timestampSeconds(Math.round(frame.timestampSeconds() * 100.0) / 100.0)
                            .prediction(prediction)
                            .build();
                    detections.add(detection);
                }
            } catch (Exception e) {
                log.warn("Failed to process frame #{} at {}s: {}", frame.frameNumber(), frame.timestampSeconds(), e.getMessage());
            }
        }

        long totalElapsed = System.currentTimeMillis() - startProcessingTime;
        log.info("Video recognition pipeline completed. Processed {} frames, found {} detections in {}ms.",
                frames.size(), detections.size(), totalElapsed);

        return VideoRecognitionResponse.builder()
                .totalDurationSeconds(Math.round(maxTimestamp * 100.0) / 100.0)
                .totalFramesProcessed(frames.size())
                .detections(detections)
                .processingTimeMs(totalElapsed)
                .build();
    }
}
