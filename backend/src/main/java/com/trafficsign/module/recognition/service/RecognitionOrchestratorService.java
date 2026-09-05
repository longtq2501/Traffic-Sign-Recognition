package com.trafficsign.module.recognition.service;

import com.trafficsign.common.dto.response.SignPredictionResponse;
import com.trafficsign.common.dto.response.VideoRecognitionResponse;

import java.awt.image.BufferedImage;
import java.io.InputStream;

public interface RecognitionOrchestratorService {

    /**
     * Orchestrates end-to-end recognition for an image input stream.
     */
    SignPredictionResponse recognizeImage(InputStream imageStream);

    /**
     * Orchestrates end-to-end recognition for raw image bytes.
     */
    SignPredictionResponse recognizeImage(byte[] imageBytes);

    /**
     * Orchestrates end-to-end recognition for a BufferedImage (e.g., from Webcam frame).
     */
    SignPredictionResponse recognizeImage(BufferedImage image);

    /**
     * Orchestrates recognition for an uploaded video file with frame sampling and filtering.
     *
     * @param videoStream Stream of video data
     * @param sampleIntervalSeconds Sampling interval in seconds (e.g., 0.5s for 2 frames/sec)
     * @param minConfidenceThreshold Minimum confidence threshold to record detection (e.g., 40.0%)
     * @return Aggregated video recognition response
     */
    VideoRecognitionResponse recognizeVideo(InputStream videoStream, double sampleIntervalSeconds, double minConfidenceThreshold);
}
