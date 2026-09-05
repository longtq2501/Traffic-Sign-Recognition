package com.trafficsign.module.recognition.service.impl;

import com.trafficsign.common.exception.ModelInferenceException;
import com.trafficsign.module.recognition.service.VideoProcessingService;
import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacv.FFmpegFrameGrabber;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.springframework.stereotype.Service;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;

@Slf4j
@Service
public class VideoProcessingServiceImpl implements VideoProcessingService {

    @Override
    public List<SampledVideoFrame> extractSampledFrames(InputStream videoInputStream, double sampleIntervalSeconds) {
        if (videoInputStream == null) {
            throw new ModelInferenceException("Video input stream cannot be null");
        }

        File tempFile = null;
        try {
            tempFile = Files.createTempFile("traffic_sign_vid_", ".tmp").toFile();
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = videoInputStream.read(buffer)) != -1) {
                    fos.write(buffer, 0, bytesRead);
                }
            }
            return extractSampledFrames(tempFile, sampleIntervalSeconds);
        } catch (Exception e) {
            log.error("Failed to process video input stream", e);
            throw new ModelInferenceException("Failed to decode video: " + e.getMessage(), e);
        } finally {
            if (tempFile != null && tempFile.exists()) {
                boolean deleted = tempFile.delete();
                if (!deleted) {
                    tempFile.deleteOnExit();
                }
            }
        }
    }

    @Override
    public List<SampledVideoFrame> extractSampledFrames(File videoFile, double sampleIntervalSeconds) {
        if (videoFile == null || !videoFile.exists()) {
            throw new ModelInferenceException("Video file does not exist");
        }
        if (sampleIntervalSeconds <= 0.0) {
            sampleIntervalSeconds = 0.5; // Default to 2 frames per second
        }

        List<SampledVideoFrame> sampledFrames = new ArrayList<>();
        long sampleIntervalUs = (long) (sampleIntervalSeconds * 1_000_000L);

        log.info("Starting video frame extraction for: {} (interval: {}s)", videoFile.getName(), sampleIntervalSeconds);

        try (FFmpegFrameGrabber grabber = new FFmpegFrameGrabber(videoFile)) {
            grabber.start();

            long totalDurationUs = grabber.getLengthInTime();
            log.info("Video length: {}s, Resolution: {}x{}, Format: {}",
                    totalDurationUs / 1_000_000.0, grabber.getImageWidth(), grabber.getImageHeight(), grabber.getFormat());

            Java2DFrameConverter converter = new Java2DFrameConverter();
            Frame frame;
            long lastSampleTimeUs = -1;
            int frameIndex = 0;

            while ((frame = grabber.grabImage()) != null) {
                long currentTimestampUs = grabber.getTimestamp();

                if (lastSampleTimeUs == -1 || (currentTimestampUs - lastSampleTimeUs) >= sampleIntervalUs) {
                    lastSampleTimeUs = currentTimestampUs;
                    BufferedImage converted = converter.convert(frame);

                    if (converted != null) {
                        // Create deep copy in standard TYPE_INT_RGB format
                        BufferedImage copy = new BufferedImage(
                                converted.getWidth(), converted.getHeight(), BufferedImage.TYPE_INT_RGB);
                        Graphics2D g = copy.createGraphics();
                        try {
                            g.drawImage(converted, 0, 0, null);
                        } finally {
                            g.dispose();
                        }

                        double timestampSeconds = currentTimestampUs / 1_000_000.0;
                        sampledFrames.add(new SampledVideoFrame(frameIndex, timestampSeconds, copy));
                    }
                }
                frameIndex++;
            }

            grabber.stop();
            log.info("Video processing completed. Extracted {} sampled frames.", sampledFrames.size());
            return sampledFrames;

        } catch (Exception e) {
            log.error("Error occurred while grabbing video frames", e);
            throw new ModelInferenceException("Video frame extraction failed: " + e.getMessage(), e);
        }
    }
}
