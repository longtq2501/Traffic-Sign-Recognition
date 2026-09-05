package com.trafficsign.module.recognition.service;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.InputStream;
import java.util.List;

public interface VideoProcessingService {

    record SampledVideoFrame(int frameNumber, double timestampSeconds, BufferedImage image) {}

    /**
     * Extracts sampled frames from a video stream at specified time intervals.
     *
     * @param videoInputStream Stream containing video bytes
     * @param sampleIntervalSeconds Sampling interval in seconds (e.g., 0.5 for 2 frames per second)
     * @return List of sampled video frames
     */
    List<SampledVideoFrame> extractSampledFrames(InputStream videoInputStream, double sampleIntervalSeconds);

    /**
     * Extracts sampled frames from a local video file.
     *
     * @param videoFile Video file
     * @param sampleIntervalSeconds Sampling interval in seconds
     * @return List of sampled video frames
     */
    List<SampledVideoFrame> extractSampledFrames(File videoFile, double sampleIntervalSeconds);
}
