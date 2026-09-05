package com.trafficsign.module.recognition.service;

import java.awt.image.BufferedImage;
import java.io.InputStream;

public interface ImageProcessingService {

    /**
     * Converts an image input stream into normalized 4D NCHW tensor [1, 3, 32, 32].
     */
    float[][][][] preprocessImage(InputStream inputStream);

    /**
     * Converts image bytes into normalized 4D NCHW tensor [1, 3, 32, 32].
     */
    float[][][][] preprocessImage(byte[] imageBytes);

    /**
     * Converts a BufferedImage into normalized 4D NCHW tensor [1, 3, 32, 32].
     */
    float[][][][] preprocessImage(BufferedImage image);

    /**
     * Resizes a BufferedImage to target dimensions using high-quality bilinear interpolation.
     */
    BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight);
}
