package com.trafficsign.module.recognition.service.impl;

import com.trafficsign.common.exception.ModelInferenceException;
import com.trafficsign.module.recognition.service.ImageProcessingService;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;

@Slf4j
@Service
public class ImageProcessingServiceImpl implements ImageProcessingService {

    public static final int TARGET_WIDTH = 32;
    public static final int TARGET_HEIGHT = 32;

    // ImageNet normalization statistics matching PyTorch training pipeline
    public static final float[] NORM_MEAN = {0.485f, 0.456f, 0.406f};
    public static final float[] NORM_STD = {0.229f, 0.224f, 0.225f};

    @Override
    public float[][][][] preprocessImage(InputStream inputStream) {
        if (inputStream == null) {
            throw new ModelInferenceException("Image input stream must not be null");
        }
        try {
            byte[] imageBytes = inputStream.readAllBytes();
            return preprocessImage(imageBytes);
        } catch (IOException e) {
            log.error("Error reading image input stream", e);
            throw new ModelInferenceException("Failed to read image stream: " + e.getMessage(), e);
        }
    }

    @Override
    public float[][][][] preprocessImage(byte[] imageBytes) {
        if (imageBytes == null || imageBytes.length == 0) {
            throw new ModelInferenceException("Image bytes must not be empty");
        }
        try (ByteArrayInputStream bais = new ByteArrayInputStream(imageBytes)) {
            BufferedImage image = ImageIO.read(bais);
            if (image == null) {
                throw new ModelInferenceException("Unable to decode image: unsupported or corrupted image format. Supported: JPEG, PNG, GIF, BMP");
            }
            return preprocessImage(image);
        } catch (IOException e) {
            throw new ModelInferenceException("Failed to read image byte array: " + e.getMessage(), e);
        }
    }

    @Override
    public float[][][][] preprocessImage(BufferedImage image) {
        if (image == null) {
            throw new ModelInferenceException("BufferedImage cannot be null");
        }

        // Step 1: Composite onto white RGB background to handle alpha/transparency (PNG, WebP)
        // Without this, transparent pixels become black (0,0,0) which skews normalization
        BufferedImage rgbImage = new BufferedImage(image.getWidth(), image.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = rgbImage.createGraphics();
        try {
            g2d.setColor(java.awt.Color.WHITE);
            g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
            g2d.drawImage(image, 0, 0, null);
        } finally {
            g2d.dispose();
        }

        // Step 2: Square-crop from center before resizing.
        // torchvision.transforms.Resize((32,32)) on non-square images does NOT stretch —
        // it resizes the shorter side to 32 then center-crops to 32x32.
        // We replicate this: crop a center square first, then resize.
        BufferedImage squared = centerCropToSquare(rgbImage);

        // Step 3: Resize to 32x32 using high-quality Bilinear interpolation
        BufferedImage resized = resizeImage(squared, TARGET_WIDTH, TARGET_HEIGHT);

        // Step 4: Build NCHW tensor [1, 3, 32, 32]
        float[][][][] tensor = new float[1][3][TARGET_HEIGHT][TARGET_WIDTH];

        for (int y = 0; y < TARGET_HEIGHT; y++) {
            for (int x = 0; x < TARGET_WIDTH; x++) {
                int rgb = resized.getRGB(x, y);

                // Extract RGB channels [0..255]
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Scale to [0.0, 1.0] and apply ImageNet normalization
                tensor[0][0][y][x] = ((r / 255.0f) - NORM_MEAN[0]) / NORM_STD[0];
                tensor[0][1][y][x] = ((g / 255.0f) - NORM_MEAN[1]) / NORM_STD[1];
                tensor[0][2][y][x] = ((b / 255.0f) - NORM_MEAN[2]) / NORM_STD[2];
            }
        }

        return tensor;
    }

    /**
     * Crops the largest center square from the image, preserving aspect ratio.
     * Equivalent to torchvision CenterCrop behavior when combined with Resize.
     */
    private BufferedImage centerCropToSquare(BufferedImage image) {
        int w = image.getWidth();
        int h = image.getHeight();
        if (w == h) {
            return image;
        }
        int side = Math.min(w, h);
        int x = (w - side) / 2;
        int y = (h - side) / 2;
        return image.getSubimage(x, y, side, side);
    }

    @Override
    public BufferedImage resizeImage(BufferedImage originalImage, int targetWidth, int targetHeight) {
        BufferedImage resizedImage = new BufferedImage(targetWidth, targetHeight, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = resizedImage.createGraphics();

        try {
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
            g2d.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.drawImage(originalImage, 0, 0, targetWidth, targetHeight, null);
        } finally {
            g2d.dispose();
        }

        return resizedImage;
    }
}
