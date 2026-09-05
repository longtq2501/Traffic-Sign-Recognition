package com.trafficsign.recognition;

import com.trafficsign.module.recognition.service.ImageProcessingService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class ImageProcessingServiceTest {

    @Autowired
    private ImageProcessingService imageProcessingService;

    @Test
    @DisplayName("Should resize image to 32x32 correctly")
    void testResizeImage() {
        BufferedImage original = new BufferedImage(100, 100, BufferedImage.TYPE_INT_RGB);
        BufferedImage resized = imageProcessingService.resizeImage(original, 32, 32);

        assertNotNull(resized);
        assertEquals(32, resized.getWidth());
        assertEquals(32, resized.getHeight());
    }

    @Test
    @DisplayName("Should preprocess BufferedImage to normalized NCHW float tensor [1, 3, 32, 32]")
    void testPreprocessBufferedImage() {
        BufferedImage testImage = new BufferedImage(64, 64, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = testImage.createGraphics();
        g.setColor(Color.RED);
        g.fillRect(0, 0, 64, 64);
        g.dispose();

        float[][][][] tensor = imageProcessingService.preprocessImage(testImage);

        assertNotNull(tensor);
        assertEquals(1, tensor.length, "Batch size must be 1");
        assertEquals(3, tensor[0].length, "Channels must be 3 (RGB)");
        assertEquals(32, tensor[0][0].length, "Height must be 32");
        assertEquals(32, tensor[0][0][0].length, "Width must be 32");

        // Verify normalized red channel for pure red (255, 0, 0)
        // Red: (1.0 - 0.485) / 0.229 ≈ 2.248
        float redVal = tensor[0][0][0][0];
        assertTrue(redVal > 2.0f && redVal < 2.5f, "Normalized red value should match ImageNet formula");
    }

    @Test
    @DisplayName("Should preprocess image from byte array")
    void testPreprocessByteArray() throws IOException {
        BufferedImage testImage = new BufferedImage(48, 48, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(testImage, "png", baos);
        byte[] imageBytes = baos.toByteArray();

        float[][][][] tensor = imageProcessingService.preprocessImage(imageBytes);

        assertNotNull(tensor);
        assertEquals(1, tensor.length);
        assertEquals(3, tensor[0].length);
        assertEquals(32, tensor[0][0].length);
        assertEquals(32, tensor[0][0][0].length);
    }
}
