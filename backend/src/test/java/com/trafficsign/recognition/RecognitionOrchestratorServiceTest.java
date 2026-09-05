package com.trafficsign.recognition;

import com.trafficsign.common.dto.response.SignPredictionResponse;
import com.trafficsign.module.recognition.service.RecognitionOrchestratorService;
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
class RecognitionOrchestratorServiceTest {

    @Autowired
    private RecognitionOrchestratorService recognitionOrchestratorService;

    @Test
    @DisplayName("Should orchestrate recognition pipeline from BufferedImage end-to-end")
    void testRecognizeBufferedImage() {
        BufferedImage image = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setColor(Color.BLUE);
        g.fillOval(5, 5, 40, 40);
        g.dispose();

        SignPredictionResponse response = recognitionOrchestratorService.recognizeImage(image);

        assertNotNull(response);
        assertTrue(response.getClassId() >= 0 && response.getClassId() < 43);
        assertNotNull(response.getSignNameEn());
        assertNotNull(response.getSignNameVi());
        assertNotNull(response.getCategory());
        assertTrue(response.getConfidence() >= 0.0 && response.getConfidence() <= 100.0);

        System.out.println("End-to-End Image Recognition Result:");
        System.out.println("  Class: " + response.getClassId() + " -> " + response.getSignNameVi() + " (" + response.getSignNameEn() + ")");
        System.out.println("  Confidence: " + response.getConfidence() + "%");
        System.out.println("  Inference Time: " + response.getInferenceTimeMs() + " ms");
    }

    @Test
    @DisplayName("Should orchestrate recognition pipeline from raw image bytes")
    void testRecognizeImageBytes() throws IOException {
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "jpg", baos);
        byte[] bytes = baos.toByteArray();

        SignPredictionResponse response = recognitionOrchestratorService.recognizeImage(bytes);

        assertNotNull(response);
        assertNotNull(response.getSignNameVi());
        assertTrue(response.getConfidence() >= 0.0);
    }
}
