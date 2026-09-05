package com.trafficsign.recognition;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
class RecognitionControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Test
    @DisplayName("GET /api/recognize/classes should return all 43 traffic sign classes")
    void testGetAllClasses() throws Exception {
        mockMvc.perform(get("/api/recognize/classes"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success", is(true)))
                .andExpect(jsonPath("$.data", aMapWithSize(43)))
                .andExpect(jsonPath("$.data['14'].nameEn", is("Stop")));
    }

    @Test
    @DisplayName("POST /api/recognize/image should fail with 400 when file is empty")
    void testRecognizeImageEmptyFile() throws Exception {
        MockMultipartFile emptyFile = new MockMultipartFile(
                "file", "empty.png", "image/png", new byte[0]);

        mockMvc.perform(multipart("/api/recognize/image").file(emptyFile))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.success", is(false)))
                .andExpect(jsonPath("$.message", containsString("cannot be empty")));
    }

    @Test
    @DisplayName("POST /api/recognize/image should fail with 400 when file is not an image")
    void testRecognizeImageInvalidFormat() throws Exception {
        MockMultipartFile textFile = new MockMultipartFile(
                "file", "document.txt", "text/plain", "Hello world".getBytes());

        mockMvc.perform(multipart("/api/recognize/image").file(textFile))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.success", is(false)))
                .andExpect(jsonPath("$.message", containsString("Invalid file format")));
    }

    @Test
    @DisplayName("POST /api/recognize/image should succeed with valid image and record log")
    void testRecognizeImageSuccess() throws Exception {
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setColor(Color.RED);
        g2d.fillRect(0, 0, 32, 32);
        g2d.dispose();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "png", baos);
        byte[] imageBytes = baos.toByteArray();

        MockMultipartFile validImage = new MockMultipartFile(
                "file", "test_red_sign.png", "image/png", imageBytes);

        mockMvc.perform(multipart("/api/recognize/image").file(validImage))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success", is(true)))
                .andExpect(jsonPath("$.data.logId", notNullValue()))
                .andExpect(jsonPath("$.data.classId", notNullValue()))
                .andExpect(jsonPath("$.data.signNameVi", notNullValue()))
                .andExpect(jsonPath("$.data.confidence", greaterThan(0.0)))
                .andExpect(jsonPath("$.data.fileName", is("test_red_sign.png")));
    }
}
