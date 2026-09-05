package com.trafficsign.recognition;

import com.trafficsign.common.dto.response.SignClassInfo;
import com.trafficsign.common.dto.response.SignPredictionResponse;
import com.trafficsign.module.recognition.service.ClassMappingService;
import com.trafficsign.module.recognition.service.ModelInferenceService;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class ModelInferenceServiceTest {

    @Autowired
    private ClassMappingService classMappingService;

    @Autowired
    private ModelInferenceService modelInferenceService;

    @Test
    @DisplayName("Should load 43 traffic sign classes from class_mapping.json")
    void testClassMappingLoaded() {
        assertEquals(43, classMappingService.getTotalClasses());

        SignClassInfo speed20 = classMappingService.getClassInfo(0);
        assertNotNull(speed20);
        assertEquals("Speed limit (20km/h)", speed20.getNameEn());
        assertEquals("Giới hạn tốc độ 20km/h", speed20.getNameVi());
        assertEquals("Prohibitory", speed20.getCategory());

        SignClassInfo stopSign = classMappingService.getClassInfo(14);
        assertNotNull(stopSign);
        assertEquals("Stop", stopSign.getNameEn());
        assertEquals("Dừng lại (Stop)", stopSign.getNameVi());
    }

    @Test
    @DisplayName("Should initialize ONNX session with matching input/output tensor names")
    void testOnnxSessionInitialized() {
        assertEquals("input", modelInferenceService.getInputName());
        assertEquals("output", modelInferenceService.getOutputName());
    }

    @Test
    @DisplayName("Should run inference on 4D NCHW tensor and return valid prediction response")
    void testInferenceOnDummyTensor() {
        // Prepare dummy NCHW tensor [1, 3, 32, 32]
        float[][][][] dummyTensor = new float[1][3][32][32];
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 32; h++) {
                for (int w = 0; w < 32; w++) {
                    dummyTensor[0][c][h][w] = 0.5f;
                }
            }
        }

        SignPredictionResponse response = modelInferenceService.predict(dummyTensor);

        assertNotNull(response);
        assertTrue(response.getClassId() >= 0 && response.getClassId() < 43, "ClassId must be in [0, 42]");
        assertNotNull(response.getSignNameEn());
        assertNotNull(response.getSignNameVi());
        assertNotNull(response.getCategory());
        assertTrue(response.getConfidence() > 0.0 && response.getConfidence() <= 100.0, "Confidence must be between 0 and 100%");
        assertTrue(response.getInferenceTimeMs() >= 0, "Inference time should be recorded");

        System.out.println("Java ONNX Inference Test Result:");
        System.out.println("  Class ID    : " + response.getClassId());
        System.out.println("  Sign (EN)   : " + response.getSignNameEn());
        System.out.println("  Sign (VI)   : " + response.getSignNameVi());
        System.out.println("  Category    : " + response.getCategory());
        System.out.println("  Confidence  : " + response.getConfidence() + "%");
        System.out.println("  Inference Time: " + response.getInferenceTimeMs() + " ms");
    }
}
