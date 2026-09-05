package com.trafficsign.persistence;

import com.trafficsign.common.dto.request.CreateRecognitionLogRequest;
import com.trafficsign.common.dto.response.RecognitionLogResponse;
import com.trafficsign.common.dto.response.RecognitionStatsResponse;
import com.trafficsign.module.persistence.model.entity.InputType;
import com.trafficsign.module.persistence.model.entity.User;
import com.trafficsign.module.persistence.repository.RecognitionLogRepository;
import com.trafficsign.module.persistence.repository.UserRepository;
import com.trafficsign.module.persistence.service.RecognitionLogService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest
class RecognitionLogServiceTest {

    @Autowired
    private RecognitionLogService recognitionLogService;

    @Autowired
    private RecognitionLogRepository recognitionLogRepository;

    @Autowired
    private UserRepository userRepository;

    @BeforeEach
    void setUp() {
        recognitionLogRepository.deleteAll();
        userRepository.deleteAll();
    }

    @Test
    @DisplayName("Should save recognition log without user successfully")
    void testSaveLogWithoutUser() {
        CreateRecognitionLogRequest request = CreateRecognitionLogRequest.builder()
                .inputType(InputType.IMAGE)
                .signClassId(14)
                .detectedSignEn("Stop")
                .detectedSignVi("Dung lai")
                .category("Danger")
                .confidenceScore(0.985)
                .inferenceTimeMs(12L)
                .fileReference("uploads/images/stop_sign.png")
                .build();

        RecognitionLogResponse response = recognitionLogService.saveLog(request);

        assertNotNull(response.getId());
        assertNull(response.getUserId());
        assertEquals(InputType.IMAGE, response.getInputType());
        assertEquals(14, response.getSignClassId());
        assertEquals("Stop", response.getDetectedSignEn());
        assertEquals("Dung lai", response.getDetectedSignVi());
        assertEquals(0.985, response.getConfidenceScore(), 0.0001);
        assertEquals(12L, response.getInferenceTimeMs());
        assertNotNull(response.getCreatedAt());
    }

    @Test
    @DisplayName("Should save recognition log with existing user")
    void testSaveLogWithUser() {
        User user = userRepository.save(User.builder()
                .username("testdriver")
                .email("driver@example.com")
                .fullName("Test Driver")
                .role("USER")
                .build());

        CreateRecognitionLogRequest request = CreateRecognitionLogRequest.builder()
                .userId(user.getId())
                .inputType(InputType.VIDEO)
                .signClassId(1)
                .detectedSignEn("Speed limit (30km/h)")
                .detectedSignVi("Gioi han toc do 30km/h")
                .category("Speed Limit")
                .confidenceScore(0.952)
                .inferenceTimeMs(18L)
                .fileReference("uploads/videos/dashcam_trip1.mp4")
                .build();

        RecognitionLogResponse response = recognitionLogService.saveLog(request);

        assertNotNull(response.getId());
        assertEquals(user.getId(), response.getUserId());
        assertEquals(InputType.VIDEO, response.getInputType());
        assertEquals(1, response.getSignClassId());
    }

    @Test
    @DisplayName("Should retrieve paginated logs and filter by input type")
    void testPaginationAndFilter() {
        for (int i = 0; i < 5; i++) {
            recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                    .inputType(InputType.IMAGE)
                    .signClassId(i)
                    .detectedSignEn("Sign " + i)
                    .detectedSignVi("Bien bao " + i)
                    .confidenceScore(0.90 + (i * 0.01))
                    .inferenceTimeMs(10L + i)
                    .build());
        }
        for (int i = 5; i < 8; i++) {
            recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                    .inputType(InputType.VIDEO)
                    .signClassId(i)
                    .detectedSignEn("Sign " + i)
                    .detectedSignVi("Bien bao " + i)
                    .confidenceScore(0.85 + (i * 0.01))
                    .inferenceTimeMs(15L + i)
                    .build());
        }

        Page<RecognitionLogResponse> allLogs = recognitionLogService.getAllLogs(PageRequest.of(0, 5));
        assertEquals(8, allLogs.getTotalElements());
        assertEquals(5, allLogs.getContent().size());

        Page<RecognitionLogResponse> videoLogs = recognitionLogService.getLogsByInputType(InputType.VIDEO, PageRequest.of(0, 10));
        assertEquals(3, videoLogs.getTotalElements());

        List<RecognitionLogResponse> recentLogs = recognitionLogService.getRecentLogs();
        assertEquals(8, recentLogs.size());
    }

    @Test
    @DisplayName("Should aggregate recognition statistics accurately")
    void testGetStats() {
        recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                .inputType(InputType.IMAGE)
                .signClassId(14)
                .detectedSignEn("Stop")
                .detectedSignVi("Dung lai")
                .confidenceScore(0.90)
                .build());

        recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                .inputType(InputType.IMAGE)
                .signClassId(14)
                .detectedSignEn("Stop")
                .detectedSignVi("Dung lai")
                .confidenceScore(1.00)
                .build());

        recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                .inputType(InputType.VIDEO)
                .signClassId(1)
                .detectedSignEn("Speed limit (30km/h)")
                .detectedSignVi("Gioi han toc do 30km/h")
                .confidenceScore(0.80)
                .build());

        RecognitionStatsResponse stats = recognitionLogService.getStats();

        assertEquals(3L, stats.getTotalRecognitions());
        assertEquals(0.90, stats.getAverageConfidence(), 0.001);
        assertNotNull(stats.getTopDetectedSigns());
        assertEquals(2L, stats.getTopDetectedSigns().get("Dung lai"));
        assertEquals(1L, stats.getTopDetectedSigns().get("Gioi han toc do 30km/h"));
    }
}
