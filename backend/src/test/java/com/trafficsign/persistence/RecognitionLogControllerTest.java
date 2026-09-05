package com.trafficsign.persistence;

import com.trafficsign.common.dto.request.CreateRecognitionLogRequest;
import com.trafficsign.module.persistence.model.entity.InputType;
import com.trafficsign.module.persistence.repository.RecognitionLogRepository;
import com.trafficsign.module.persistence.service.RecognitionLogService;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.web.servlet.MockMvc;

import static org.hamcrest.Matchers.*;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@SpringBootTest
@AutoConfigureMockMvc
class RecognitionLogControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @Autowired
    private RecognitionLogService recognitionLogService;

    @Autowired
    private RecognitionLogRepository recognitionLogRepository;

    @BeforeEach
    void setUp() {
        recognitionLogRepository.deleteAll();

        recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                .inputType(InputType.IMAGE)
                .signClassId(14)
                .detectedSignEn("Stop")
                .detectedSignVi("Dung lai")
                .confidenceScore(0.95)
                .inferenceTimeMs(15L)
                .build());

        recognitionLogService.saveLog(CreateRecognitionLogRequest.builder()
                .inputType(InputType.VIDEO)
                .signClassId(1)
                .detectedSignEn("Speed limit (30km/h)")
                .detectedSignVi("Gioi han toc do 30km/h")
                .confidenceScore(0.88)
                .inferenceTimeMs(20L)
                .build());
    }

    @Test
    @DisplayName("GET /api/logs should return paginated list of recognition logs")
    void testGetLogs() throws Exception {
        mockMvc.perform(get("/api/logs?page=0&size=10"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success", is(true)))
                .andExpect(jsonPath("$.data.page.totalElements", is(2)))
                .andExpect(jsonPath("$.data.content", hasSize(2)));
    }

    @Test
    @DisplayName("GET /api/logs?inputType=IMAGE should filter logs by input type")
    void testGetLogsByInputType() throws Exception {
        mockMvc.perform(get("/api/logs?inputType=IMAGE"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success", is(true)))
                .andExpect(jsonPath("$.data.page.totalElements", is(1)))
                .andExpect(jsonPath("$.data.content[0].inputType", is("IMAGE")))
                .andExpect(jsonPath("$.data.content[0].detectedSignEn", is("Stop")));
    }

    @Test
    @DisplayName("GET /api/logs/recent should return top recent logs")
    void testGetRecentLogs() throws Exception {
        mockMvc.perform(get("/api/logs/recent"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success", is(true)))
                .andExpect(jsonPath("$.data", hasSize(2)));
    }

    @Test
    @DisplayName("GET /api/logs/stats should return recognition aggregated metrics")
    void testGetStats() throws Exception {
        mockMvc.perform(get("/api/logs/stats"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.success", is(true)))
                .andExpect(jsonPath("$.data.totalRecognitions", is(2)))
                .andExpect(jsonPath("$.data.averageConfidence", notNullValue()))
                .andExpect(jsonPath("$.data.topDetectedSigns.['Dung lai']", is(1)));
    }
}
