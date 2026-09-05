package com.trafficsign.common.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.Map;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class RecognitionStatsResponse {
    private Long totalRecognitions;
    private Double averageConfidence;
    private Map<String, Long> topDetectedSigns;
}
