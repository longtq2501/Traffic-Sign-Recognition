package com.trafficsign.common.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class FrameDetectionResult {
    private Integer frameNumber;
    private Double timestampSeconds;
    private SignPredictionResponse prediction;
}
