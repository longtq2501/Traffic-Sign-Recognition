package com.trafficsign.common.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class VideoRecognitionResponse {
    private Double totalDurationSeconds;
    private Integer totalFramesProcessed;
    private List<FrameDetectionResult> detections;
    private Long processingTimeMs;
}
