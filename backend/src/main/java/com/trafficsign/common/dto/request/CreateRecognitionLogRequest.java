package com.trafficsign.common.dto.request;

import com.trafficsign.module.persistence.model.entity.InputType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class CreateRecognitionLogRequest {
    private Long userId;
    private InputType inputType;
    private Integer signClassId;
    private String detectedSignEn;
    private String detectedSignVi;
    private String category;
    private Double confidenceScore;
    private Long inferenceTimeMs;
    private String fileReference;
}
