package com.trafficsign.common.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class ImageRecognitionResponse {
    private Long logId;
    private Integer classId;
    private String signNameEn;
    private String signNameVi;
    private String category;
    private Double confidence;
    private Long inferenceTimeMs;
    private String fileName;
}
