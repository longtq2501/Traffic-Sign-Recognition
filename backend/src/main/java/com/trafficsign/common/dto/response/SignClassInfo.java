package com.trafficsign.common.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class SignClassInfo {
    private Integer id;
    private String nameEn;
    private String nameVi;
    private String category;
}
