package com.trafficsign.module.persistence.service;

import com.trafficsign.common.dto.request.CreateRecognitionLogRequest;
import com.trafficsign.common.dto.response.RecognitionLogResponse;
import com.trafficsign.common.dto.response.RecognitionStatsResponse;
import com.trafficsign.module.persistence.model.entity.InputType;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;

import java.util.List;

public interface RecognitionLogService {

    RecognitionLogResponse saveLog(CreateRecognitionLogRequest request);

    Page<RecognitionLogResponse> getAllLogs(Pageable pageable);

    Page<RecognitionLogResponse> getLogsByUserId(Long userId, Pageable pageable);

    Page<RecognitionLogResponse> getLogsByInputType(InputType inputType, Pageable pageable);

    List<RecognitionLogResponse> getRecentLogs();

    RecognitionStatsResponse getStats();
}
