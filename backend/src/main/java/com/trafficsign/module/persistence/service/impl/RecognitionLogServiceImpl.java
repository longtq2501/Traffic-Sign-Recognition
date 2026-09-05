package com.trafficsign.module.persistence.service.impl;

import com.trafficsign.common.dto.request.CreateRecognitionLogRequest;
import com.trafficsign.common.dto.response.RecognitionLogResponse;
import com.trafficsign.common.dto.response.RecognitionStatsResponse;
import com.trafficsign.module.persistence.model.entity.InputType;
import com.trafficsign.module.persistence.model.entity.RecognitionLog;
import com.trafficsign.module.persistence.model.entity.User;
import com.trafficsign.module.persistence.repository.RecognitionLogRepository;
import com.trafficsign.module.persistence.repository.UserRepository;
import com.trafficsign.module.persistence.service.RecognitionLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

@Slf4j
@Service
@RequiredArgsConstructor
@Transactional(readOnly = true)
public class RecognitionLogServiceImpl implements RecognitionLogService {

    private final RecognitionLogRepository recognitionLogRepository;
    private final UserRepository userRepository;

    @Override
    @Transactional
    public RecognitionLogResponse saveLog(CreateRecognitionLogRequest request) {
        User user = null;
        if (request.getUserId() != null) {
            user = userRepository.findById(request.getUserId()).orElse(null);
        }

        RecognitionLog logEntity = RecognitionLog.builder()
                .user(user)
                .inputType(request.getInputType())
                .signClassId(request.getSignClassId())
                .detectedSignEn(request.getDetectedSignEn())
                .detectedSignVi(request.getDetectedSignVi())
                .category(request.getCategory())
                .confidenceScore(request.getConfidenceScore())
                .inferenceTimeMs(request.getInferenceTimeMs())
                .fileReference(request.getFileReference())
                .build();

        RecognitionLog savedLog = recognitionLogRepository.save(logEntity);
        log.debug("Saved recognition log id: {} for sign: {}", savedLog.getId(), savedLog.getDetectedSignVi());

        return mapToResponse(savedLog);
    }

    @Override
    public Page<RecognitionLogResponse> getAllLogs(Pageable pageable) {
        return recognitionLogRepository.findAllByOrderByCreatedAtDesc(pageable)
                .map(this::mapToResponse);
    }

    @Override
    public Page<RecognitionLogResponse> getLogsByUserId(Long userId, Pageable pageable) {
        return recognitionLogRepository.findByUserIdOrderByCreatedAtDesc(userId, pageable)
                .map(this::mapToResponse);
    }

    @Override
    public Page<RecognitionLogResponse> getLogsByInputType(InputType inputType, Pageable pageable) {
        return recognitionLogRepository.findByInputTypeOrderByCreatedAtDesc(inputType, pageable)
                .map(this::mapToResponse);
    }

    @Override
    public List<RecognitionLogResponse> getRecentLogs() {
        return recognitionLogRepository.findTop10ByOrderByCreatedAtDesc()
                .stream()
                .map(this::mapToResponse)
                .collect(Collectors.toList());
    }

    @Override
    public RecognitionStatsResponse getStats() {
        long totalCount = recognitionLogRepository.count();
        Double avgConfidence = recognitionLogRepository.getAverageConfidence();
        List<Object[]> distribution = recognitionLogRepository.getSignDetectionDistribution();

        Map<String, Long> topSigns = new LinkedHashMap<>();
        for (Object[] row : distribution) {
            String signName = (String) row[1];
            if (signName == null || signName.isBlank()) {
                signName = "Class " + row[0];
            }
            Long count = ((Number) row[2]).longValue();
            topSigns.put(signName, count);
        }

        return RecognitionStatsResponse.builder()
                .totalRecognitions(totalCount)
                .averageConfidence(avgConfidence != null ? Math.round(avgConfidence * 10000.0) / 10000.0 : 0.0)
                .topDetectedSigns(topSigns)
                .build();
    }

    private RecognitionLogResponse mapToResponse(RecognitionLog logEntity) {
        return RecognitionLogResponse.builder()
                .id(logEntity.getId())
                .userId(logEntity.getUser() != null ? logEntity.getUser().getId() : null)
                .inputType(logEntity.getInputType())
                .signClassId(logEntity.getSignClassId())
                .detectedSignEn(logEntity.getDetectedSignEn())
                .detectedSignVi(logEntity.getDetectedSignVi())
                .category(logEntity.getCategory())
                .confidenceScore(logEntity.getConfidenceScore())
                .inferenceTimeMs(logEntity.getInferenceTimeMs())
                .fileReference(logEntity.getFileReference())
                .createdAt(logEntity.getCreatedAt())
                .build();
    }
}
