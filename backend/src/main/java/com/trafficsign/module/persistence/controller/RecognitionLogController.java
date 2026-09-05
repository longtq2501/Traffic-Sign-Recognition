package com.trafficsign.module.persistence.controller;

import com.trafficsign.common.dto.response.ApiResponse;
import com.trafficsign.common.dto.response.RecognitionLogResponse;
import com.trafficsign.common.dto.response.RecognitionStatsResponse;
import com.trafficsign.module.persistence.model.entity.InputType;
import com.trafficsign.module.persistence.service.RecognitionLogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Pageable;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@Slf4j
@RestController
@RequestMapping("/api/logs")
@RequiredArgsConstructor
public class RecognitionLogController {

    private final RecognitionLogService recognitionLogService;

    @GetMapping
    public ResponseEntity<ApiResponse<Page<RecognitionLogResponse>>> getLogs(
            @RequestParam(value = "page", defaultValue = "0") int page,
            @RequestParam(value = "size", defaultValue = "10") int size,
            @RequestParam(value = "inputType", required = false) InputType inputType,
            @RequestParam(value = "userId", required = false) Long userId) {

        Pageable pageable = PageRequest.of(Math.max(0, page), Math.max(1, Math.min(size, 100)));
        Page<RecognitionLogResponse> logs;

        if (userId != null) {
            logs = recognitionLogService.getLogsByUserId(userId, pageable);
        } else if (inputType != null) {
            logs = recognitionLogService.getLogsByInputType(inputType, pageable);
        } else {
            logs = recognitionLogService.getAllLogs(pageable);
        }

        return ResponseEntity.ok(ApiResponse.success("Recognition logs retrieved successfully", logs));
    }

    @GetMapping("/recent")
    public ResponseEntity<ApiResponse<List<RecognitionLogResponse>>> getRecentLogs() {
        List<RecognitionLogResponse> recentLogs = recognitionLogService.getRecentLogs();
        return ResponseEntity.ok(ApiResponse.success("Recent recognition logs retrieved successfully", recentLogs));
    }

    @GetMapping("/stats")
    public ResponseEntity<ApiResponse<RecognitionStatsResponse>> getStats() {
        RecognitionStatsResponse stats = recognitionLogService.getStats();
        return ResponseEntity.ok(ApiResponse.success("Recognition statistics retrieved successfully", stats));
    }
}
