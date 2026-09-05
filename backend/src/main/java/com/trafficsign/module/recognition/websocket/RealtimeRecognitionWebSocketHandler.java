package com.trafficsign.module.recognition.websocket;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.trafficsign.common.dto.response.ApiResponse;
import com.trafficsign.common.dto.response.SignPredictionResponse;
import com.trafficsign.module.recognition.service.RecognitionOrchestratorService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;
import org.springframework.web.socket.CloseStatus;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import java.util.Base64;

@Slf4j
@Component
@RequiredArgsConstructor
public class RealtimeRecognitionWebSocketHandler extends TextWebSocketHandler {

    private final RecognitionOrchestratorService recognitionOrchestratorService;
    private final ObjectMapper objectMapper;

    @Override
    public void afterConnectionEstablished(WebSocketSession session) {
        log.info("Realtime WebSocket connection established: {}", session.getId());
    }

    @Override
    protected void handleTextMessage(WebSocketSession session, TextMessage message) throws Exception {
        try {
            String payload = message.getPayload();
            if (payload == null || payload.isBlank()) {
                return;
            }

            // Strip data URL scheme if present (e.g. data:image/jpeg;base64,...)
            String base64Data = payload;
            int commaIndex = payload.indexOf(',');
            if (commaIndex != -1 && payload.startsWith("data:")) {
                base64Data = payload.substring(commaIndex + 1);
            }

            byte[] imageBytes = Base64.getDecoder().decode(base64Data.trim());
            SignPredictionResponse prediction = recognitionOrchestratorService.recognizeImage(imageBytes);

            ApiResponse<SignPredictionResponse> response = ApiResponse.success(prediction);
            String jsonResponse = objectMapper.writeValueAsString(response);

            session.sendMessage(new TextMessage(jsonResponse));

        } catch (Exception e) {
            log.warn("Error processing realtime frame for session {}: {}", session.getId(), e.getMessage());
            ApiResponse<Void> errorResponse = ApiResponse.error("Failed to process frame: " + e.getMessage());
            session.sendMessage(new TextMessage(objectMapper.writeValueAsString(errorResponse)));
        }
    }

    @Override
    public void afterConnectionClosed(WebSocketSession session, CloseStatus status) {
        log.info("Realtime WebSocket connection closed: {} with status: {}", session.getId(), status);
    }
}
