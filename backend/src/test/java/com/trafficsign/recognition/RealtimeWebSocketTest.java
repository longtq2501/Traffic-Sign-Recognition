package com.trafficsign.recognition;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.web.socket.TextMessage;
import org.springframework.web.socket.WebSocketSession;
import org.springframework.web.socket.client.standard.StandardWebSocketClient;
import org.springframework.web.socket.handler.TextWebSocketHandler;

import javax.imageio.ImageIO;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.util.Base64;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
class RealtimeWebSocketTest {

    @LocalServerPort
    private int port;

    @Autowired
    private ObjectMapper objectMapper;

    @Test
    @DisplayName("WebSocket /ws/realtime should receive image base64 frame and return prediction JSON")
    void testRealtimeWebSocketFrameInference() throws Exception {
        BufferedImage image = new BufferedImage(32, 32, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2d = image.createGraphics();
        g2d.setColor(Color.BLUE);
        g2d.fillRect(0, 0, 32, 32);
        g2d.dispose();

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        ImageIO.write(image, "jpg", baos);
        String base64Data = "data:image/jpeg;base64," + Base64.getEncoder().encodeToString(baos.toByteArray());

        CompletableFuture<String> responseFuture = new CompletableFuture<>();

        StandardWebSocketClient client = new StandardWebSocketClient();
        String wsUri = "ws://localhost:" + port + "/ws/realtime";

        WebSocketSession session = client.execute(new TextWebSocketHandler() {
            @Override
            protected void handleTextMessage(WebSocketSession s, TextMessage message) {
                responseFuture.complete(message.getPayload());
            }
        }, wsUri).get(5, TimeUnit.SECONDS);

        session.sendMessage(new TextMessage(base64Data));

        String responsePayload = responseFuture.get(5, TimeUnit.SECONDS);
        session.close();

        assertNotNull(responsePayload);
        JsonNode root = objectMapper.readTree(responsePayload);
        assertTrue(root.get("success").asBoolean());
        assertNotNull(root.get("data").get("classId"));
        assertNotNull(root.get("data").get("signNameVi"));
        assertTrue(root.get("data").get("confidence").asDouble() > 0.0);
    }
}
