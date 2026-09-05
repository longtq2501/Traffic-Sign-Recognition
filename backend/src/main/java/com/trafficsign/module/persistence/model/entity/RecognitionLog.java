package com.trafficsign.module.persistence.model.entity;

import jakarta.persistence.*;
import lombok.*;

import java.time.LocalDateTime;

@Entity
@Table(name = "recognition_logs", indexes = {
        @Index(name = "idx_log_user_id", columnList = "user_id"),
        @Index(name = "idx_log_input_type", columnList = "input_type"),
        @Index(name = "idx_log_created_at", columnList = "created_at"),
        @Index(name = "idx_log_sign_class_id", columnList = "sign_class_id")
})
@Getter
@Setter
@NoArgsConstructor
@AllArgsConstructor
@Builder
@ToString(exclude = {"user"})
public class RecognitionLog {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = true)
    private User user;

    @Enumerated(EnumType.STRING)
    @Column(name = "input_type", nullable = false, length = 20)
    private InputType inputType;

    @Column(name = "sign_class_id", nullable = false)
    private Integer signClassId;

    @Column(name = "detected_sign_en", length = 150)
    private String detectedSignEn;

    @Column(name = "detected_sign_vi", length = 150)
    private String detectedSignVi;

    @Column(name = "category", length = 50)
    private String category;

    @Column(name = "confidence_score", nullable = false)
    private Double confidenceScore;

    @Column(name = "inference_time_ms")
    private Long inferenceTimeMs;

    @Column(name = "file_reference", length = 255)
    private String fileReference;

    @Column(name = "created_at", updatable = false)
    private LocalDateTime createdAt;

    @PrePersist
    protected void onCreate() {
        if (this.createdAt == null) {
            this.createdAt = LocalDateTime.now();
        }
    }
}
