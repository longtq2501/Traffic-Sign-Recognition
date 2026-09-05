package com.trafficsign.module.persistence.repository;

import com.trafficsign.module.persistence.model.entity.InputType;
import com.trafficsign.module.persistence.model.entity.RecognitionLog;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface RecognitionLogRepository extends JpaRepository<RecognitionLog, Long> {

    Page<RecognitionLog> findAllByOrderByCreatedAtDesc(Pageable pageable);

    Page<RecognitionLog> findByUserIdOrderByCreatedAtDesc(Long userId, Pageable pageable);

    Page<RecognitionLog> findByInputTypeOrderByCreatedAtDesc(InputType inputType, Pageable pageable);

    List<RecognitionLog> findTop10ByOrderByCreatedAtDesc();

    @Query("SELECT COALESCE(AVG(r.confidenceScore), 0.0) FROM RecognitionLog r")
    Double getAverageConfidence();

    @Query("SELECT r.signClassId, r.detectedSignVi, COUNT(r) FROM RecognitionLog r GROUP BY r.signClassId, r.detectedSignVi ORDER BY COUNT(r) DESC")
    List<Object[]> getSignDetectionDistribution();
}
