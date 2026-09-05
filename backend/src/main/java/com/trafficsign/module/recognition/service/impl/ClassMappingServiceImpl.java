package com.trafficsign.module.recognition.service.impl;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.trafficsign.common.dto.response.SignClassInfo;
import com.trafficsign.common.exception.ModelInferenceException;
import com.trafficsign.module.recognition.service.ClassMappingService;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Slf4j
@Service
@RequiredArgsConstructor
public class ClassMappingServiceImpl implements ClassMappingService {

    private final ResourceLoader resourceLoader;
    private final ObjectMapper objectMapper;

    @Value("${app.model.mapping-path:classpath:model/class_mapping.json}")
    private String mappingPath;

    private final Map<Integer, SignClassInfo> classMap = new ConcurrentHashMap<>();

    @PostConstruct
    public void init() {
        log.info("Loading class mapping configuration from: {}", mappingPath);
        try {
            Resource resource = resourceLoader.getResource(mappingPath);
            if (!resource.exists()) {
                throw new ModelInferenceException("Class mapping resource not found at: " + mappingPath);
            }

            try (InputStream is = resource.getInputStream()) {
                JsonNode root = objectMapper.readTree(is);
                JsonNode classesNode = root.get("classes");
                if (classesNode == null || !classesNode.isArray()) {
                    throw new ModelInferenceException("Invalid class mapping format: missing 'classes' array");
                }

                classMap.clear();
                for (JsonNode item : classesNode) {
                    int id = item.get("id").asInt();
                    String nameEn = item.has("name_en") ? item.get("name_en").asText() : "Unknown";
                    String nameVi = item.has("name_vi") ? item.get("name_vi").asText() : "Chưa rõ";
                    String category = item.has("category") ? item.get("category").asText() : "Other";

                    SignClassInfo info = SignClassInfo.builder()
                            .id(id)
                            .nameEn(nameEn)
                            .nameVi(nameVi)
                            .category(category)
                            .build();
                    classMap.put(id, info);
                }
                log.info("Successfully loaded {} traffic sign classes into memory.", classMap.size());
            }
        } catch (Exception e) {
            log.error("Failed to load traffic sign class mapping", e);
            throw new ModelInferenceException("Failed to initialize ClassMappingService: " + e.getMessage(), e);
        }
    }

    @Override
    public SignClassInfo getClassInfo(int classId) {
        SignClassInfo info = classMap.get(classId);
        if (info == null) {
            return SignClassInfo.builder()
                    .id(classId)
                    .nameEn("Unknown Class (" + classId + ")")
                    .nameVi("Biển báo chưa xác định (" + classId + ")")
                    .category("Unknown")
                    .build();
        }
        return info;
    }

    @Override
    public Map<Integer, SignClassInfo> getAllClasses() {
        return Collections.unmodifiableMap(classMap);
    }

    @Override
    public int getTotalClasses() {
        return classMap.size();
    }
}
