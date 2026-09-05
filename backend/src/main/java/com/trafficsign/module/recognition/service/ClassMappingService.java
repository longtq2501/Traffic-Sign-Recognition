package com.trafficsign.module.recognition.service;

import com.trafficsign.common.dto.response.SignClassInfo;
import java.util.Map;

public interface ClassMappingService {
    SignClassInfo getClassInfo(int classId);
    Map<Integer, SignClassInfo> getAllClasses();
    int getTotalClasses();
}
