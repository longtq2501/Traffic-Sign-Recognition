package com.trafficsign.module.recognition.service.impl;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtSession.Result;
import com.trafficsign.common.dto.response.SignClassInfo;
import com.trafficsign.common.dto.response.SignPredictionResponse;
import com.trafficsign.common.exception.ModelInferenceException;
import com.trafficsign.module.recognition.service.ClassMappingService;
import com.trafficsign.module.recognition.service.ModelInferenceService;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.io.Resource;
import org.springframework.core.io.ResourceLoader;
import org.springframework.stereotype.Service;

import java.io.InputStream;
import java.util.Collections;
import java.util.Map;

@Slf4j
@Service
@RequiredArgsConstructor
public class ModelInferenceServiceImpl implements ModelInferenceService {

    private final ResourceLoader resourceLoader;
    private final ClassMappingService classMappingService;

    @Value("${app.model.onnx-path:classpath:model/traffic_sign_model.onnx}")
    private String modelPath;

    private OrtEnvironment env;
    private OrtSession session;
    private String inputName;
    private String outputName;

    @PostConstruct
    public void init() {
        log.info("Initializing ONNX Runtime environment and loading model from: {}", modelPath);
        try {
            Resource resource = resourceLoader.getResource(modelPath);
            if (!resource.exists()) {
                throw new ModelInferenceException("ONNX model file not found at: " + modelPath);
            }

            byte[] modelBytes;
            try (InputStream is = resource.getInputStream()) {
                modelBytes = is.readAllBytes();
            }

            this.env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);

            this.session = env.createSession(modelBytes, options);
            this.inputName = session.getInputNames().iterator().next();
            this.outputName = session.getOutputNames().iterator().next();

            log.info("ONNX Model loaded successfully. Input: '{}', Output: '{}'", inputName, outputName);
        } catch (Exception e) {
            log.error("Failed to load ONNX model into memory", e);
            throw new ModelInferenceException("Failed to initialize ONNX Runtime: " + e.getMessage(), e);
        }
    }

    @Override
    public SignPredictionResponse predict(float[][][][] inputTensor) {
        if (inputTensor == null || inputTensor.length == 0) {
            throw new ModelInferenceException("Input tensor must not be empty");
        }
        if (inputTensor[0].length != 3 || inputTensor[0][0].length != 32 || inputTensor[0][0][0].length != 32) {
            throw new ModelInferenceException("Invalid tensor dimensions. Expected NCHW [batch, 3, 32, 32]");
        }

        long startTime = System.currentTimeMillis();
        OnnxTensor tensor = null;
        Result results = null;

        try {
            tensor = OnnxTensor.createTensor(env, inputTensor);
            Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
            results = session.run(inputs);

            float[][] outputLogits = (float[][]) results.get(0).getValue();
            float[] logits = outputLogits[0];

            SoftmaxResult softmax = calculateSoftmax(logits);
            long inferenceTimeMs = System.currentTimeMillis() - startTime;

            SignClassInfo classInfo = classMappingService.getClassInfo(softmax.classId);

            return SignPredictionResponse.builder()
                    .classId(softmax.classId)
                    .signNameEn(classInfo.getNameEn())
                    .signNameVi(classInfo.getNameVi())
                    .category(classInfo.getCategory())
                    .confidence(Math.round(softmax.confidence * 100.0) / 100.0)
                    .inferenceTimeMs(inferenceTimeMs)
                    .build();

        } catch (OrtException e) {
            log.error("ONNX Runtime execution error during inference", e);
            throw new ModelInferenceException("Inference failed: " + e.getMessage(), e);
        } finally {
            if (results != null) {
                results.close();
            }
            if (tensor != null) {
                tensor.close();
            }
        }
    }

    @Override
    public SignPredictionResponse predictSingle(float[][][] singleImageTensor) {
        if (singleImageTensor == null) {
            throw new ModelInferenceException("Single image tensor cannot be null");
        }
        float[][][][] batch = new float[1][][][];
        batch[0] = singleImageTensor;
        return predict(batch);
    }

    @Override
    public String getInputName() {
        return inputName;
    }

    @Override
    public String getOutputName() {
        return outputName;
    }

    @PreDestroy
    public void destroy() {
        log.info("Closing ONNX Runtime session and environment.");
        try {
            if (session != null) {
                session.close();
            }
            if (env != null) {
                env.close();
            }
        } catch (Exception e) {
            log.warn("Error releasing ONNX resources", e);
        }
    }

    private SoftmaxResult calculateSoftmax(float[] logits) {
        float max = Float.NEGATIVE_INFINITY;
        for (float val : logits) {
            if (val > max) {
                max = val;
            }
        }

        double sum = 0.0;
        for (float val : logits) {
            sum += Math.exp(val - max);
        }

        int bestClass = 0;
        double maxProb = -1.0;
        for (int i = 0; i < logits.length; i++) {
            double prob = Math.exp(logits[i] - max) / sum;
            if (prob > maxProb) {
                maxProb = prob;
                bestClass = i;
            }
        }

        return new SoftmaxResult(bestClass, maxProb * 100.0);
    }

    private record SoftmaxResult(int classId, double confidence) {}
}
