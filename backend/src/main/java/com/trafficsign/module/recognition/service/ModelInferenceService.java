package com.trafficsign.module.recognition.service;

import com.trafficsign.common.dto.response.SignPredictionResponse;

public interface ModelInferenceService {
    /**
     * Runs inference on a 4D NCHW tensor [batch, channels=3, height=32, width=32].
     *
     * @param inputTensor Normalized float tensor
     * @return Prediction response with predicted class and confidence
     */
    SignPredictionResponse predict(float[][][][] inputTensor);

    /**
     * Convenience method for a single 3D image tensor [channels=3, height=32, width=32].
     *
     * @param singleImageTensor Normalized float tensor for 1 image
     * @return Prediction response
     */
    SignPredictionResponse predictSingle(float[][][] singleImageTensor);

    /**
     * Gets model input tensor name.
     */
    String getInputName();

    /**
     * Gets model output tensor name.
     */
    String getOutputName();
}
