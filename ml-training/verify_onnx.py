import os
import sys
import json
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from PIL import Image
# pyrefly: ignore [missing-import]
import onnxruntime as ort
# pyrefly: ignore [missing-import]
from torchvision.datasets import GTSRB

# Ensure standard UTF-8 output on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

def preprocess_image(pil_img, input_size=(32, 32), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Standard preprocessing matching model training:
    RGB -> Resize(32, 32) -> Float [0, 1] -> Normalize(mean, std) -> NCHW [1, 3, 32, 32]
    """
    img = pil_img.convert('RGB').resize(input_size)
    arr = np.array(img, dtype=np.float32) / 255.0  # HWC, [0, 1]
    
    # Normalize per channel
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    arr = (arr - mean) / std
    
    # Transpose HWC to CHW, then add batch dim -> [1, 3, 32, 32]
    chw = np.transpose(arr, (2, 0, 1))
    tensor = np.expand_dims(chw, axis=0).astype(np.float32)
    return tensor

def verify_onnx_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(base_dir, 'traffic_sign_model.onnx')
    mapping_path = os.path.join(base_dir, 'class_mapping.json')
    data_dir = os.path.join(base_dir, 'data')

    if not os.path.exists(onnx_path):
        print(f"Error: ONNX model not found at {onnx_path}. Run export_onnx.py first!")
        sys.exit(1)
        
    if not os.path.exists(mapping_path):
        print(f"Error: Class mapping not found at {mapping_path}")
        sys.exit(1)

    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping_data = json.load(f)
    classes = {item['id']: item for item in mapping_data['classes']}

    print("--- Loading ONNX Inference Session ---")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    print(f"Model Input : {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"Model Output: {session.get_outputs()[0].name}, shape: {session.get_outputs()[0].shape}")

    print("\n--- Running Test Sample Predictions (Phase 1 Acceptance Criteria) ---")
    raw_test = GTSRB(root=data_dir, split='test', download=False)
    
    # Pick 5 diverse test samples
    sample_indices = [0, 50, 100, 200, 500]
    total_correct = 0

    for idx in sample_indices:
        pil_img, true_label = raw_test[idx]
        input_tensor = preprocess_image(pil_img)
        
        ort_outputs = session.run(None, {input_name: input_tensor})
        ort_arr = np.asarray(ort_outputs[0])
        logits = ort_arr[0]
        probs = softmax(logits)
        
        pred_label = int(np.argmax(probs))
        confidence = float(probs[pred_label]) * 100.0
        
        true_info = classes.get(true_label, {"name_en": "Unknown", "name_vi": "Chưa rõ"})
        pred_info = classes.get(pred_label, {"name_en": "Unknown", "name_vi": "Chưa rõ"})
        
        is_match = (true_label == pred_label)
        if is_match:
            total_correct += 1
            
        status = "[CORRECT]" if is_match else "[MISMATCH]"
        print(f"Sample #{idx:04d} {status}")
        print(f"  True Class : ID {true_label:02d} -> {true_info['name_en']} ({true_info['name_vi']})")
        print(f"  Pred Class : ID {pred_label:02d} -> {pred_info['name_en']} ({pred_info['name_vi']})")
        print(f"  Confidence : {confidence:.2f}%")
        print("-" * 60)

    print(f"\nVerification Result: {total_correct}/{len(sample_indices)} test samples correctly identified.")
    if total_correct == len(sample_indices):
        print("ALL ACCEPTANCE CRITERIA FOR PHASE 1 ONNX INFERENCE ARE MET!")
        return True
    else:
        print("Warning: Some samples did not match.")
        return False

if __name__ == "__main__":
    verify_onnx_model()
