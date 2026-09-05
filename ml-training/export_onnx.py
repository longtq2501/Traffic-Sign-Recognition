import os
import sys
import torch
import numpy as np
import onnx
import onnxruntime as ort
from model import TrafficSignCNN

def export_to_onnx(model_path=None, output_onnx_path=None, opset_version=17):
    """
    Exports trained PyTorch TrafficSignCNN to standard ONNX format.
    Ensures NCHW [batch_size, 3, 32, 32] format with dynamic batch axis.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(base_dir, 'best_model.pth')
    if output_onnx_path is None:
        output_onnx_path = os.path.join(base_dir, 'traffic_sign_model.onnx')
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    print(f"Loading checkpoint from: {model_path}")
    device = torch.device('cpu')
    model = TrafficSignCNN(num_classes=43).to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32, dtype=torch.float32, device=device)

    print(f"Exporting model to ONNX (Opset: {opset_version})...")
    export_kwargs = {
        "export_params": True,
        "opset_version": opset_version,
        "do_constant_folding": True,
        "input_names": ['input'],
        "output_names": ['output'],
        "dynamic_axes": {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    }
    
    # Use legacy TorchScript exporter if dynamo parameter is available
    try:
        torch.onnx.export(model, (dummy_input,), output_onnx_path, dynamo=False, **export_kwargs)
    except TypeError:
        torch.onnx.export(model, (dummy_input,), output_onnx_path, **export_kwargs)

    print(f"Model exported successfully to: {output_onnx_path}")
    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print(f"ONNX Model File Size: {file_size_mb:.2f} MB")

    # Step 1: Validate ONNX graph structure
    print("\n--- Validating ONNX Model Integrity ---")
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker validation: PASSED (Graph is structurally sound)")

    # Step 2: Validate Numerical Equivalence with ONNX Runtime
    print("\n--- Validating Numerical Equivalence (PyTorch vs ONNX Runtime) ---")
    ort_session = ort.InferenceSession(output_onnx_path, providers=['CPUExecutionProvider'])
    
    # Test Single Input (Batch = 1)
    with torch.no_grad():
        torch_out = model(dummy_input).numpy()
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_out = np.asarray(ort_session.run(None, ort_inputs)[0])
    
    max_diff = np.max(np.abs(torch_out - ort_out))
    print(f"Single item inference max absolute diff: {max_diff:.6e}")
    np.testing.assert_allclose(torch_out, ort_out, rtol=1e-3, atol=1e-4,
                               err_msg="Discrepancy detected between PyTorch and ONNX Runtime outputs!")
    print("Batch size 1 verification: PASSED")

    # Test Dynamic Batching (Batch = 4)
    dummy_batch = torch.randn(4, 3, 32, 32, dtype=torch.float32, device=device)
    with torch.no_grad():
        torch_batch_out = model(dummy_batch).numpy()
    ort_batch_inputs = {ort_session.get_inputs()[0].name: dummy_batch.numpy()}
    ort_batch_out = np.asarray(ort_session.run(None, ort_batch_inputs)[0])
    
    max_batch_diff = np.max(np.abs(torch_batch_out - ort_batch_out))
    print(f"Dynamic batch (N=4) inference max absolute diff: {max_batch_diff:.6e}")
    np.testing.assert_allclose(torch_batch_out, ort_batch_out, rtol=1e-3, atol=1e-4,
                               err_msg="Discrepancy detected on dynamic batching!")
    print("Dynamic batching verification: PASSED")

    print("\n=======================================================")
    print("ONNX EXPORT SUMMARY:")
    print(f"  - Model File   : {output_onnx_path}")
    print(f"  - Input Tensor : '{ort_session.get_inputs()[0].name}' (Shape: {ort_session.get_inputs()[0].shape})")
    print(f"  - Output Tensor: '{ort_session.get_outputs()[0].name}' (Shape: {ort_session.get_outputs()[0].shape})")
    print(f"  - Format       : NCHW [Batch, Channels=3, Height=32, Width=32]")
    print(f"  - Classes      : 43")
    print("=======================================================\n")
    return output_onnx_path

if __name__ == "__main__":
    export_to_onnx()
