import os, torch, onnx, numpy as np
import onnxruntime as ort

BASE_DIR = r'd:\traffic-sign-recognition\ml-training'
PTH  = os.path.join(BASE_DIR, 'best_model_vn.pth')
ONNX = os.path.join(BASE_DIR, 'traffic_sign_model.onnx')
NUM_CLASSES = 15

class TrafficSignCNN(torch.nn.Module):
    def __init__(self, num_classes=15):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(True),
            torch.nn.Conv2d(32,64,3,padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2), torch.nn.Dropout2d(0.25),
            torch.nn.Conv2d(64,128,3,padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            torch.nn.Conv2d(128,128,3,padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2), torch.nn.Dropout2d(0.25),
        )
        with torch.no_grad():
            flat = self.features(torch.zeros(1,3,32,32)).view(1,-1).size(1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(flat,256), torch.nn.BatchNorm1d(256), torch.nn.ReLU(True), torch.nn.Dropout(0.5),
            torch.nn.Linear(256,num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))

model = TrafficSignCNN(NUM_CLASSES)
model.load_state_dict(torch.load(PTH, map_location='cpu', weights_only=True))
model.eval()

dummy = torch.randn(1,3,32,32)
try:
    torch.onnx.export(model,(dummy,),ONNX,dynamo=False,export_params=True,opset_version=17,
                      input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})
except TypeError:
    torch.onnx.export(model,(dummy,),ONNX,export_params=True,opset_version=17,
                      input_names=['input'],output_names=['output'],
                      dynamic_axes={'input':{0:'batch_size'},'output':{0:'batch_size'}})

onnx.checker.check_model(onnx.load(ONNX))
sess = ort.InferenceSession(ONNX)
pt_out = model(dummy).detach().numpy()
ort_out = sess.run(None,{'input':dummy.numpy()})[0]
diff = float(np.max(np.abs(pt_out-ort_out)))
print(f'Max diff PyTorch vs ONNX: {diff:.2e}')
print(f'ONNX exported: {ONNX}  ({os.path.getsize(ONNX)/1024/1024:.2f} MB)')
