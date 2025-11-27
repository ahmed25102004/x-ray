import torch
from torchvision import transforms, models
from torch import nn
import pickle
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Linear(in_features=512, out_features=2, bias=True)
model.load_state_dict(torch.load(r"data\models\model.pth", map_location=device, weights_only=True))
model.to(device)
model.eval()

# Simple class mapping (adjust based on your training classes)
class_names = ['NORMAL', 'PNEUMONIA']  # Common for chest X-ray classification

def infer(img):
    img_gray = img.convert("L")
    transform = transforms.ToTensor()
    img_trans = transform(img_gray).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(img_trans)
        clas_index = torch.argmax(preds).item()
    
    # Use simple class mapping
    if clas_index < len(class_names):
        return class_names[clas_index]
    else:
        return f"Class_{clas_index}"

# Test
img_path = r"data\resized\train_NORMAL_NORMAL2-IM-1188-0001.jpeg"
if os.path.exists(img_path):
    img = Image.open(img_path)
    preds = infer(img)
    print(f"Prediction: {preds}")
else:
    print(f"Image file not found: {img_path}")