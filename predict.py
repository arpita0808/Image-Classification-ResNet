import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ['airplane','automobile','bird','cat','deer',
           'dog','frog','horse','ship','truck']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 10)

model.load_state_dict(torch.load("resnet_model.pth", map_location=device))
model = model.to(device)
model.eval()

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        probs = F.softmax(output, dim=1)

    top3 = torch.topk(probs, 3)

    print("Top Predictions:")

    for i in range(3):
        idx = top3.indices[0][i].item()
        conf = top3.values[0][i].item()
        print(f"{classes[idx]}: {conf*100:.2f}%")

    pred = probs.argmax(1).item()
    confidence = probs[0][pred].item()

    plt.imshow(image)
    plt.title(f"{classes[pred]} ({confidence*100:.2f}%)")
    plt.axis("off")
    plt.show()

predict_image("test.jpg")
