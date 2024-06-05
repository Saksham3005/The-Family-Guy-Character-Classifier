import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image

checkpoint = torch.load('model_best_checkpoint.pth.tar')

mean = [0.6922, 0.6629, 0.6335]
std = [0.2564, 0.2448, 0.2510]


resnet = models.resnet18()
num_ftrs = resnet.fc.in_features
num_cls = 4
resnet.fc = nn.Linear(num_ftrs, num_cls)
resnet.load_state_dict(checkpoint['model'])

torch.save(resnet, 'best_model.pth')

classes = [
    "Brain Griffin",
    "Lois Griffin", 
    "Peter Griffin",
    "Stewie Griffin"
]
model = torch.load('best_model.pth')

image_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image).float()
    image = image.unsqueeze(0)

    output = model(image)
    _, predicted = torch.max(output.data, 1)


    print("The image you provided shows", classes[predicted.item()])


filename = input("Enter the file path:  ")

classify(resnet, image_transforms, filename, classes)