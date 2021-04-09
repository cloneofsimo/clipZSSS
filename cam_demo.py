import cv2
import torch
import torchvision.models as models
from CAMERAS import CAMERAS
from torchvision import transforms
import matplotlib.pyplot as plt

normalizeTransform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
normalizeImageTransform = transforms.Compose([transforms.ToTensor(), normalizeTransform])

def loadImage(imagePath, imageSize):
    rawImage = cv2.imread(imagePath)
    rawImage = cv2.resize(rawImage, (224,) * 2, interpolation=cv2.INTER_LINEAR)
    rawImage = cv2.resize(rawImage, (imageSize,) * 2, interpolation=cv2.INTER_LINEAR)
    image = normalizeImageTransform(rawImage[..., ::-1].copy())
    return image, rawImage


if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    model.eval()
    model = model.cuda()

    cameras = CAMERAS(model, targetLayerName="layer4")
    file = "./cat_dog.png"

    image, rawImage = loadImage(file, imageSize=224)
    image = torch.unsqueeze(image, dim=0)

    saliencyMap = cameras.run(image, classOfInterest=243).cpu()
    print(saliencyMap, saliencyMap.mean(), saliencyMap.max(), saliencyMap.min())
    plt.imshow(saliencyMap)
    plt.show()
