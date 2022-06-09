import cv2 
import numpy as np
import torch
import torch.nn.functional as F
from model import CNN
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# import the trained model for predictions

trained_model = torch.load('/Users/szokirov/Documents/GitHub/MNIST-CNN/model_trained.pth')
model = CNN()
model.load_state_dict(trained_model)

#Plotting function
def view_classify(img, ps):

    ps = ps.cpu().data.numpy().squeeze()
    img= img.cpu().data.numpy()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.reshape(1, 28, 28).squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

# function to show images
def imshow(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(15, 15))
    plt.imshow(image)


def predict(image=str, path_save=str):
    img = cv2.imread(image)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    retval, dst= cv2.threshold(gray_img, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, h = cv2.findContours(dst, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key = cv2.contourArea, reverse=True)
    copy = img.copy()
    x, y, w, h = cv2.boundingRect(sorted_contours[0])
    x = x-300
    y=y-400
    w = w+600
    h = h+600
    # cv2.rectangle(copy, (x,y), (x+w, y+h), (0, 255, 0), 2)
    number_itself = img[y:y+h, x:x+w]
    number_itself = cv2.resize(number_itself, (28, 28))
    gray_n=number_itself.copy()
    gray_n=cv2.cvtColor(gray_n, cv2.COLOR_BGR2GRAY)
    retval, dst= cv2.threshold(gray_n, 0,255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cv2.imwrite(path_save, dst)
    # imshow(dst)
    img_transformer=transforms.Compose([transforms.Grayscale(),transforms.Resize((28, 28)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[(0.5)], std=[(0.5)])])

    dst=Image.open(path_save)
    input_tensor= img_transformer(dst)
    input_batch=input_tensor.unsqueeze(0)

    output=model.forward(input_batch)
    probability=F.softmax(output, dim=1)
    result = torch.argmax(probability, dim=1)
    probability = probability.cpu().data.numpy().squeeze()
    
    print(f"The number is {result.item()}")

    plt.subplot(1,2,1)
    plt.imshow(dst)
    plt.subplot(1,2,2)
    plt.barh(np.arange(10), probability)
    plt.yticks(np.arange(10))
    plt.title('Class Probability')
    plt.xlim(0, 1.1)
    plt.show()

predict('/Users/szokirov/Documents/GitHub/opencv_projects/handwritten_digits/8.jpeg', 'handwritten_digits/predd.png')
