import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision
from load_dataset import MyDataset
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from net import Net

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def val():
    classes = ('male', 'female')
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((32, 32)),
        transforms.ToTensor()])

    val_data = MyDataset('val.txt', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=4, shuffle=True, num_workers=0)
    net = Net()
    net.load_state_dict(torch.load('sex_net.pth'))

    correct = 0
    total = 0

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in val_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # collect the correct predictions for each class
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print('Accuracy of the network on the val images: %d %% ' % (
            100 * correct / total))


val()
