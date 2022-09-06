import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torchvision import transforms
import os
import pandas as pd
from torchvision.io import read_image
import torchvision.utils
import numpy as np


image_dir = "pcb_dataset/"
csv_file = "pcb_dataset.csv"
save_model_to = './pcb_components4.pth'
training = 1
epochs = 20


wanted_comps = ["bosa", "cap", "jack_ng", "m_cap", "m_jack", "m_nut", "m_rst", "m_wps", "nut_ng", "rst",  "wps"]
labels_map = {
    0: wanted_comps[0],
    1: wanted_comps[1],
    2: wanted_comps[2],
    3: wanted_comps[3],
    4: wanted_comps[4],
    5: wanted_comps[5],
    6: wanted_comps[6],
    7: wanted_comps[7],
    8: wanted_comps[8],
    9: wanted_comps[9],
    10: wanted_comps[10]
}


class pcb_dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, labels_map[self.img_labels.iloc[idx, 1]], self.img_labels.iloc[idx, 0])
        #print(f"img_path: {img_path} self.img_labels.iloc[idx, 0]: {self.img_labels.iloc[idx, 0]} 1: {self.img_labels.iloc[idx, 1]}")
        
        #sys.exit(0)
        image = read_image(img_path, mode=torchvision.io.image.ImageReadMode.RGB)
        # image = io.imre(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



transform_img = transforms.Compose([
                            #transforms.Resize(1200),
                            transforms.ToPILImage(),
                            transforms.RandomResizedCrop(size=(32,32), scale=(0.8,1)),
                            #transforms.RandomRotation((0,180)),
                            transforms.ToTensor()
])

training_data = pcb_dataset(csv_file, image_dir, transform=transform_img)

# spliting the data to training and testing with ratio 20% : 80%
train_size = int(0.8 * len(training_data))
test_size = len(training_data) - train_size

# not sure random split should be used here
train_dataset, test_dataset = torch.utils.data.random_split(training_data, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


# for X, y in test_dataloader:
#     print(f"Shape of X: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break


device = f"{torch.cuda.get_device_name()}" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 11)

    def forward(self, x):
        # Conv layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # flatting all dimensions except batch for feeding a 1d array to the linear layers 
        x = torch.flatten(x, 1) 
        #x = x.view(x.size(0), -1)

        # linear layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = NeuralNetwork().to(device)

if training:

    model.load_state_dict(torch.load(save_model_to))

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        # activates training mode
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        # activates testing mode
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), save_model_to)

    

else:
    # loading the model
    model.load_state_dict(torch.load(save_model_to))
    model.eval()
    def imshow(img):
        #img = img / 2 + 0.5     # unnormalize
        #npimg = img.cpu().numpy()
        #plt.imshow(np.transpose(npimg, (1, 2, 0)))
        img = img.permute((1,2,0))
        plt.imshow(img.cpu())
        plt.show()


    dataiter = iter(test_dataloader)
    images, labels = dataiter.next()
    images, labels = images.to(device), labels.to(device)
    img = torchvision.utils.make_grid(images)
    img = img.permute((1,2,0))
    
    print('GroundTruth: ', ' '.join(f'{labels_map[labels[j].item()]}' for j in range(4)))

    outputs = model(images)

    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join(f'{labels_map[predicted[j].item()]}'
                                for j in range(4)))

    plt.imshow(img.cpu())
    plt.show()

