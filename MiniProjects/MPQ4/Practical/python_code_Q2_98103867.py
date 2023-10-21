student_number = 98103867
Name = 'Mohammad'
Last_Name = 'Abolnejadian'

from Helper_codes.validator import *

python_code = extract_python("./Q2.ipynb")
with open(f'python_code_Q2_{student_number}.py', 'w') as file:
    file.write(python_code)

import torch
from torchvision import datasets
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import tqdm
from ae_helper import init_mnist_subset_directories

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

mnist_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())

p1 = torch.tensor([3.], requires_grad=True)
p2 = torch.tensor([7.], requires_grad=True)

L = 3 * p1**3 - 7*p2**2 + torch.sin(p1)*p2**2

def l_derivative_wrt_p1(p_1, p_2):
  return 9*p_1**2 + torch.cos(p_1)*p_2**2
def l_derivative_wrt_p2(p_1, p_2):
  return -14*p_2 + 2*p_2*torch.sin(p_1)

print(l_derivative_wrt_p1(p1, p2))
print(l_derivative_wrt_p2(p1, p2))

L.backward()

print(f"P_1 grad: {p1.grad.item()}\nP_2 grad: {p2.grad.item()}")

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
      self.p = p
  
    def __call__(self, x):
      self.x = x
      flipped = torch.flip(x, [0, 1])
      select_p = random.random()
      return flipped if select_p < self.p else x

class RandomColorSwap(object):
  def __init__ (self, p=0.5):
    self.p = p
  
  def __call__(self, x):
    self.x = x
    m = torch.max(x)
    swapped = m - x
    select_p = random.random()
    return swapped if select_p < self.p else x

trans = transforms.Compose([
  RandomHorizontalFlip(p=0.7),
  RandomColorSwap()
])

num_imgs = 10
fig, axs = plt.subplots(2, num_imgs, figsize=(25, 5))
for i, idx in enumerate(torch.randint(0, len(mnist_dataset), [num_imgs])):
    x, y = mnist_dataset[idx]
    axs[0, i].imshow(x[0], cmap='gray')
    axs[1, i].imshow(trans(x)[0], cmap='gray')
    for k in range(2):
        axs[k, i].set_yticks([])
        axs[k, i].set_xticks([])

axs[0, 0].set_ylabel("Original")
axs[1, 0].set_ylabel("Transformed");

dataset_path = "new_mnist"
init_mnist_subset_directories(mnist_dataset, dataset_path)

class MNISTDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.transform = transform
        trans = transforms.Compose([transform])
        self.dataset = []
        folders = os.listdir(root_dir)
        for folder in folders:
          label = int(folder)
          folder_images = os.listdir(root_dir + '/' + folder)
          for image in folder_images:
            pth = root_dir + '/' + folder + '/' + image
            self.dataset.append((trans(torch.load(pth)[0]), label))  
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
        pass

my_dataset = MNISTDataset(root_dir=dataset_path, transform=RandomColorSwap())
len(my_dataset)

fig, axs = plt.subplots(10, figsize=(25, 25))
for i in range(10):
  axs[i].imshow(my_dataset[random.randint(0, len(my_dataset))][0])

class DigitRecognizer(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(784, 512), #The image is 28*28
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 64),
        nn.Linear(64,32),
        nn.ReLU(),
        nn.BatchNorm1d(32),
        nn.Linear(32,10),
    )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.net(x)
    return x


model = DigitRecognizer().to(device)
model

transform_compose = transforms.Compose([
    transforms.ToTensor(),
])

mnist_dataset = datasets.MNIST(root='dataset', train=True, download=True, transform=transform_compose)
train_size = int(0.9 * len(my_dataset))
val_size = len(my_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(my_dataset, (train_size, val_size))

test_dataset = datasets.MNIST(root='dataset', train=False, download=True, transform=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 40

train_loss_arr, val_loss_arr = [], []
for epoch in range(num_epochs):
    train_loss, val_loss = 0, 0

    model.train()
    for i, (x, y) in enumerate(train_loader):
      x = x.to(device)
      y = y.to(device)
      p = model(torch.flatten(x, start_dim = 1))
      loss = criterion(p, y) * 32

      train_loss += float(loss)
      predictions = p.argmax(-1)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
  
    model.eval()
    for i, (x, y) in enumerate(val_loader):
      x = x.to(device)
      y = y.to(device)
      p = model(torch.flatten(x, start_dim = 1))
      loss = criterion(p, y) * 32

      val_loss += float(loss)
      predictions = p.argmax(-1)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    train_loss /= len(train_loader.dataset)
    val_loss /= len(val_loader.dataset)
    train_loss_arr.append(train_loss)
    val_loss_arr.append(val_loss)

    print(f"[Epoch {epoch}]\t"
        f"Train Loss: {train_loss:.4f}\t"
        f"Validation Loss: {val_loss:.4f}")

xs = np.arange(1, num_epochs+1)
plt.plot(xs, train_loss_arr)
plt.title("Train Loss")
plt.show()

plt.plot(xs, val_loss_arr)
plt.title("Validation Loss")
plt.show()

epoch_loss = 0
epoch_true = 0
epoch_all = 0
i = 0

model.eval()
with torch.no_grad(), tqdm.tqdm(enumerate(test_loader), total=len(test_loader)) as pbar:
    for i, (x, y) in pbar:
        x = x.to(device)
        y = y.to(device)
        p = model(torch.flatten(x, start_dim = 1))

        loss = criterion(p, y) * 32
        epoch_loss += float(loss)

        predictions = p.argmax(-1)
        epoch_all += len(predictions)
        epoch_true += (predictions == y).sum()

        pbar.set_description(f'Loss: {epoch_loss / (i + 1):.3e} - Acc: {epoch_true * 100. / epoch_all:.2f}%')

wrong_counted = 0
wrong_predictions = []
for i, (x, y) in enumerate(test_loader):
  if wrong_counted >= 8:
    break
  x = x.to(device)
  y = y.to(device)
  p = model(torch.flatten(x, start_dim = 1))
  predictions = p.argmax(-1)

  for j, prediction in enumerate(predictions):
    if wrong_counted >= 8:
      break
    if (prediction != y[j]):
      wrong_counted += 1
      wrong_image = x[j]
      wrong_predictions.append((prediction, y[j], wrong_image[0]))

fig, axs = plt.subplots(ncols = 10, figsize=(50, 50))
for i, wrong_prediction in enumerate(wrong_predictions):
  axs[i].imshow(wrong_prediction[2])
  axs[i].set_title(f"predicted: {wrong_prediction[0]} - Label: {wrong_prediction[1]}")

