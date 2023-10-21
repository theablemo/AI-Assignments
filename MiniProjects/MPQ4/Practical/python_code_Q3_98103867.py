student_number = 98103867
Name = 'Mohammad'
Last_Name = 'Abolnejadian'

from Helper_codes.validator import *

python_code = extract_python("./Q3.ipynb")
with open(f'python_code_Q3_{student_number}.py', 'w') as file:
    file.write(python_code)

from ae_helper import get_data
from sklearn.model_selection import train_test_split

X, Y, y = get_data()

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(X, Y, y, test_size=0.2, random_state=17)
X_train, X_val, Y_train, Y_val, y_train, y_val = train_test_split(X_train, Y_train, y_train, test_size=0.1, random_state=17)

import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pathlib

device = torch.device('cpu')
device

class AutoEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 9)
    )

    self.decoder = torch.nn.Sequential(
            torch.nn.Linear(9, 18),
            torch.nn.ReLU(),
            torch.nn.Linear(18, 36),
            torch.nn.ReLU(),
            torch.nn.Linear(36, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 28 * 28),
            torch.nn.Sigmoid()
        )
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class MnistNextDigitDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, ...]:
        return self.X[i], self.Y[i], self.y[i]

train_dataloader = DataLoader(
    MnistNextDigitDataset(X_train, Y_train, y_train),
    batch_size=512,
    shuffle=True
)
val_dataloader = DataLoader(
    MnistNextDigitDataset(X_val, Y_val, y_val),
    batch_size=1024,
    shuffle=False
)

model = AutoEncoder().to(device)
loss_function = torch.nn.MSELoss()
learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100

train_loss_arr, val_loss_arr = [], []

best_loss = float("inf")
best_model = None

for epoch in range(num_epochs):
  train_loss, val_loss = 0, 0

  model.train()

  for i, (x, y, _) in enumerate(train_dataloader):
    x = x.to(device)
    y = y.to(device)
    p = model(torch.flatten(x, start_dim = 1))
    loss = loss_function(p, torch.flatten(y, start_dim = 1))

    train_loss += float(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

  model.eval()
  for i, (x, y, _) in enumerate(val_dataloader):
    x = x.to(device)
    y = y.to(device)
    p = model(torch.flatten(x, start_dim = 1))
    loss = loss_function(p, torch.flatten(y, start_dim = 1))

    val_loss += float(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    

  train_loss /= len(train_dataloader.dataset)
  val_loss /= len(val_dataloader.dataset)
  epoch_loss = (train_loss + val_loss)/2
  if epoch_loss < best_loss:
    best_loss = epoch_loss
    best_model = model
  train_loss_arr.append(train_loss)
  val_loss_arr.append(val_loss)

  print(f"[Epoch {epoch}]\t"
      f"Train Loss: {train_loss:.8f}\t"
      f"Validation Loss: {val_loss:.8f}")

torch.save(best_model.state_dict(), 'state_dict_model.pt')

trained_model = AutoEncoder().to(device)
trained_model.load_state_dict(torch.load('state_dict_model.pt'))
trained_model

test_dataloader = DataLoader(
    MnistNextDigitDataset(X_test, Y_test, y_test),
    batch_size=40,
    shuffle=True
)

trained_model.eval()

for j, (x, _, __) in enumerate(test_dataloader):
  x = x.to(device)
  p = trained_model(torch.flatten(x, start_dim = 1))
  fig, axes = plt.subplots(10, 8, figsize = (15,15))
  counter = 0
  for i in range(40):
    axes[int(counter/8)][counter%8].imshow(x[i])
    axes[int(counter/8)][counter%8].set_title("input")
    axes[int(counter/8)][counter%8].axis('off')
    counter += 1
    axes[int(counter/8)][counter%8].imshow(p[i].reshape([28, 28]).detach().numpy())
    axes[int(counter/8)][counter%8].set_title("output")
    axes[int(counter/8)][counter%8].axis('off')
    counter += 1
  break

