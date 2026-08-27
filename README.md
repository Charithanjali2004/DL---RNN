# DL- Developing a Recurrent Neural Network Model for Stock Prediction

### Name: Kanamarlapudi Sai Charithanjali

### Register Number: 212224240069

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset
Develop a Recurrent Neural Network (RNN) model using PyTorch to predict future stock closing prices based on historical closing-price data. The historical stock data is divided into training and testing datasets. The closing prices are normalized using Min-Max Scaling, and sequential input samples of 60 previous days are created. The RNN learns the temporal patterns in these sequences and predicts the closing price for the next day. The predicted prices are then converted back to their original scale and compared with the actual stock prices.


## DESIGN STEPS
### STEP 1: 

Load Dataset – Load training and testing stock data and extract the Close price.

### STEP 2: 

Preprocess Data – Normalize closing prices using Min-Max Scaling.

### STEP 3: 

Create Sequences – Create 60-day input sequences and the next-day target price.

### STEP 4: 

Build RNN Model – Design a 2-layer RNN with 64 hidden units and a linear output layer.

### STEP 5: 

Train Model – Train the RNN using Adam optimizer and MSE loss for 20 epochs.

### STEP 6: 

Predict & Evaluate – Predict test prices, inverse-transform them, and compare actual vs predicted prices.

## PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

df_train = pd.read_csv("C:\\Users\\admin\\Downloads\\trainset.csv")
df_test = pd.read_csv("C:\\Users\\admin\\Downloads\\testset.csv")

train_prices = df_train['Close'].values.reshape(-1, 1)
test_prices = df_test['Close'].values.reshape(-1, 1)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_prices)
scaled_test = scaler.transform(test_prices)

def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(x), np.array(y)

seq_length = 60
x_train, y_train = create_sequences(scaled_train, seq_length)
x_test, y_test = create_sequences(scaled_test, seq_length)

x_train.shape, y_train.shape, x_test.shape, y_test.shape

x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(RNNModel, self).__init__()
        self.rnn=nn.RNN(input_size, hidden_size, num_layers, batch_first = True)
        self.fc=nn.Linear(hidden_size, output_size)

    def forward(self,x):
        out, _=self.rnn(x)
        out=self.fc(out[:, -1, :])
        return out

model = RNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

from torchinfo import summary
# input_size = (batch_size, seq_len, input_size)
summary(model, input_size=(64, 60, 1))

criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

## Step 3: Train the Model
epochs = 20
model.train()
train_losses = []
for epoch in range(epochs):
  epoch_loss = 0
  for x_batch, y_batch in train_loader:
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    optimizer.zero_grad()
    outputs = model(x_batch)
    loss = criterion(outputs, y_batch)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  train_losses.append(epoch_loss / len(train_loader))
  print(f"Epoch [{epoch+1}/{epochs}], Loss:{train_losses[-1]:.4f}")
# Plot training loss

plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.show()

## Step 4: Make Predictions on Test Set
model.eval()
with torch.no_grad():
    predicted = model(x_test_tensor.to(device)).cpu().numpy()
    actual = y_test_tensor.cpu().numpy()
    
# Inverse transform the predictions and actual values
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(actual)

# Plot the predictions vs actual prices
plt.figure(figsize=(10, 6))
plt.plot(actual_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price Prediction using RNN')
plt.legend()
plt.show()
print(f'Predicted Price: {predicted_prices[-1]}')
print(f'Actual Price: {actual_prices[-1]}')
```

### OUTPUT

## Training Loss Over Epochs Plot

<img width="1118" height="772" alt="image" src="https://github.com/user-attachments/assets/a693f6ad-a005-4842-8389-0ed0ca7747ac" />

## True Stock Price, Predicted Stock Price vs time

<img width="1440" height="577" alt="image" src="https://github.com/user-attachments/assets/bc7cf972-95e5-43fe-888a-1d15701958f2" />

## RESULT
Thus, a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data has been developed successfully.
