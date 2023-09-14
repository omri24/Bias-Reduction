import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# This line prevents randomness in the results
torch.manual_seed(0)

# The number of samples in the test csv:
length_param = np.loadtxt("exported_test.csv", delimiter=",", dtype=np.float32).shape[0]

input_size = length_param
hidden_size = 50
output = 1
learning_rate = 0.00001
num_epochs = 100
min_epoch_to_break = 20
loss_breaker = 10 ** (-8)
time_bound = -1
modulu_param = 1


# This part chooses GPU if possible and CPU if not
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SampleDataset(Dataset):

    def __init__(self, file_name):
        csv = np.loadtxt(file_name, delimiter=",", dtype=np.float32)
        rows_to_remove = csv.shape[0] % length_param
        d = int(csv.shape[0] / length_param)
        model_outs_reshaped = np.reshape(csv[rows_to_remove:, 0], (length_param, d))
        real_values_reshaped = np.reshape(csv[rows_to_remove:, 1], (length_param, d))
        average_model_outs = np.mean(model_outs_reshaped, axis=0, dtype=np.float32, keepdims=True)
        average_real_values = np.mean(real_values_reshaped, axis=0, dtype=np.float32, keepdims=True)
        biases = average_real_values - average_model_outs
        self.x = torch.from_numpy(model_outs_reshaped)
        self.y = torch.from_numpy(biases)
        self.n = model_outs_reshaped.shape[1]
        self.m = biases.shape[1]

    def __len__(self):
        if (self.m != self.n):
            raise IndexError("There are " + str(self.n) + " inputs (x for NN) but " + str(self.m) + " outputs (y)")
        else:
            return self.n

    def __getitem__(self, index):
        return (self.x[:, index], self.y[:, index])

train_dataset = SampleDataset(file_name="exported_train.csv")

# FNN structure:
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size,hidden_size)
        self.l4 = nn.Linear(hidden_size, output)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        out = self.relu(out)
        out = self.l4(out)
        return out

model = NeuralNet(input_size=input_size, hidden_size=hidden_size, output=output)

# Loss (cost function) and optimizer
criterion = nn.L1Loss()  # For this algorithm, L1 norm is the most reasonable loss func
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # try using different optimizers

for epoch in range(num_epochs):
    if (epoch > min_epoch_to_break) and (loss <= loss_breaker):
        break
    for i, (present_data, future_data) in enumerate(train_dataset):
        if (i > time_bound) and (i % modulu_param == 0):
            present_data = present_data.to(device)
            future_data = future_data.to(device)

            # Forward pass
            model_out = model(present_data)
            loss = criterion(model_out, future_data)
            if loss <= loss_breaker:
                break

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print("Epoch = " + str(epoch + 1) + ", step = " + str(i + 1) + ", loss = " + str(loss.item()))

test_csv = np.loadtxt("exported_test.csv", delimiter=",", dtype=np.float32)
data_to_fix_bias = torch.from_numpy(test_csv[:,0])
with torch.no_grad():
    final_bias = model(data_to_fix_bias)

predictions_before_bias_fix = [i for i in test_csv[:,0]]
predictions_after_bias_fix = [i + final_bias.item() for i in test_csv[:,0]]
real_values_to_plot = [i for i in test_csv[:, 1]]
x_values_for_plot = [i + 1 for i in range(test_csv.shape[0])]
real_bias = (sum(real_values_to_plot) / len(real_values_to_plot)) -\
            (sum(predictions_before_bias_fix) / len(predictions_before_bias_fix))
print("Predicted bias for test dataset = " + str(final_bias.item()))
print("Real bias of the test dataset = " + str(real_bias))
shift = (sum(real_values_to_plot) / len(real_values_to_plot)) -\
            (sum(predictions_after_bias_fix) / len(predictions_after_bias_fix))
print("Bias after correction = " + str(shift))
print("Bias improvement/deterioration = " + str(abs(real_bias) - abs(shift)))
print("Positive value = improvement, negative value = deterioration")
plt.scatter(x=x_values_for_plot, y=predictions_after_bias_fix, c="r", label="Predictions")
plt.scatter(x=x_values_for_plot, y=real_values_to_plot, c="b", label="Real data")
plt.legend()
plt.title("Predictions + Estimated Bias and Real Values")
plt.xlabel("discrete time")
plt.ylabel("EU price in USD")
plt.show()
plt.scatter(x=x_values_for_plot, y=predictions_after_bias_fix, c="r", label="Predictions")
plt.legend()
plt.title("Predictions + Estimated Bias")
plt.xlabel("discrete time")
plt.ylabel("EU price in USD")
plt.show()
plt.scatter(x=x_values_for_plot, y=real_values_to_plot, c="b", label="Real data")
plt.legend()
plt.title("Real Values")
plt.xlabel("discrete time")
plt.ylabel("EU price in USD")
plt.show()
plt.scatter(x=x_values_for_plot, y= [i + shift for i in predictions_after_bias_fix], c="r", label="Predictions")
plt.scatter(x=x_values_for_plot, y=real_values_to_plot, c="b", label="Real data")
plt.legend()
plt.title("Predictions- with nullified bias and Real Values")
plt.xlabel("discrete time")
plt.ylabel("EU price in USD")
plt.show()
