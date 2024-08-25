import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        # Apply the linear layer to each time step
        out = self.linear(out)  # This will now be applied to each time step
        return out

# Function to load data from .mat files and truncate/pad each to 150 samples
def load_data(path):
    exclude_indices = {1, 3, 5, 7, 9, 10, 12, 15, 16, 17}  # Indices to exclude
    all_data = []

    # Loop over the range of indices (assuming files are named from 0 to 53)
    for i in range(54):
        if i not in exclude_indices:
            # Load data from the .mat file
            mat_data = scipy.io.loadmat(f'{path}final_resultant{i}.mat')
            resultant_waveform = mat_data['final_resultant'].flatten()

            # Truncate or pad the waveform to exactly 150 samples
            if len(resultant_waveform) > 150:
                resultant_waveform = resultant_waveform[:150]
            else:
                resultant_waveform = np.pad(resultant_waveform, (0, 150 - len(resultant_waveform)), 'constant')

            all_data.append(resultant_waveform.tolist())  # Convert numpy array to list

    return all_data

# Function to prepare training data (X_train, y_train) from wave data
def data_to_X_and_y(data, sequence_length=5):
    X_train = []
    y_train = []

    # Iterate through each wave in the data
    for wave_index, wave in enumerate(data):
        # Make sure each wave has exactly 150 samples
        if len(wave) != 150:
            print(f"Wave {wave_index} length error: {len(wave)} samples")
            continue

        # Create sequences and corresponding labels
        sequences = [wave[i:i + sequence_length] for i in range(len(wave) - sequence_length)]
        labels = [wave[i + sequence_length] for i in range(len(wave) - sequence_length)]

        # Convert sequences and labels from numpy arrays to lists
        sequences_list = [list(seq) for seq in sequences]
        labels_list = labels  # Already a list if 'wave' is a list

        X_train.append(sequences_list)
        y_train.append(labels_list)
   
    # convert to tensors because of issue with numpy to nested list conversions
    X_train_final, y_train_final = convert_to_tensors(X_train, y_train)

    return X_train_final, y_train_final

def convert_to_tensors(X_train, y_train):
    # Convert lists of lists of lists to lists of tensors
    X_train_tensors = [torch.tensor(sequence) for sequence in X_train]
    y_train_tensors = [torch.tensor(labels) for labels in y_train]

    # Pad the sequences so each is the same length
    # We pad with zeros which is a common choice for padding
    X_train_padded = pad_sequence(X_train_tensors, batch_first=True, padding_value=0)
    y_train_padded = pad_sequence(y_train_tensors, batch_first=True, padding_value=0)

    # Since pad_sequence pads based on the longest sequence in the batch, you may need to trim if the longest
    # sequence is longer than 145, or further pad if it's shorter.
    # Ensuring they are exactly 145 sequences long:
    X_train_final = X_train_padded[:, :145, :5]  # Ensures every sample has exactly 145 sequences of length 5
    y_train_final = y_train_padded[:, :145]      # Ensures every sample has exactly 145 labels

    return X_train_final, y_train_final

def check_shape(X_train_final, y_train_final):
    # Now, check shapes
    print("Shape of X_train_final:", X_train_final.shape)
    print("Shape of y_train_final:", y_train_final.shape)

    # Extract the first wave's sequences and labels
    first_wave_sequences = X_train_final[0]
    first_wave_labels = y_train_final[0]
   
    print("Sequence: \t Label:")
    for sequence, label in zip(first_wave_sequences, first_wave_labels):
        print(f"{sequence.tolist()} \t {label.item()}")

# Normalize and scale the data
def scale_data(data):
    max_val = data.max()
    min_val = data.min()
    normalized_data = 2 * (data - min_val) / (max_val - min_val) - 1
    return normalized_data, max_val, min_val

# MAIN
directory_path = 'AFIRM_DATA/Final_Resultants/'
data = load_data(directory_path)
X_train_final, y_train_final = data_to_X_and_y(data)

# Normalize inputs and outputs
X_train_normalized, max_val_X, min_val_X = scale_data(X_train_final.float())
Y_train_normalized, max_val_Y, min_val_Y = scale_data(y_train_final.float())

# Extract data for a specific waveform
index = 19
X_train_wave1 = X_train_normalized[index]
Y_train_wave1 = Y_train_normalized[index]

model = LSTMModel(5, 50, 1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
num_epochs = 30

# Train the model
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train_wave1.unsqueeze(0))
    loss = criterion(outputs.squeeze(), Y_train_wave1)
    loss.backward()
    optimizer.step()

# Predicting after training
predicted_waveform = model(X_train_wave1.unsqueeze(0)).squeeze().detach()
predicted_waveform = (predicted_waveform + 1) / 2 * (max_val_Y - min_val_Y) + min_val_Y
Y_train_wave1 = (Y_train_wave1 + 1) / 2 * (max_val_Y - min_val_Y) + min_val_Y

# Plotting the results
plt.figure(figsize=(12, 6))
plt.plot(predicted_waveform.numpy(), label='Predicted', marker='o')
plt.plot(Y_train_wave1.numpy(), label='Labelled', marker='x')
plt.title('Comparison of Predicted and Labelled Values')
plt.xlabel('Sequence Index')
plt.ylabel('Waveform Value')
plt.legend()
plt.grid(True)
plt.show()