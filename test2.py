import numpy as np
import scipy.io
import torch
from torch.nn.utils.rnn import pad_sequence

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

    return X_train, y_train

### MAIN ###
directory_path = 'AFIRM_DATA/Final_Resultants/'
data = load_data(directory_path)
X_train, y_train = data_to_X_and_y(data, sequence_length=5)

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

# Now, check shapes
print("Shape of X_train_final:", X_train_final.shape)
print("Shape of y_train_final:", y_train_final.shape)