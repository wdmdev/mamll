import torch
import os

# Directory containing the .pt files
folder_path = 'predictions'

# List all .pt files in the directory
files = [file for file in os.listdir(folder_path) if file.endswith('.pt')]

# Initialize a list to hold the loaded tensors
predictions = []

# Load each tensor file
for file in files:
    file_path = os.path.join(folder_path, file)
    predictions.append(torch.load(file_path))

# Compute the mean of the predictions along the 0th dimension
mean_predictions = torch.mean(torch.stack(predictions), dim=0)

# Save the mean predictions to a new file
torch.save(mean_predictions, 'test_predictions.pt')
