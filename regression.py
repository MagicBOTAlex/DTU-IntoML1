import pandas as pd
from IPython.display import display
from sklearn.model_selection import LeaveOneOut
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Load CSV and split features and labels
df = pd.read_csv('null-corrected.csv', index_col=0)
display(df.head())
X = df.iloc[:, :-1]
Y = df.iloc[:, -1:]
display(X.head())
display(Y.head())

# Standardizing the data: subtract mean and divide by standard deviation
X_ = X - X.mean(axis=0)
X_hat = X_ / X.std(axis=0)

# Convert standardized data to torch tensors
X_tensor = torch.tensor(X_hat.values, dtype=torch.float32)
Y_tensor = torch.tensor(Y.values, dtype=torch.float32)

# Setup Leave-One-Out cross validation
loo = LeaveOneOut()
num_epochs = 1000
learning_rate = 0.01
accuracy_list = []

# Outer loop: tqdm to iterate through each LOOCV fold
for train_index, test_index in tqdm(loo.split(X_tensor), total=len(X_tensor), desc="LOO Folds"):
    # Split data into training and test sets for the current fold
    X_train = X_tensor[train_index]
    Y_train = Y_tensor[train_index]
    X_test = X_tensor[test_index]
    Y_test = Y_tensor[test_index]
    
    # Initialize a new model for each fold
    model = nn.Linear(X_tensor.shape[1], Y_tensor.shape[1])
    criterion = nn.MSELoss()  # For binary classification consider using BCEWithLogitsLoss
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # Inner loop: tqdm to show progress over epochs (set leave=False to hide after completion)
    for epoch in tqdm(range(num_epochs), leave=False, desc="Epochs"):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, Y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluate the model on the left-out sample
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test)
        # Apply sigmoid and threshold at 0.5 for binary classification
        predicted_class = (torch.sigmoid(test_pred) >= 0.5).float()
        true_class = Y_test
        # For a single sample, accuracy is either 1 or 0
        acc = (predicted_class == true_class).sum().item() / len(true_class)
        accuracy_list.append(acc)

# Calculate and display overall accuracy across all folds
overall_accuracy = sum(accuracy_list) / len(accuracy_list)
print("Overall Leave-One-Out Accuracy:", overall_accuracy)
