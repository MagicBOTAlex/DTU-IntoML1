import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from dtuimldmtools import draw_neural_net, train_neural_net

# ----------------------
# Data Loading and Preprocessing
# ----------------------

# Load CSV file with data
df = pd.read_csv('null-corrected.csv', index_col=0)

# One-hot encoding for categorical variables (dropping the first category to avoid multicollinearity)
df = pd.get_dummies(df, columns=['cp', 'thal', 'restecg', 'slope'], drop_first=True, dtype=float)

# Separate features and target
# Note: df.drop(columns='num') returns only the input features
X = df.drop(columns='num').to_numpy()
y = df['num'].to_numpy()

# Standardizing input data (subtract mean and divide by standard deviation)
N = len(X)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std

# Store the number of input features; this avoids the variable being overwritten later.
n_features = X_scaled.shape[1]

# Standardizing target variable
N_y = len(y)
y_mean = y.mean()
y_std = y.std()
# Reshape target to have one column (shape: N x 1)
Y_scaled = ((y - y_mean) / y_std).reshape(-1, 1)

# Retrieve attribute names for later use (diagram)
attributeNames = list(df.drop(columns='num').columns)

# ----------------------
# Cross-validation Setup
# ----------------------

K1 = 10  # outer folds
K2 = 5   # inner folds

CV_outer = KFold(n_splits=K1, shuffle=True, random_state=42)
folds = list(CV_outer.split(X_scaled))

# Set the device to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Convert data to torch tensors and move them to the chosen device
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_scaled, dtype=torch.float32).to(device)

# ----------------------
# Training Parameters and Containers for Results
# ----------------------

max_iter = 1000
n_replicates = 1  # Number of networks trained per fold
loss_fn = torch.nn.MSELoss()  # Mean-squared error loss

# Define candidate numbers for hidden units
hidden_units_candidates = [1, 2, 4, 8, 16]

# Lists to store errors and optimal number of hidden neurons for each outer fold
ANN_errors = []
n_hidden_layer_error = np.zeros((K1, len(hidden_units_candidates)))
best_n_hidden_neurons = []

# Setup plotting figures for learning curves and error rates
summaries, summaries_axes = plt.subplots(1, 2, figsize=(10, 5))
color_list = [
    "tab:orange", "tab:green", "tab:purple", "tab:brown", "tab:pink", 
    "tab:gray", "tab:olive", "tab:cyan", "tab:red", "tab:blue"
]

# ----------------------
# Outer Cross-Validation Loop
# ----------------------

for outer_fold_idx, (outer_train_index, outer_test_index) in enumerate(folds, start=0):
    print("\nOuter cross-validation fold: {0}/{1}".format(outer_fold_idx + 1, K1))
    # Split the data for this outer fold
    X_train_outer_tensor = X_tensor[outer_train_index]
    X_test_outer_tensor  = X_tensor[outer_test_index]
    Y_train_outer_tensor = Y_tensor[outer_train_index]
    Y_test_outer_tensor  = Y_tensor[outer_test_index]

    # Inner cross-validation setup on the outer training set
    CV_inner = KFold(n_splits=K2, shuffle=True, random_state=42)
    inner_folds = list(CV_inner.split(X_train_outer_tensor))

    for inner_fold_idx, (inner_train_index, validation_index) in enumerate(inner_folds, start=0):
        print("  Inner cross-validation fold: {0}/{1}".format(inner_fold_idx + 1, K2))
        # Use the global X_tensor and Y_tensor indexing if your inner indices match the positions in X_train_outer_tensor.
        # Alternatively, you could index from X_train_outer_tensor itself.
        X_train_inner_tensor = X_tensor[inner_train_index]
        X_validate_tensor = X_tensor[validation_index]
        Y_train_inner_tensor = Y_tensor[inner_train_index]
        Y_validate_tensor = Y_tensor[validation_index]

        # Test different numbers of hidden neurons
        for candidate_idx, n_hidden_neuron in enumerate(hidden_units_candidates):
            print(f"    Testing with {n_hidden_neuron} hidden neuron(s)")
            # Define the model with the correct input dimension (n_features)
            model_fn = lambda: torch.nn.Sequential(
                torch.nn.Linear(n_features, n_hidden_neuron),
                torch.nn.Tanh(),
                torch.nn.Linear(n_hidden_neuron, 1)
            ).to(device)

            # Train network on the inner training data
            net, final_loss, learning_curve = train_neural_net(
                model_fn,
                loss_fn,
                X=X_train_inner_tensor,
                y=Y_train_inner_tensor,
                n_replicates=n_replicates,
                max_iter=max_iter,
            )

            # Evaluate on inner validation set
            y_validate_estimate = net(X_validate_tensor)
            squared_error = (y_validate_estimate.float() - Y_validate_tensor.float()) ** 2
            mse = squared_error.mean().item()

            # Accumulate the error (averaging over outer folds later)
            n_hidden_layer_error[outer_fold_idx, candidate_idx] += mse / K1

    # Determine the best number of hidden neurons for this outer fold
    best_candidate_index = np.argmin(n_hidden_layer_error[outer_fold_idx])
    best_n_hidden_neuron = hidden_units_candidates[best_candidate_index]
    best_n_hidden_neurons.append(best_n_hidden_neuron)
    print(f"Best hidden neurons for outer fold {outer_fold_idx + 1}: {best_n_hidden_neuron}")

    # Train the network on the outer training set using the optimal hyperparameter found
    model_fn = lambda: torch.nn.Sequential(
        torch.nn.Linear(n_features, best_n_hidden_neuron),
        torch.nn.Tanh(),
        torch.nn.Linear(best_n_hidden_neuron, 1)
    ).to(device)

    net, final_loss, learning_curve = train_neural_net(
        model_fn,
        loss_fn,
        X=X_train_outer_tensor,
        y=Y_train_outer_tensor,
        n_replicates=n_replicates,
        max_iter=max_iter,
    )

    # Evaluate the model on the outer test set
    y_test_estimate = net(X_test_outer_tensor)
    mse_test = ((y_test_estimate.float() - Y_test_outer_tensor.float()) ** 2).mean().item()
    ANN_errors.append(mse_test)

    # Plot the learning curve for the current fold
    (h,) = summaries_axes[0].plot(learning_curve, color=color_list[outer_fold_idx])
    h.set_label("CV fold {0}".format(outer_fold_idx + 1))
    summaries_axes[0].set_xlabel("Iterations")
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel("Loss")
    summaries_axes[0].set_title("Learning Curves")

# ----------------------
# Plotting and Results Summary
# ----------------------

# Plot test MSE across folds as a bar chart
summaries_axes[1].bar(np.arange(1, K1 + 1), ANN_errors, color=color_list)
summaries_axes[1].set_xlabel("Fold")
summaries_axes[1].set_xticks(np.arange(1, K1 + 1))
summaries_axes[1].set_ylabel("MSE")
summaries_axes[1].set_title("Test Mean-Squared Error")

plt.tight_layout()
plt.show()

print("ANN errors:", ANN_errors)
print("Best number of hidden neurons per fold:", best_n_hidden_neurons)

# ----------------------
# Visualizing the Neural Network
# ----------------------

print("Diagram of best neural net in the last outer fold:")
# Retrieve weights and biases from the best network
# Here, the network architecture assumed is Linear -> Tanh -> Linear.
weights = [net[i].weight.data.cpu().numpy().T for i in [0, 2]]
biases = [net[i].bias.data.cpu().numpy() for i in [0, 2]]
tf = [str(net[i]) for i in [1, 2]]

# Draw the neural network diagram
# draw_neural_net(weights, biases, tf, attribute_names=attributeNames)
