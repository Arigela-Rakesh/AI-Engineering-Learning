# ============================================================
# STUDENT PASS / FAIL PREDICTION USING PYTORCH
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 2. CREATE DATASET
# ============================================================

# Features:
# Study Hours
# Attendance
# Previous Marks

X = torch.tensor([
    [5, 80, 70],
    [8, 90, 85],
    [2, 60, 45],
    [3, 70, 50],
    [7, 85, 80],
    [1, 55, 40],
    [6, 88, 75],
    [4, 75, 65],
    [9, 95, 90],
    [2, 65, 48]
], dtype=torch.float32)


# Target:
# 0 = Fail
# 1 = Pass

y = torch.tensor([
    [1],
    [1],
    [0],
    [0],
    [1],
    [0],
    [1],
    [0],
    [1],
    [0]
], dtype=torch.float32)


# ============================================================
# 3. CUSTOM DATASET
# ============================================================

class StudentDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


# Create complete dataset

dataset = StudentDataset(X, y)

print("Total samples:", len(dataset))


# ============================================================
# 4. SPLIT DATA
# ============================================================

# Training data
X_train = X[:6]
y_train = y[:6]

# Validation data
X_val = X[6:8]
y_val = y[6:8]

# Test data
X_test = X[8:]
y_test = y[8:]


print("\nData Split:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# ============================================================
# 5. CREATE THREE DATASETS
# ============================================================

train_dataset = StudentDataset(
    X_train,
    y_train
)

val_dataset = StudentDataset(
    X_val,
    y_val
)

test_dataset = StudentDataset(
    X_test,
    y_test
)


# ============================================================
# 6. CREATE DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=2,
    shuffle=False
)


# ============================================================
# 7. CREATE NEURAL NETWORK
# ============================================================

class StudentModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Input: 3 features
        # Output: 8 neurons
        self.layer1 = nn.Linear(3, 8)

        self.relu = nn.ReLU()

        # 8 neurons -> 4 neurons
        self.layer2 = nn.Linear(8, 4)

        # 4 neurons -> 1 output
        self.output = nn.Linear(4, 1)

    def forward(self, x):

        x = self.layer1(x)

        x = self.relu(x)

        x = self.layer2(x)

        x = self.relu(x)

        x = self.output(x)

        return x


# ============================================================
# 8. SELECT DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:", device)


# Create model

model = StudentModel().to(device)


# ============================================================
# 9. LOSS FUNCTION
# ============================================================

# Binary classification:
# 0 = Fail
# 1 = Pass

loss_function = nn.BCEWithLogitsLoss()


# ============================================================
# 10. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 11. TRAINING
# ============================================================

epochs = 20

print("\nStarting Training...\n")

for epoch in range(epochs):

    # Training mode
    model.train()

    total_train_loss = 0

    for X_batch, y_batch in train_loader:

        # Move data to device
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # ----------------------------------------------------
        # Forward Pass
        # ----------------------------------------------------

        predictions = model(X_batch)

        # ----------------------------------------------------
        # Calculate Loss
        # ----------------------------------------------------

        loss = loss_function(
            predictions,
            y_batch
        )

        # ----------------------------------------------------
        # Clear Old Gradients
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # Update Parameters
        # ----------------------------------------------------

        optimizer.step()

        total_train_loss += loss.item()

    # Average training loss

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # ========================================================
    # 12. VALIDATION
    # ========================================================

    model.eval()

    total_val_loss = 0

    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass

            predictions = model(X_batch)

            # Validation loss

            val_loss = loss_function(
                predictions,
                y_batch
            )

            total_val_loss += val_loss.item()


    average_val_loss = (
        total_val_loss / len(val_loader)
    )


    # Print results

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {average_train_loss:.4f} "
        f"Val Loss: {average_val_loss:.4f}"
    )


# ============================================================
# 13. TEST THE MODEL
# ============================================================

model.eval()

correct = 0
total = 0

print("\nTesting Model...\n")

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # Model output = logits

        logits = model(X_batch)

        # Convert logits to probabilities

        probabilities = torch.sigmoid(logits)

        # Convert probabilities to classes

        predictions = (
            probabilities >= 0.5
        ).float()

        # Count correct predictions

        correct += (
            predictions == y_batch
        ).sum().item()

        total += y_batch.size(0)


# Calculate accuracy

accuracy = correct / total

print("Test Accuracy:", accuracy)


# ============================================================
# 14. SAVE THE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "student_model.pth"
)

print("\nModel saved as: student_model.pth")


# ============================================================
# 15. LOAD THE MODEL
# ============================================================

loaded_model = StudentModel()

loaded_model.load_state_dict(
    torch.load(
        "student_model.pth",
        map_location=device
    )
)

loaded_model.to(device)

loaded_model.eval()

print("Model loaded successfully!")


# ============================================================
# 16. MAKE A NEW PREDICTION
# ============================================================

# New student:
#
# Study Hours = 7
# Attendance = 88
# Previous Marks = 80

new_student = torch.tensor(
    [[7, 88, 80]],
    dtype=torch.float32
)


# Move to same device

new_student = new_student.to(device)


# ============================================================
# PREDICTION
# ============================================================

loaded_model.eval()

with torch.no_grad():

    # Get raw model output
    logits = loaded_model(new_student)

    # Convert logits -> probability
    probability = torch.sigmoid(logits)

    # Probability -> class
    prediction = (
        probability >= 0.5
    ).float()


# Convert tensors to Python values

probability_value = probability.item()

prediction_value = int(prediction.item())


# ============================================================
# 17. DISPLAY RESULT
# ============================================================

print("\n--------------------------------")
print("NEW STUDENT PREDICTION")
print("--------------------------------")

print("Study Hours:", 7)
print("Attendance:", 88)
print("Previous Marks:", 80)

print(
    f"Pass Probability: "
    f"{probability_value:.4f}"
)

print(
    f"Pass Probability: "
    f"{probability_value * 100:.2f}%"
)


if prediction_value == 1:

    print("Prediction: PASS")

else:

    print("Prediction: FAIL")