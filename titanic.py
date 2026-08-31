# ==========================================
# Titanic Classification Project
#
# Complete Pipeline:
# 1. Data Preprocessing
# 2. Random Forest Feature Selection
# 3. Top 3 / Top 6 / Top 9 Experiments
# 4. Training / Validation Curves
# 5. Select Best Feature Set
# 6. Final Model Training
# 7. Final Training Curves
# 8. Test Prediction
# 9. Generate submission.csv
# ==========================================


# ==========================================
# 1. Import Libraries
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==========================================
# 2. Set Random Seeds
# ==========================================

np.random.seed(42)
torch.manual_seed(42)


# ==========================================
# 3. Load Dataset
# ==========================================

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")


print("==========================================")
print("Dataset Information")
print("==========================================")

print("\nTrain shape:")
print(train_df.shape)

print("\nTest shape:")
print(test_df.shape)

print("\nFirst 5 rows:")
print(train_df.head())


# ==========================================
# 4. Separate Target
# ==========================================

y = train_df["Survived"]

X = train_df.drop(
    columns=["Survived"]
)


# ==========================================
# 5. Combine Train and Test
# ==========================================
#
# This makes sure that train and test have
# the same preprocessing / encoded columns.
#
# We DO NOT use test labels.
# ==========================================

X_all = pd.concat(
    [
        X,
        test_df
    ],
    axis=0,
    ignore_index=True
)


# ==========================================
# 6. Handle Missing Values
# ==========================================

# Age
age_median = X_all["Age"].median()

X_all["Age"] = X_all["Age"].fillna(
    age_median
)


# Embarked
embarked_mode = X_all["Embarked"].mode()[0]

X_all["Embarked"] = X_all["Embarked"].fillna(
    embarked_mode
)


# ==========================================
# 7. Encode Sex
# ==========================================

X_all["Sex"] = X_all["Sex"].map({
    "male": 0,
    "female": 1
})


# ==========================================
# 8. One-Hot Encode Embarked
# ==========================================

X_all = pd.get_dummies(
    X_all,
    columns=["Embarked"],
    dtype=int
)


# ==========================================
# 9. Remove Unused Columns
# ==========================================

X_all = X_all.drop(
    columns=[
        "PassengerId",
        "Name",
        "Ticket",
        "Cabin"
    ]
)


# ==========================================
# 10. Split Back Into Train and Test
# ==========================================

X_train_full = X_all.iloc[
    :len(train_df)
].copy()


X_test = X_all.iloc[
    len(train_df):
].copy()


print("\n==========================================")
print("Features After Preprocessing")
print("==========================================")

print(
    X_train_full.columns.tolist()
)


# ==========================================
# 11. Train / Validation Split
# ==========================================
#
# This split is ONLY used for experiments.
#
# It allows us to compare Top 3 / Top 6 /
# Top 9 features.
# ==========================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\n==========================================")
print("Train / Validation Split")
print("==========================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Validation samples:",
    len(X_val)
)


# ==========================================
# 12. Random Forest Feature Selection
# ==========================================

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)


rf.fit(
    X_train,
    y_train
)


importance = pd.Series(
    rf.feature_importances_,
    index=X_train.columns
)


importance = importance.sort_values(
    ascending=False
)


print("\n==========================================")
print("Random Forest Feature Importance")
print("==========================================")

print(importance)


# ==========================================
# 13. Define Neural Network
# ==========================================

class TitanicClassifier(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                16
            ),

            nn.ReLU(),

            nn.Linear(
                16,
                8
            ),

            nn.ReLU(),

            nn.Linear(
                8,
                1
            )
        )


    def forward(self, x):

        return self.network(x)


# ==========================================
# 14. Experiment Function
# ==========================================

def run_experiment(
    num_features,
    num_epochs=50,
    batch_size=32,
    learning_rate=0.001
):

    print("\n")
    print("==========================================")
    print(
        f"Experiment: Top {num_features} Features"
    )
    print("==========================================")


    # ======================================
    # Select Features
    # ======================================

    selected_features = (
        importance
        .head(num_features)
        .index
        .tolist()
    )


    print("\nSelected Features:")

    for feature in selected_features:

        print(
            f"- {feature}"
        )


    # ======================================
    # Select Data
    # ======================================

    X_train_selected = X_train[
        selected_features
    ]

    X_val_selected = X_val[
        selected_features
    ]


    # ======================================
    # Scaling
    # ======================================

    scaler = StandardScaler()


    # IMPORTANT:
    # Fit scaler only on training data

    X_train_scaled = scaler.fit_transform(
        X_train_selected
    )


    X_val_scaled = scaler.transform(
        X_val_selected
    )


    # ======================================
    # Convert to Tensor
    # ======================================

    X_train_tensor = torch.tensor(
        X_train_scaled,
        dtype=torch.float32
    )


    X_val_tensor = torch.tensor(
        X_val_scaled,
        dtype=torch.float32
    )


    y_train_tensor = torch.tensor(
        y_train.values,
        dtype=torch.float32
    )


    y_val_tensor = torch.tensor(
        y_val.values,
        dtype=torch.float32
    )


    # ======================================
    # Dataset
    # ======================================

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor
    )


    val_dataset = TensorDataset(
        X_val_tensor,
        y_val_tensor
    )


    # ======================================
    # DataLoader
    # ======================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    # ======================================
    # Create Model
    # ======================================

    model = TitanicClassifier(
        input_size=num_features
    )


    # ======================================
    # Loss Function
    # ======================================

    criterion = nn.BCEWithLogitsLoss()


    # ======================================
    # Optimizer
    # ======================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )


    # ======================================
    # History
    # ======================================

    train_losses = []
    val_losses = []

    train_accuracies = []
    val_accuracies = []


    # ======================================
    # Training
    # ======================================

    for epoch in range(num_epochs):


        # ----------------------------------
        # Training Mode
        # ----------------------------------

        model.train()

        total_train_loss = 0.0

        train_correct = 0
        train_total = 0


        for X_batch, y_batch in train_loader:

            # Clear gradients
            optimizer.zero_grad()


            # Forward pass
            outputs = model(
                X_batch
            ).squeeze(1)


            # Calculate loss
            loss = criterion(
                outputs,
                y_batch
            )


            # Backpropagation
            loss.backward()


            # Update parameters
            optimizer.step()


            # Loss
            total_train_loss += (
                loss.item()
            )


            # Prediction
            predictions = (
                torch.sigmoid(outputs) >= 0.5
            ).float()


            train_correct += (
                predictions == y_batch
            ).sum().item()


            train_total += (
                y_batch.size(0)
            )


        # Calculate training metrics

        train_loss = (
            total_train_loss /
            len(train_loader)
        )


        train_accuracy = (
            train_correct /
            train_total
        )


        # ----------------------------------
        # Validation Mode
        # ----------------------------------

        model.eval()

        total_val_loss = 0.0

        val_correct = 0
        val_total = 0


        with torch.no_grad():

            for X_batch, y_batch in val_loader:

                # Forward pass
                outputs = model(
                    X_batch
                ).squeeze(1)


                # Validation loss
                loss = criterion(
                    outputs,
                    y_batch
                )


                total_val_loss += (
                    loss.item()
                )


                # Prediction
                predictions = (
                    torch.sigmoid(outputs) >= 0.5
                ).float()


                val_correct += (
                    predictions == y_batch
                ).sum().item()


                val_total += (
                    y_batch.size(0)
                )


        # Calculate validation metrics

        val_loss = (
            total_val_loss /
            len(val_loader)
        )


        val_accuracy = (
            val_correct /
            val_total
        )


        # ----------------------------------
        # Save History
        # ----------------------------------

        train_losses.append(
            train_loss
        )


        val_losses.append(
            val_loss
        )


        train_accuracies.append(
            train_accuracy
        )


        val_accuracies.append(
            val_accuracy
        )


        # ----------------------------------
        # Print Progress
        # ----------------------------------

        if (
            epoch == 0
            or (epoch + 1) % 10 == 0
            or epoch == num_epochs - 1
        ):

            print(
                f"Epoch [{epoch + 1:02d}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.4f} "
                f"Val Loss: {val_loss:.4f} "
                f"Val Acc: {val_accuracy:.4f}"
            )


    # ======================================
    # Find Best Validation Accuracy
    # ======================================

    best_val_accuracy = max(
        val_accuracies
    )


    best_epoch = (
        val_accuracies.index(
            best_val_accuracy
        ) + 1
    )


    best_train_accuracy = (
        train_accuracies[
            best_epoch - 1
        ]
    )


    best_val_loss = (
        val_losses[
            best_epoch - 1
        ]
    )


    print("\nBest Result:")

    print(
        f"Best Epoch: {best_epoch}"
    )


    print(
        f"Best Train Accuracy: "
        f"{best_train_accuracy:.4f}"
    )


    print(
        f"Best Validation Accuracy: "
        f"{best_val_accuracy:.4f}"
    )


    print(
        f"Validation Loss at Best Epoch: "
        f"{best_val_loss:.4f}"
    )


    # ======================================
    # Plot Loss Curve
    # ======================================

    epochs = range(
        1,
        num_epochs + 1
    )


    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        epochs,
        train_losses,
        label="Training Loss"
    )


    plt.plot(
        epochs,
        val_losses,
        label="Validation Loss"
    )


    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.title(
        f"Top {num_features} Features - Loss"
    )

    plt.legend()

    plt.grid(True)


    plt.savefig(
        f"top{num_features}_loss.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    # ======================================
    # Plot Accuracy Curve
    # ======================================

    plt.figure(
        figsize=(8, 5)
    )


    plt.plot(
        epochs,
        train_accuracies,
        label="Training Accuracy"
    )


    plt.plot(
        epochs,
        val_accuracies,
        label="Validation Accuracy"
    )


    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        f"Top {num_features} Features - Accuracy"
    )

    plt.legend()

    plt.grid(True)


    plt.savefig(
        f"top{num_features}_accuracy.png",
        dpi=300,
        bbox_inches="tight"
    )


    plt.show()

    plt.close()


    # ======================================
    # Return Result
    # ======================================

    return {

        "num_features": num_features,

        "features": selected_features,

        "best_epoch": best_epoch,

        "train_accuracy": best_train_accuracy,

        "val_accuracy": best_val_accuracy,

        "val_loss": best_val_loss,

        "train_losses": train_losses,

        "val_losses": val_losses,

        "train_accuracies": train_accuracies,

        "val_accuracies": val_accuracies
    }


# ==========================================
# 15. Run Three Experiments
# ==========================================

results = []


# Top 3
result_top3 = run_experiment(
    num_features=3
)

results.append(
    result_top3
)


# Top 6
result_top6 = run_experiment(
    num_features=6
)

results.append(
    result_top6
)


# Top 9
result_top9 = run_experiment(
    num_features=9
)

results.append(
    result_top9
)


# ==========================================
# 16. Compare Experiments
# ==========================================

results_df = pd.DataFrame({

    "Features": [
        result["num_features"]
        for result in results
    ],

    "Best Epoch": [
        result["best_epoch"]
        for result in results
    ],

    "Training Accuracy": [
        result["train_accuracy"]
        for result in results
    ],

    "Validation Accuracy": [
        result["val_accuracy"]
        for result in results
    ],

    "Validation Loss": [
        result["val_loss"]
        for result in results
    ]
})


print("\n\n")
print("==========================================")
print("FINAL EXPERIMENT COMPARISON")
print("==========================================")


print(
    results_df.to_string(
        index=False
    )
)


# ==========================================
# 17. Find Best Feature Set
# ==========================================

best_result = max(
    results,
    key=lambda x: x["val_accuracy"]
)


print("\n")
print("==========================================")
print("BEST FEATURE SET")
print("==========================================")


print(
    f"Number of Features: "
    f"{best_result['num_features']}"
)


print("\nSelected Features:")


for feature in best_result["features"]:

    print(
        f"- {feature}"
    )


print(
    f"\nBest Epoch: "
    f"{best_result['best_epoch']}"
)


print(
    f"Best Validation Accuracy: "
    f"{best_result['val_accuracy']:.4f}"
)


print(
    f"Best Validation Loss: "
    f"{best_result['val_loss']:.4f}"
)


# ==========================================
# 18. Final Model Configuration
# ==========================================
#
# Based on our experiments:
#
# Top 6 was the best.
# Best Epoch was 32.
#
# We now use ALL 891 training samples.
# ==========================================

final_features = best_result[
    "features"
]


final_epochs = best_result[
    "best_epoch"
]


print("\n")
print("==========================================")
print("FINAL MODEL CONFIGURATION")
print("==========================================")


print(
    f"Number of Features: "
    f"{len(final_features)}"
)


print(
    f"Epochs: "
    f"{final_epochs}"
)


print("\nFinal Features:")


for feature in final_features:

    print(
        f"- {feature}"
    )


# ==========================================
# 19. Prepare Full Training Data
# ==========================================

X_final_train = X_train_full[
    final_features
].copy()


X_final_test = X_test[
    final_features
].copy()


# ==========================================
# 20. Scale Full Training and Test Data
# ==========================================

final_scaler = StandardScaler()


# Fit scaler ONLY on full training data

X_final_train_scaled = (
    final_scaler.fit_transform(
        X_final_train
    )
)


# Apply same scaler to test data

X_final_test_scaled = (
    final_scaler.transform(
        X_final_test
    )
)


# ==========================================
# 21. Convert to Tensor
# ==========================================

X_final_train_tensor = torch.tensor(
    X_final_train_scaled,
    dtype=torch.float32
)


X_final_test_tensor = torch.tensor(
    X_final_test_scaled,
    dtype=torch.float32
)


y_final_train_tensor = torch.tensor(
    y.values,
    dtype=torch.float32
)


print("\n")
print("==========================================")
print("Final Tensor Shapes")
print("==========================================")


print(
    "X_train:",
    X_final_train_tensor.shape
)


print(
    "y_train:",
    y_final_train_tensor.shape
)


print(
    "X_test:",
    X_final_test_tensor.shape
)


# ==========================================
# 22. Final Dataset
# ==========================================

final_dataset = TensorDataset(
    X_final_train_tensor,
    y_final_train_tensor
)


# ==========================================
# 23. Final DataLoader
# ==========================================

final_batch_size = 32


final_loader = DataLoader(
    final_dataset,
    batch_size=final_batch_size,
    shuffle=True
)


# ==========================================
# 24. Create Final Model
# ==========================================

final_model = TitanicClassifier(
    input_size=len(final_features)
)


print("\n")
print("==========================================")
print("FINAL MODEL")
print("==========================================")


print(final_model)


# ==========================================
# 25. Loss Function
# ==========================================

final_criterion = nn.BCEWithLogitsLoss()


# ==========================================
# 26. Optimizer
# ==========================================

final_optimizer = torch.optim.Adam(
    final_model.parameters(),
    lr=0.001
)


# ==========================================
# 27. Final Training
# ==========================================

final_train_losses = []
final_train_accuracies = []


print("\n")
print("==========================================")
print("FINAL MODEL TRAINING")
print("==========================================")


for epoch in range(final_epochs):

    final_model.train()


    total_loss = 0.0

    correct = 0
    total = 0


    for X_batch, y_batch in final_loader:

        # Clear gradients
        final_optimizer.zero_grad()


        # Forward pass
        outputs = final_model(
            X_batch
        ).squeeze(1)


        # Loss
        loss = final_criterion(
            outputs,
            y_batch
        )


        # Backpropagation
        loss.backward()


        # Update weights
        final_optimizer.step()


        # Record loss
        total_loss += (
            loss.item()
        )


        # Prediction
        predictions = (
            torch.sigmoid(outputs) >= 0.5
        ).float()


        correct += (
            predictions == y_batch
        ).sum().item()


        total += y_batch.size(0)


    # Calculate epoch metrics

    epoch_loss = (
        total_loss /
        len(final_loader)
    )


    epoch_accuracy = (
        correct /
        total
    )


    # Save history

    final_train_losses.append(
        epoch_loss
    )


    final_train_accuracies.append(
        epoch_accuracy
    )


    # Print progress

    if (
        epoch == 0
        or (epoch + 1) % 5 == 0
        or epoch == final_epochs - 1
    ):

        print(
            f"Epoch [{epoch + 1:02d}/{final_epochs}] "
            f"Loss: {epoch_loss:.4f} "
            f"Accuracy: {epoch_accuracy:.4f}"
        )


# ==========================================
# 28. Final Training Result
# ==========================================

print("\n")
print("==========================================")
print("FINAL TRAINING RESULT")
print("==========================================")


print(
    f"Final Training Loss: "
    f"{final_train_losses[-1]:.4f}"
)


print(
    f"Final Training Accuracy: "
    f"{final_train_accuracies[-1]:.4f}"
)


# ==========================================
# 29. Plot Final Training Loss
# ==========================================

final_epochs_range = range(
    1,
    final_epochs + 1
)


plt.figure(
    figsize=(8, 5)
)


plt.plot(
    final_epochs_range,
    final_train_losses,
    label="Training Loss"
)


plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title(
    "Final Model Training Loss"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "final_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


# ==========================================
# 30. Plot Final Training Accuracy
# ==========================================

plt.figure(
    figsize=(8, 5)
)


plt.plot(
    final_epochs_range,
    final_train_accuracies,
    label="Training Accuracy"
)


plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.title(
    "Final Model Training Accuracy"
)

plt.legend()

plt.grid(True)


plt.savefig(
    "final_training_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)


plt.show()

plt.close()


# ==========================================
# 31. Predict Test Dataset
# ==========================================

final_model.eval()


with torch.no_grad():

    test_outputs = final_model(
        X_final_test_tensor
    ).squeeze(1)


    # Convert logits to probability

    test_probabilities = torch.sigmoid(
        test_outputs
    )


    # Convert probability to 0 / 1

    test_predictions = (
        test_probabilities >= 0.5
    ).int()


# Convert to NumPy

predictions = (
    test_predictions.numpy()
)


# ==========================================
# 32. Check Predictions
# ==========================================

print("\n")
print("==========================================")
print("TEST PREDICTIONS")
print("==========================================")


print(
    "Number of predictions:",
    len(predictions)
)


print("\nFirst 20 predictions:")


print(
    predictions[:20]
)


# ==========================================
# 33. Create Submission
# ==========================================

submission = pd.DataFrame({

    "PassengerId":
        test_df["PassengerId"],

    "Survived":
        predictions

})


# ==========================================
# 34. Save submission.csv
# ==========================================

submission.to_csv(
    "submission.csv",
    index=False
)


# ==========================================
# 35. Check Submission
# ==========================================

print("\n")
print("==========================================")
print("SUBMISSION")
print("==========================================")


print(
    submission.head(10)
)


print("\nSubmission shape:")


print(
    submission.shape
)


print("\nPrediction distribution:")


print(
    submission["Survived"].value_counts()
)


print("\n==========================================")
print("PROJECT COMPLETED")
print("==========================================")


print(
    "\nSubmission saved successfully!"
)


print(
    "File: submission.csv"
)