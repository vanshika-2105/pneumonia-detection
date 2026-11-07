
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from data_loader import load_datasets


# 1. Load the test dataset
_, _, test_ds = load_datasets()

# 2. Load the trained model
model = tf.keras.models.load_model("models/best_model.keras")

print("\nModel loaded successfully!")


# 3. Get actual labels
y_true = np.concatenate([
    labels.numpy().flatten()
    for images, labels in test_ds
])

# 4. Predict probabilities
y_prob = model.predict(test_ds)

# 5. Convert probabilities into class predictions
y_pred = (y_prob.flatten() >= 0.5).astype(int)


# 6. Calculate evaluation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)


# 7. Display results
print("\n----- MODEL EVALUATION RESULTS -----")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


# 8. Display confusion matrix
cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 9. Display classification report
print("\nClassification Report:")

print(classification_report(
    y_true,
    y_pred,
    target_names=["NORMAL", "PNEUMONIA"]
))
