
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import load_datasets
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

MODEL_PATH = "models/best_model.keras"
OUTPUT_PATH = Path("reports/confusion_matrix.png")
REPORT_PATH = Path("reports/performance_analysis.md")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def analyze_performance():
    # Load test dataset and trained model
    _, _, test_ds = load_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    # Get actual labels
    y_true = np.concatenate([
        labels.numpy().flatten() for _, labels in test_ds
    ]).astype(int)

    # Get model predictions
    probabilities = model.predict(test_ds, verbose=1).flatten()
    y_pred = (probabilities >= 0.5).astype(int)

    # Calculate performance metrics
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )

    tn, fp, fn, tp = cm.ravel()

    # Display results
    print("\n========== PERFORMANCE ANALYSIS ==========")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")

    print("\nCLASSIFICATION REPORT:")
    print(report)

    # Plot confusion matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Pneumonia Detection - Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=14
            )

    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    # Save analysis report
    report_text = f"""# Day 9 - Model Performance Analysis

## Overall Accuracy
{accuracy:.2%}

## Confusion Matrix

| Metric | Count |
|---|---:|
| True Negatives (TN) | {tn} |
| False Positives (FP) | {fp} |
| False Negatives (FN) | {fn} |
| True Positives (TP) | {tp} |

## Classification Report

```text
{report}
```

## Error Analysis

- False Positives: {fp} normal images were incorrectly predicted as pneumonia.
- False Negatives: {fn} pneumonia images were incorrectly predicted as normal.
- True Negatives: {tn} normal images were correctly identified.
- True Positives: {tp} pneumonia images were correctly identified.

## Conclusion

The model identifies pneumonia cases with high recall, but it also produces
false positive predictions for some normal images. This means further
validation would be necessary before any real-world medical use.

**Note:** These results describe performance on this test dataset and do
not establish clinical reliability.
"""

    REPORT_PATH.write_text(report_text, encoding="utf-8")

    print("\nConfusion matrix saved to:", OUTPUT_PATH)
    print("Analysis report saved to:", REPORT_PATH)


if __name__ == "__main__":
    analyze_performance()
