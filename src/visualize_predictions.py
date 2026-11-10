
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data_loader import load_datasets

MODEL_PATH = "models/best_model.keras"
OUTPUT_PATH = Path("reports/predictions_visualization.png")

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


def visualize_predictions():

    # Load test dataset and model
    _, _, test_ds = load_datasets()
    model = tf.keras.models.load_model(MODEL_PATH)

    # Select 4 NORMAL and 5 PNEUMONIA images
    selected_images = []
    selected_labels = []
    class_counts = {0: 0, 1: 0}
    required_counts = {0: 4, 1: 5}

    for images, labels in test_ds:
        for image, label in zip(images, labels):
            class_id = int(label.numpy().flatten()[0])

            if class_counts[class_id] < required_counts[class_id]:
                selected_images.append(image.numpy())
                selected_labels.append(class_id)
                class_counts[class_id] += 1

        if class_counts == required_counts:
            break

    # Convert to arrays
    images_array = np.array(selected_images)
    actual_labels = np.array(selected_labels)

    # Generate predictions
    probabilities = model.predict(images_array, verbose=0).flatten()
    predictions = (probabilities >= 0.5).astype(int)

    # Create visualization
    plt.figure(figsize=(15, 12))

    for i in range(len(images_array)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images_array[i])

        actual = CLASS_NAMES[actual_labels[i]]
        predicted = CLASS_NAMES[predictions[i]]

        confidence = (
            probabilities[i]
            if predictions[i] == 1
            else 1 - probabilities[i]
        )

        status = (
            "Correct"
            if actual_labels[i] == predictions[i]
            else "Incorrect"
        )

        plt.title(
            f"Actual: {actual}\n"
            f"Predicted: {predicted}\n"
            f"Confidence: {confidence:.2%}\n"
            f"{status}"
        )

        plt.axis("off")

    plt.suptitle(
        "Pneumonia Detection - Model Predictions",
        fontsize=16
    )

    plt.tight_layout()

    # Save visualization
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.show()

    print("\nVisualization saved successfully!")
    print("File location:", OUTPUT_PATH)


if __name__ == "__main__":
    visualize_predictions()
