
from pathlib import Path
import tensorflow as tf

from data_loader import load_datasets
from model import build_model

# Training settings
EPOCHS = 15
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train():
    # 1. Load training and validation datasets
    train_ds, val_ds, _ = load_datasets()

    # 2. Build a fresh CNN model
    model = build_model()

    # 3. Define class weights
    class_weights = {
        0: 1.9448,  # NORMAL
        1: 0.6730   # PNEUMONIA
    }

    # 4. Configure callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_improved_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    # 5. Train with class weights
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # 6. Save final improved model
    model.save(MODEL_DIR / "final_improved_model.keras")

    print("\nImproved training complete!")
    print(
        "Best validation accuracy:",
        max(history.history["val_accuracy"])
    )


if __name__ == "__main__":
    train()
