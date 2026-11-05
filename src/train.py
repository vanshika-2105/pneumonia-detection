from pathlib import Path
import tensorflow as tf

from data_loader import load_datasets
from model import build_model

EPOCHS = 10
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train():
    train_ds, val_ds, _ = load_datasets()
    model = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(MODEL_DIR / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_DIR / "final_model.keras")

    print("\nTraining complete.")
    print("Best validation accuracy:", max(history.history["val_accuracy"]))


if __name__ == "__main__":
    train()
