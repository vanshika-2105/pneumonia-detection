from pathlib import Path
import tensorflow as tf

DATA_DIR = Path("data/raw/chest_xray")
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32


def normalize_images(images, labels):
    images = tf.cast(images, tf.float32) / 255.0
    return images, labels


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        seed=42,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False,
    )

    train_ds = train_ds.map(normalize_images)
    val_ds = val_ds.map(normalize_images)
    test_ds = test_ds.map(normalize_images)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    train_ds, val_ds, test_ds = load_datasets()

    print("Class names:", ["NORMAL", "PNEUMONIA"])

    for images, labels in train_ds.take(1):
        print("Image batch shape:", images.shape)
        print("Label batch shape:", labels.shape)
        print("Pixel range:", float(tf.reduce_min(images)),
              "to", float(tf.reduce_max(images)))