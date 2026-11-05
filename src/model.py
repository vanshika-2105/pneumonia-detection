import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape=(150, 150, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


if __name__ == "__main__":
    model = build_model()
    model.summary()

    dummy_input = tf.random.normal((1, 150, 150, 3))
    prediction = model(dummy_input)

    print("Output shape:", prediction.shape)
