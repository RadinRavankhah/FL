import keras
from keras import layers

model_a = keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

print("Model A weights:")
print(model_a.get_weights()[0].shape)
print(model_a.get_weights()[0][0][0])


model_a.save_weights('test/saving_model_weights_to_file/model.weights.h5')

model_b = keras.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])

model_b.load_weights('test/saving_model_weights_to_file/model.weights.h5')

print("Model B weights:")
print(model_b.get_weights()[0].shape)
print(model_b.get_weights()[0][0][0])