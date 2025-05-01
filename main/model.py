from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, Flatten, Dense, Dropout
from keras.optimizers import Adam

def create_model(input_shape=(28, 28, 1)):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        BatchNormalization(),
        MaxPool2D((2,2)),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(26, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model
