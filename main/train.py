import numpy as np
import matplotlib.pyplot as plt
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model import create_model

def save_training_plots(history):
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(12,6))
    # Accuracy plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Accuracy Curves')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    # Loss plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Loss Curves')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('outputs/plots/training_history.png')
    plt.close()
def train_model():
    train_data = np.load('data/processed/train_data.npz')
    test_data = np.load('data/processed/test_data.npz')
    x_train, y_train = train_data['x_train'], train_data['y_train']
    x_test, y_test = test_data['x_test'], test_data['y_test']
    model = create_model()
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ModelCheckpoint('models/best_model.keras', save_best_only=True)
    ]
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=256,
        validation_data=(x_test, y_test),
        callbacks=callbacks
    )
    model.save('models/final_model.keras')
    save_training_plots(history)
    return history

if __name__ == '__main__':
    train_model()
