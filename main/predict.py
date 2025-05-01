import numpy as np
import os
from keras.models import load_model

class AlphabetRecognizer:
    def __init__(self, model_path=None):
        if model_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(current_dir, '..', 'models', 'best_model.keras')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}") 
        self.model = load_model(model_path)
        self.labels = [chr(i) for i in range(65, 91)]  # A-Z
    def predict(self, image):
        if image.shape != (28, 28, 1):
            raise ValueError(f"Invalid image shape: {image.shape}. Expected (28,28,1)")
        image = image.astype('float32') / 255.0
        pred = self.model.predict(np.array([image]), verbose=0)[0]
        return self.labels[np.argmax(pred)], np.max(pred)

if __name__ == '__main__':
    recognizer = AlphabetRecognizer()
    test_image = np.random.rand(28, 28, 1)
    char, confidence = recognizer.predict(test_image)
    print(f"Predicted: {char} (Confidence: {confidence:.2%})")
