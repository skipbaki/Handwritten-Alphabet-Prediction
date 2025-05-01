import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import os

def load_data(data_path='data/raw/A_Z Handwritten Data.csv'):
    df = pd.read_csv(data_path).astype('float32')
    x = df.drop('0', axis=1)
    y = df['0'].astype(int)
    return x, y
def plot_label_distribution(y):
    os.makedirs('outputs/plots', exist_ok=True)
    plt.figure(figsize=(12,6))
    pd.Series(y).value_counts().sort_index().plot(kind='bar')
    plt.title("Class Distribution")
    plt.xlabel("Class (0=A, 25=Z)")
    plt.ylabel("Count")
    plt.savefig('outputs/plots/class_distribution.png')
    plt.close()
def preprocess_data(x, y):
    x = np.reshape(x.values, (x.shape[0], 28, 28))
    x = x / 255.0
    x = x.reshape(-1, 28, 28, 1)
    y = to_categorical(y, num_classes=26)
    return train_test_split(x, y, test_size=0.2, stratify=y)

if __name__ == '__main__':
    x, y = load_data()
    plot_label_distribution(y)
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    np.savez('data/processed/train_data.npz', x_train=x_train, y_train=y_train)
    np.savez('data/processed/test_data.npz', x_test=x_test, y_test=y_test)
