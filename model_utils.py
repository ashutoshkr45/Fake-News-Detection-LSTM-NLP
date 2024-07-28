import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense

nltk.download('stopwords')


def load_and_clean_data(file_path):
    """
    Load and clean the dataset, remove NaN values, and combine title and author into a single column.
    
    Parameters:
    file_path (str): Path to the CSV file containing the dataset.
    
    Returns:
    DataFrame: Cleaned DataFrame with combined 'news' column.
    """
    df = pd.read_csv(file_path)
    df = df.dropna()
    X = df.drop('label', axis=1)
    y = df['label']
    X.reset_index(inplace=True, drop=True)
    X['news'] = X['title'] + ' ' + X['author']
    X = X.drop(columns=['title', 'author', 'text'], axis=1)
    return X, y


def preprocess_text(corpus):
    """
    Preprocess the text data by removing special characters, converting to lowercase, removing stopwords, and stemming.
    
    Parameters:
    corpus (list): List of text data.
    
    Returns:
    list: Preprocessed text data.
    """
    ps = PorterStemmer()
    processed_corpus = []
    for text in corpus:
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
        text = ' '.join(text)
        processed_corpus.append(text)
    return processed_corpus


def create_model(voc_size, vector_features, input_length):
    """
    Create and compile the neural network model.
    
    Parameters:
    voc_size (int): Vocabulary size.
    vector_features (int): Number of vector features for the embedding layer.
    input_length (int): Input length for the embedding layer.
    
    Returns:
    Sequential: Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Embedding(voc_size, vector_features, input_length=input_length))
    model.add(Dropout(0.3))
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def get_callbacks(patience=5, min_delta=0.0001):
    """
    Get the list of callbacks for model training.
    
    Parameters:
    patience (int): Number of epochs with no improvement after which training will be stopped.
    min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
    
    Returns:
    list: List of Keras callbacks.
    """
    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=min_delta,
        patience=patience,
        verbose=1,
        mode='auto',
        restore_best_weights=True
    )
    return [callback]


# Function to plot Confusion matrix Heatmap
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='PiYG', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels',fontsize=16)
    plt.ylabel('True Labels',fontsize=16)
    plt.title('Confusion Matrix',fontsize=20,color='blue')
    plt.show();