import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem.porter import PorterStemmer
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Load the tokenizer and model
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model('fake_news_model.h5')

app = Flask(__name__)

def preprocess_text(text):
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        title = request.form['title']
        author = request.form['author']
        article_text = request.form['text']
        
        # Combine the inputs
        combined_text = title + ' ' + author + ' ' + article_text
        preprocessed_text = preprocess_text(combined_text)
        
        # Tokenize and pad the sequence
        seq = tokenizer.texts_to_sequences([preprocessed_text])
        padded_seq = pad_sequences(seq, maxlen=50, padding='pre')

        # Predict the label
        pred = model.predict(padded_seq)
        prediction = 'Fake' if pred[0][0] > 0.5 else 'Reliable'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
