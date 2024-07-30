# Fake News Detector

## DESCRIPTION
This project aims to detect fake news using a Deep Learning model. The primary focus is on political news, particularly during the 2016 USA elections. The model processes the title, author, and text of articles to classify them as Reliable or Fake.

## DATASET
The dataset is sourced from a Kaggle competition and can be accessed [here](https://www.kaggle.com/competitions/fake-news/data?select=train.csv). It includes:
- **train.csv**: Contains articles with the following attributes:
  - `id`: Unique ID for a news article
  - `title`: Title of the news article
  - `author`: Author of the news article
  - `text`: Text of the article (could be incomplete)
  - `label`: Label indicating if the article is potentially unreliable (`1` for Fake, `0` for Reliable)
- **test.csv**: Similar to `train.csv` but without the `label` attribute.

## OVERVIEW
The project workflow includes:
1. **Data Visualization and Exploration** (`viz_utils.py` and `textual_eda.ipynb`)
2. **Model Building and Training** (`model_utils.py` and `fake_news_detection_main.ipynb`)
3. **Model Deployment** (`app.py` for Flask app)

## MOTIVATION
With the rise of misinformation, especially in the political domain, there is a need for automated systems to help identify fake news. This project contributes to this effort by developing a robust model capable of detecting unreliable news articles.

## TECHNICAL ASPECTS
### Data Visualization and Exploration
- **`textual_eda.ipynb`**: Performs Exploratory Data Analysis, visualizing label distribution, article lengths, author statistics and wordclouds.

### Model Building and Training
- **`fake_news_detection_main.ipynb`**: Involves text preprocessing, tokenization, sequence padding, and model creation. Utilizes a vocabulary size of 5000, sequence length of 30, and embedding vector dimension of 50.

### Saving and Loading Models
- The model and tokenizer are saved for deployment purposes, ensuring the web app uses the same configuration as during training.

### Model Deployment
- **Flask App**: The web app is built using Flask, HTML, and CSS. It processes user input, predicts the reliability of news articles, and displays the result.
- **Web App Link**: [Fake News Detector Web App (RENDER)](https://fake-news-detection-ashutosh.onrender.com/)

## RESULTS
The model achieved an accuracy of 99% on the splitted test set, demonstrating its effectiveness in detecting fake news.

## CONCLUSION
The project successfully demonstrates a high-accuracy Fake News Detection model. Both Jupyter notebooks are well-documented, providing clear insights into the data processing and model development stages.

## REQUIREMENTS
The required dependencies are listed in `requirements.txt` and can be installed via:

pip install -r requirements.txt
