import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_label_distribution(df):
    """
    Plots a pie chart showing the distribution of labels.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with a 'label' column.
    """
    label_counts = df['label'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=['Reliable', 'Fake'], autopct='%1.1f%%', colors=['green', 'red'])
    plt.title('Distribution of Labels', fontsize=18, weight='bold')
    plt.show();


def plot_label_count(df):
    """
    Plots a bar chart showing the count of reliable vs fake news.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with a 'label' column.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x='label', hue='label', data=df, palette='coolwarm', dodge=False, legend=False)
    plt.title('Distribution of Reliable vs Fake News', fontsize=16, weight='bold')
    plt.xlabel('Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(ticks=[0, 1], labels=['Reliable', 'Fake'], fontsize=12)
    plt.yticks(fontsize=12)
    plt.show();


def plot_article_lengths(df, max_length=5000):
    """
    Plots a histogram showing the distribution of article lengths.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'text' and 'label' columns.
    max_length (int): Maximum length of articles to include in the plot.
    """
    temp_df = pd.DataFrame()
    temp_df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
    temp_df['label'] = df['label']
    filtered_df = temp_df[temp_df['text_length'] <= max_length]

    plt.figure(figsize=(10, 6))
    sns.histplot(data=filtered_df, x='text_length', bins=25, kde=False, hue='label', multiple='stack', palette='coolwarm')
    plt.title('Distribution of Article Lengths', fontsize=16, weight='bold')
    plt.xlabel('Number of Words', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.legend(title='Label', labels=['Fake','Reliable'])
    plt.show();


def plot_top_authors(df, top_n=20):
    """
    Plots a stacked bar chart showing the number of articles by top authors.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'author' and 'label' columns.
    top_n (int): Number of top authors to include in the plot.
    """
    author_label_counts = df.groupby(['author', 'label']).size().unstack(fill_value=0)
    top_authors = df['author'].value_counts().head(top_n).index
    filtered_author_label_counts = author_label_counts.loc[top_authors]

    plt.figure(figsize=(14, 10))
    filtered_author_label_counts.plot(kind='barh', stacked=True, color=['green', 'red'], alpha=0.7)
    plt.title('Top 20 Authors', fontsize=16, weight='bold')
    plt.xlabel('Number of Articles', fontsize=14)
    plt.ylabel('Authors', fontsize=14)
    plt.legend(title='Label', labels=['Reliable', 'Fake'])
    plt.show();


def plot_word_cloud(text, title):
    """
    Generate and display a word cloud from a given text.
    
    Parameters:
    text (str): Text to generate the word cloud from.
    title (str): Title of the word cloud plot.
    """
    wordcloud = WordCloud(width=1200, height=800, background_color='white').generate(text)
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=16, weight='bold')
    plt.axis('off')
    plt.show();


def plot_title_lengths(df, max_length=30):
    """
    Plot a histogram showing the distribution of article title lengths by reliability.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data with 'title' and 'label' columns.
    max_length (int): Maximum length of titles to include in the plot.
    """
    temp_df = pd.DataFrame()
    temp_df['title_length'] = df['title'].apply(lambda x: len(str(x).split()))
    temp_df['label'] = df['label']
    filtered_df = temp_df[temp_df['title_length'] <= max_length]

    plt.figure(figsize=(10, 6))
    sns.histplot(data=filtered_df, x='title_length', bins=12, kde=False, hue='label', multiple='stack', palette='coolwarm')
    plt.title('Distribution of Title Lengths by Reliability', fontsize=16, weight='bold')
    plt.xlabel('Number of Words', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    ax = plt.gca()
    ax.grid(False)
    plt.legend(title='Label', labels=['Fake','Reliable'])
    plt.show();
