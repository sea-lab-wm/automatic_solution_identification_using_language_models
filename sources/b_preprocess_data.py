import json
import os
import re
import string
import contractions
import nltk
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from stopwordsiso import stopwords

from utils import *

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer


def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


def remove_special_characters(text):
    # Define the pattern to match special characters
    pattern = r'[^a-zA-Z0-9\s]'  # This pattern matches anything that is not a letter, digit, or whitespace

    # Use the sub() function to replace all matches of the pattern with an empty string
    clean_text = re.sub(pattern, ' ', text)

    return clean_text


def consolidate_tags(input_string):
    # Define a regular expression pattern to match consecutive occurrences of the same tag
    pattern = re.compile(r'\b(\w+)\b(?: +\1)+')

    # Use the pattern to find all consecutive tag occurrences
    matches = pattern.findall(input_string)

    # Replace consecutive tag occurrences with a single tag
    output_string = re.sub(pattern, r'\1', input_string)

    return output_string


def preprocess_text(text, remove_urls=True, lowercase=True, remove_numbers=True,
                    remove_html=True, remove_stopwords=True, stemming=True,
                    remove_punctuations=True, remove_single_characters=True, extra_whitespace=True,
                    lemmatization=True, contraction=True):
    try:
        # Lowercase
        if lowercase:
            text = text.lower()

        replaced_word = ['URL', 'NUMBER', 'HTML']

        # Remove URLs
        if remove_urls:
            text = re.sub(r'https?\S+|www\S+', ' URL ', text)

        # Removing numbers
        if remove_numbers:
            text = re.sub(r'\d+', ' NUMBER ', text)

        # Remove HTML tags and accented characters
        if remove_html:
            soup = BeautifulSoup(text, 'html.parser')
            replacement_text = " HTML "
            html_tag = soup.find('html')

            if html_tag:
                html_tag.replace_with(replacement_text)

            # Print the updated mixed content
            text = str(soup)

        if contraction:  # expand contractions
            text = expand_contractions(text)

        if extra_whitespace:  # remove extra whitespaces
            text = remove_whitespace(text)

        # Tokenization
        tokens = word_tokenize(text)

        # Remove articles, prepositions, or adjectives
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]

        # Lemmatization
        if lemmatization:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

        # Stemming
        if stemming:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(token) if token not in replaced_word else token for token in tokens]

        # Remove Punctuations
        if remove_punctuations:
            tokens = [token.translate(str.maketrans('', '', string.punctuation)) for token in tokens]

        # Remove single characters and whitespaces
        if remove_single_characters:
            tokens = [token for token in tokens if len(token) > 1]

        # Remove code, stack traces, output logs, environment information
        # ...

        # Join the tokens back into a single string
        preprocessed_text = ' '.join(tokens)

        preprocessed_text = remove_whitespace(consolidate_tags(preprocessed_text))

        return preprocessed_text
    except Exception as e:
        print(f"An error occurred during text preprocessing. Error: {str(e)}")
        print(f"Problematic text: {text}")
        return None


def text_preprocess(text, mode):
    if mode == 'none':
        return text
    elif mode == 'essential':
        return preprocess_text(text, remove_urls=True, lowercase=True, remove_numbers=False,
                               remove_html=False, remove_stopwords=False, stemming=False,
                               remove_punctuations=True, remove_single_characters=False, extra_whitespace=True,
                               lemmatization=True, contraction=True)
    elif mode == 'all':
        return preprocess_text(text, remove_urls=True, lowercase=True, remove_numbers=True,
                               remove_html=True, remove_stopwords=True, stemming=True,
                               remove_punctuations=True, remove_single_characters=True, extra_whitespace=True,
                               lemmatization=True, contraction=True)
    else:
        return text


def check_null(data):
    null_rows = data[data.isnull().any(axis=1)]
    data = data.dropna() 

    # Display the result
    print(f'Length of Null Value rows {len(null_rows)}')
    if len(null_rows) != 0:
        print(f'After Dropping Null Values {len(data)}')
    return data


def preprocess_data(data_df):
    print(data_df.shape)

    data_df = data_df.copy()

    exp_config = get_experiment_config()

    for mode in exp_config['preprocess']:
        data_df[f'text_{mode}'] = data_df['text'].apply(text_preprocess, mode=mode)

    return data_df


def add_tfidf_embedding(data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['text'])
    data['tfidf_embedding'] = tfidf_matrix.toarray().tolist()
    return data


def preprocess_comment():
    comment_data = pd.read_csv(embedding_comment_data)
    comment_data = check_null(comment_data)
    pre_data = preprocess_data(comment_data)
    pre_data.to_csv(preprocess_comment_data, index=False)
    print(f"Preprocessed File saved to {preprocess_comment_data}")
