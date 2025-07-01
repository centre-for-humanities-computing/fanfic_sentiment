import re
import logging
from loguru import logger
from typing import List


import nltk
nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

import rpy2.robjects.packages as rpackages

# import saffine.multi_detrending as md
# from math import log
# import numpy as np


# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_whitespace(text: str) -> str:
    # check if text is a string
    if not isinstance(text, str):
        logger.warning(f"Expected string, got {type(text)}. Returning original text.")
        return text
    # rm newline characters
    text = text.replace('\n', ' ')
    # multiple spaces -> single space
    text = re.sub(r'\s+', ' ', text)
    # rm spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # rm excess spaces after punctuation (.,!? etc.)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # leading and trailing spaces
    text = text.strip()
    return text



## ---- Sentiment analysis ---- ##

# conversion of label+scores to continuous scale
# this will need to be updated sometimes (depending on the model used, it might have unreasonable names for the labels)
LABEL_NORMALIZATION = { 
    "positive": {"positive", "positiv", "pos"},
    "neutral": {"neutral", "neutr", "neut", "neu"},
    "negative": {"negative", "negativ", "neg"},} 
# remember to plug new ones in here if you use a new model where you suspect labels might differ

def normalize_label(label):
    """
    Normalizes the model's sentiment label to a standard format.
    """
    label = label.lower().strip() # make sure we have a clean label
    for standard_label, variants in LABEL_NORMALIZATION.items():
        if label in variants:
            return standard_label
    raise ValueError(f"Unrecognized sentiment label: {label}")

def conv_scores(label, score):
    """
    Converts the sentiment score to a continuous scale based on (normalized) label.
    """
    sentiment = normalize_label(label)
    if sentiment == "positive":
        return score
    elif sentiment == "neutral":
        return 0
    elif sentiment == "negative":
        return -score


# Function to find the maximum allowed tokens for the model
def find_max_tokens(tokenizer):
    """
    Determines the maximum token length for the tokenizer, ensuring it doesn't exceed a reasonable limit.
    """
    max_length = tokenizer.model_max_length
    if max_length > 20000:  # sometimes, they set this value to ridiculously high (although not high for real), 
        # so we default to a max
        max_length = 512
    return max_length


# split long sentences into chunks
def split_long_sentence(text, tokenizer) -> list:
    """
    Splits long sentences into chunks if their token length exceeds the model's maximum length.
    """
    words = text.split()
    parts = []
    current_part = []
    current_length = 0

    max_length = find_max_tokens(tokenizer)

    for word in words:
        # Encode word and get the token length
        tokens = tokenizer.encode(word)
        seq_len = len(tokens)

        # Check if adding this word would exceed max length
        if current_length + seq_len > max_length:
            parts.append(" ".join(current_part))  # Append the current part as a chunk
            current_part = [word]  # Start a new part with the current word
            current_length = seq_len  # Reset the current length to the length of the current word
        else:
            current_part.append(word)  # Add the word to the current chunk
            current_length += seq_len  # Update the current length

    # Append any remaining part as a chunk
    if current_part:
        parts.append(" ".join(current_part))

    return parts


# get SA scores from xlm-roberta
def get_sentiment(sentences, text_id, pipe, tokenizer):
    """
    Gets the sentiment score for a given text, including splitting long sentences into chunks if needed.
    """
    #spec_labs = get_model_labels(pipe)  # labels for the model

    sentences_scores = []
    n_chunks = []

    for sent in sentences:
        # Check that the text is a string
        if not isinstance(sent, str):
            print(f"Warning: Text is not a string for text: '{text_id}'. Skipping.")
            logger.warning(f"Expected string, got {type(text_id)}. Returning None.")
            return None
    
        # Split the sentence into chunks if it's too long
        chunks = split_long_sentence(sent, tokenizer)

        if not chunks:
            print(f"Warning: No chunks created for text: '{text_id}'. Skipping.")
            logger.warning(f"No chunks created for text: '{text_id}'. Returning None.")
            return None

        # If there is only one chunk, we can directly use the original text
        if len(chunks) == 1:
            chunks = [sent]  # Just use the original text

        # If the sentence is split into multiple chunks, print a warning
        elif len(chunks) > 1:
            print(f"Warning: Sentence split into {len(chunks)} chunks for text: '{text_id}'.")
            logger.info(f"Sentence split into {len(chunks)} chunks for text: '{text_id}'.")
        
        n_chunks.append(len(chunks))  # Store the number of chunks for this sentence

        # Loop through the chunks and get sentiment scores for each
        sentiment_scores = []

        for chunk in chunks:
            # Get sentiment from the model
            sent = pipe(chunk)
            model_label = sent[0].get("label")
            model_score = sent[0].get("score")

            # Transform score to continuous scale
            converted_score = float(conv_scores(model_label, model_score))
            sentiment_scores.append(converted_score)

        # Calculate the mean sentiment score from the chunks (if there are chunks)
        mean_score = sum(sentiment_scores) / len(sentiment_scores)
        sentences_scores.append(mean_score)

    return sentences_scores, n_chunks


## ---- Dictionary-based sentiment analysis ---- ##

# vader
def get_vader(sents: List[str]) -> List[float]:
    """
    Computes the VADER sentiment compound scores for a list of sentences.
    Returns an empty list if input is empty or None.
    """
    if not sents:
        return []

    arc = []
    for sentence in sents:
        compound_pol = sid.polarity_scores(sentence)["compound"]
        arc.append(compound_pol)

    return arc


# and syuzhet

def start_up_syuzhet():
    utils = rpackages.importr("utils")

    # select a mirror for R packages
    utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

    # install the package if is not already installed
    if not rpackages.isinstalled("syuzhet"):
        utils.install_packages("syuzhet")
    
    return None

def get_syuzhet(sents: list[str]) -> list[float]:
    """
    Computes the Syuzhet sentiment scores for a list of sentences using R.
    Returns an empty list if input is empty or None.
    """
    if not sents:
        return []

    syuzhet = rpackages.importr("syuzhet")
    syuzhet_result = syuzhet.get_sentiment(sents, method="syuzhet")

    return list(syuzhet_result)
