

# Sentiment Analysis for fanfic & other texts 🎭

this script runs sentiment analysis on text datasets (like fanfic or whatever you have) using huggingface transformer models 🤗 plus dictionary-based tools like vader and syuzhet.

## what it does ⚙️

- takes a csv dataset with a `text` column (and optionally a `work_id` or similar id)  
- cleans up the text (whitespace, etc)  
- skips empty, missing, or super long texts (>2 million chars)  
- splits text into sentences using spacy’s (En) sentence segmenter  -- so we are scoring *sentences* here
- runs sentiment analysis with vader, syuzhet, and any huggingface transformer models you specify
- saves partial results as json files along the way so you don’t lose progress  
- logs everything for debugging  

## how to do 🚀

install requirements, then, in terminal:

```bash

python -m src.get_sent \
    --dataset-name data/MythFic_texts.csv \
    --model-names cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual \
    --n-rows 5

```
- dataset path
- HF model names (can be multiple); *still when adding a new one please make sure that the labels (pos, neg, neutr) map onto the expected labels in* ```utils.LABEL_NORMALIZATION```
- n-rows (optional) is for testing the script and setup, for example 5 rows


## details
### SA transformer-based details
- if any sentence is too long (i.e., exceeds model max tokens), it will be chunked into the necessary chunks and the scores returned will be an average of all chunks
- max token length is retrieved automatically, so chunking will conform to the max token length of the model
- binary scores (pos, neg, neutr) are converted to a continuous scale using the confidence score for the assingment. So a score of pos, confidence: 0.6 will give us a score of 0.6. A score of neg, confidence: 0.6 will give us -0.6. Neutral labels will always tranform to 0. For more detail, see ```utils.get_sentiment```