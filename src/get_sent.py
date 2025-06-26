import pandas as pd
import json

from transformers import pipeline, AutoTokenizer
import typer
from loguru import logger
from datetime import datetime
from pathlib import Path

from typing import List, Optional
from tqdm import tqdm

from src.utils import clean_whitespace, get_sentiment, get_syuzhet, get_vader, start_up_syuzhet

import spacy
#nlp = spacy.load("en_core_web_sm")

MAX_SAFE_LENGTH = 2_000_000

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
nlp.max_length = MAX_SAFE_LENGTH
nlp.enable_pipe("senter")

import glob

def load_processed_ids(output_dir: Path) -> set:
    processed_ids = set()
    json_files = sorted(glob.glob(str(output_dir / "partial_results_*.json")))
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for entry in data:
                    processed_ids.add(entry["text_id"])
            except Exception as e:
                logger.warning(f"Could not load {jf}: {e}")
    return processed_ids

app = typer.Typer()
logger.add("logs/sentiment.log", format="{time} {message}")


@app.command()
def main(
    model_names: List[str] = typer.Option(..., help="List of HuggingFace model names"),
    dataset_name: str = typer.Option(..., help="path to CSV (must contain 'text' column)"),
    n_rows: Optional[int] = typer.Option(None, help="Limit to first N rows"),
    output_dir: Path = typer.Option("results", help="Directory where the results CSV will be saved"),
):
    timestamp = datetime.now()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_name_save = Path(dataset_name).stem
    # make output folder for partial results
    output_jsons_partial = output_dir / "partial_results" / dataset_name_save
    output_jsons_partial.mkdir(parents=True, exist_ok=True)
    logger.info(f"==== Starting sentiment analysis at {timestamp} ====")
    logger.info(f"Model names: {model_names}")
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Limiting to first {n_rows} rows" if n_rows else "No row limit specified")

    start_up_syuzhet()

    # load data
    df = pd.read_csv(dataset_name)
    logger.info(f"Loaded dataset: {dataset_name_save} with {len(df)} rows")

    # take only the first n_rows if specified
    if n_rows:
        df = df.head(n_rows)

    if 'text' not in df.columns:
        raise ValueError("Dataset must contain a 'text' column.")
    
    # Clean text
    df['text'] = df['text'].astype(str).apply(clean_whitespace)

    # check if we already have some processed texts
    processed_ids = load_processed_ids(output_jsons_partial)
    logger.info(f"Skipping {len(processed_ids)} already-processed texts.")

    # setup transformer models
    model_pipes = {}
    model_tokenizers = {}
    for model_name in model_names:
        model_pipes[model_name] = pipeline("text-classification", model=model_name)
        model_tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)

    # list to store results
    partial_results = []

    # main processing loop
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows"):

        # get and process text_id
        text_id = row.get('work_id', i)
        if text_id in processed_ids:
            continue
        
        # get text
        raw_text = row['text']

        if pd.isna(raw_text):
            logger.warning(f"Skipping empty or invalid text_id {text_id} due to missing text")
            continue
        
        # force text to string and strip whitespace
        text = str(raw_text).strip()
        if text == "":
            logger.warning(f"Skipping empty or invalid text_id {text_id} due to empty text after stripping")
            continue
        
        if len(text) > MAX_SAFE_LENGTH:
            logger.warning(f"Skipping text_id {text_id} due to excessive length ({len(text)} chars).")
            continue

        doc = nlp(text)
        sentences = [sent.text for sent in doc.sents]

        vader_scores = list(get_vader(sentences))
        syuzhet_scores = list(get_syuzhet(sentences))

        result_entry = {
            "text_id": text_id,
            "vader": vader_scores,
            "syuzhet": syuzhet_scores,
            "text": text,
            "sentences": sentences,
        }

        for model_name in model_names:
            pipe = model_pipes[model_name]
            tokenizer = model_tokenizers[model_name]
            try:
                scores = get_sentiment(sentences, text_id=text_id, pipe=pipe, tokenizer=tokenizer)
            except Exception as e:
                logger.warning(f"Error in model {model_name} for text_id {text_id}. Error: {e}")
                scores = [None] * len(sentences)
            save_model_name = model_name.split("/")[-1].lower()  # Use the last part of the model name for saving
            result_entry[save_model_name] = scores

        partial_results.append(result_entry)

        # Save every 100 rows
        if i % 100 == 0 and i > 0:
            json_path = output_jsons_partial / f"partial_results_{dataset_name_save}_{i}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(partial_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved partial results to {json_path}")
            partial_results = []  # clear for next chunk

    if partial_results:
        json_path = output_jsons_partial / f"partial_results_final_{dataset_name_save}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(partial_results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved final partial results to {json_path}")

    logger.info(f"Results saved to DIR: {output_jsons_partial}")
    print(f"\nSaved results to {output_jsons_partial}")
    logger.info(f"time elapsed: {datetime.now() - timestamp}")

if __name__ == "__main__":
    app()