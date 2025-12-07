import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from tqdm import tqdm
import torch
import warnings
import re
from datetime import datetime
import gc

warnings.filterwarnings('ignore')

# Check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class SentimentAnalyzer:
    def __init__(self, batch_size=32):
        """
        Initialize CardiffNLP models for sentiment analysis
        """
        self.batch_size = batch_size
        self.device = device

        print("Loading models...")

        # Political sentiment model
        print("Loading political sentiment model...")
        self.political_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/xlm-twitter-politics-sentiment")
        self.political_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/xlm-twitter-politics-sentiment"
        ).to(device)
        self.political_model.eval()

        # General sentiment model
        print("Loading general sentiment model...")
        self.general_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.general_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        ).to(device)
        self.general_model.eval()

        print("Models loaded successfully!")

    def preprocess_text(self, text):
        """Clean tweet text"""
        if pd.isna(text):
            return ""
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def batch_analyze(self, texts, model, tokenizer):
        """Analyze a batch of texts"""
        # Preprocess
        clean_texts = [self.preprocess_text(t) for t in texts]

        # Filter empty texts
        valid_indices = [i for i, t in enumerate(clean_texts) if t]
        if not valid_indices:
            return [[0.33, 0.34, 0.33] for _ in texts]

        valid_texts = [clean_texts[i] for i in valid_indices]

        # Process batch
        with torch.no_grad():
            encoded = tokenizer(
                valid_texts,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            outputs = model(**encoded)
            scores = outputs.logits.cpu().numpy()
            probs = softmax(scores, axis=1)

        # Map results back
        results = []
        valid_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                results.append(probs[valid_idx].tolist())
                valid_idx += 1
            else:
                results.append([0.33, 0.34, 0.33])

        return results


def analyze_dataset(csv_path, output_path="sentiment_results.csv", batch_size=32):
    """
    Main function to analyze the entire dataset
    """
    print("=" * 60)
    print("SENTIMENT ANALYSIS")
    print("=" * 60)

    # Load data
    print(f"\nLoading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df):,} tweets")

    # Initialize analyzer
    analyzer = SentimentAnalyzer(batch_size=batch_size)

    # Prepare results columns
    df['political_negative'] = 0.0
    df['political_neutral'] = 0.0
    df['political_positive'] = 0.0
    df['general_negative'] = 0.0
    df['general_neutral'] = 0.0
    df['general_positive'] = 0.0

    # Process in batches
    total_batches = (len(df) + batch_size - 1) // batch_size
    print(f"\nProcessing {total_batches} batches...")

    start_time = datetime.now()

    for batch_start in tqdm(range(0, len(df), batch_size), desc="Analyzing"):
        batch_end = min(batch_start + batch_size, len(df))
        batch_texts = df['Tweet_Content'].iloc[batch_start:batch_end].tolist()

        # Analyze with political model
        political_scores = analyzer.batch_analyze(
            batch_texts,
            analyzer.political_model,
            analyzer.political_tokenizer
        )

        # Analyze with general model
        general_scores = analyzer.batch_analyze(
            batch_texts,
            analyzer.general_model,
            analyzer.general_tokenizer
        )

        # Store results
        for i, idx in enumerate(range(batch_start, batch_end)):
            df.loc[idx, 'political_negative'] = political_scores[i][0]
            df.loc[idx, 'political_neutral'] = political_scores[i][1]
            df.loc[idx, 'political_positive'] = political_scores[i][2]
            df.loc[idx, 'general_negative'] = general_scores[i][0]
            df.loc[idx, 'general_neutral'] = general_scores[i][1]
            df.loc[idx, 'general_positive'] = general_scores[i][2]

        # Clear memory periodically
        if batch_start % 5000 == 0 and batch_start > 0:
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()

    # Add dominant sentiment columns
    df['political_sentiment'] = df[['political_negative', 'political_neutral', 'political_positive']].idxmax(axis=1)
    df['political_sentiment'] = df['political_sentiment'].str.replace('political_', '')

    df['general_sentiment'] = df[['general_negative', 'general_neutral', 'general_positive']].idxmax(axis=1)
    df['general_sentiment'] = df['general_sentiment'].str.replace('general_', '')

    # Calculate processing time
    end_time = datetime.now()
    processing_time = (end_time - start_time).total_seconds()

    # Save results
    print(f"\nSaving results to {output_path}...")
    df.to_csv(output_path, index=False)

    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"Processing time: {processing_time / 60:.1f} minutes")
    print(f"Speed: {len(df) / processing_time:.1f} tweets/second")
    print(f"\nPolitical Sentiment Distribution:")
    print(df['political_sentiment'].value_counts())
    print(f"\nGeneral Sentiment Distribution:")
    print(df['general_sentiment'].value_counts())
    print(f"\nModel Agreement: {(df['political_sentiment'] == df['general_sentiment']).mean():.1%}")
    print(f"\nResults saved to: {output_path}")

    return df


def quick_analysis(csv_path, sample_size=100):
    """
    Quick test with a small sample
    """
    print(f"\nQuick test with {sample_size} samples...")
    df = pd.read_csv(csv_path)
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)

    # Save sample for testing
    df_sample.to_csv("test_sample.csv", index=False)
    result = analyze_dataset("test_sample.csv", "test_results.csv")

    print("\nQuick test complete!")
    return result


# Simple usage
if __name__ == "__main__":
    # Option 1: Quick test (recommended first)
    # quick_analysis("your_tweets.csv", sample_size=100)

    # Option 2: Full analysis
    analyze_dataset(
        csv_path="HarrisTweetsMerged_sorted.csv.csv",  # Your input file
        output_path="Harris_sentiment_results.csv",  # Output file name
        batch_size=32  # Batch size (16 for CPU, 32-64 for GPU)
    )