# Twitter Sentiment Analysis Project

This project performs sentiment analysis on Twitter/X data related to political figures (Kamala Harris and Donald Trump) using state-of-the-art transformer models. The analysis includes both political-specific and general sentiment classification, with daily aggregation weighted by engagement metrics.

## Project Overview

This sentiment analysis pipeline processes Twitter data through multiple stages:
1. **Data Preprocessing**: Cleaning, deduplication, and time-based sorting
2. **Sentiment Analysis**: Dual-model analysis using CardiffNLP transformers
3. **Daily Aggregation**: Engagement-weighted daily sentiment scores

## Features

- **Dual Sentiment Models**: 
  - Political sentiment model (`cardiffnlp/xlm-twitter-politics-sentiment`)
  - General sentiment model (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- **Batch Processing**: Efficient GPU/CPU batch processing for large datasets
- **Engagement Weighting**: Daily sentiment scores weighted by reply, repost, like, view, and bookmark counts
- **Time Series Analysis**: Daily sentiment aggregation for trend analysis

## Project Structure

```
sentiment/
├── sentiment_analysis.py      # Main sentiment analysis script
├── daily_sentiment.py          # Daily sentiment aggregation
├── pinTogether.py              # Merge and sort CSV files
├── remove_duplicate.py         # Remove duplicate tweets
├── sort_by_time.py             # Sort tweets by timestamp
├── HarrisTweetsFinal.csv       # Processed Harris tweets
├── TrumpTweetsFinal.csv        # Processed Trump tweets
├── Harris_sentiment_results.csv    # Sentiment analysis results
├── Trump_sentiment_results.csv     # Sentiment analysis results
├── Harris_daily_sentiment.csv      # Daily aggregated sentiment
└── Trump_daily_sentiment.csv       # Daily aggregated sentiment
```

## Requirements

### Python Packages

```bash
pip install pandas numpy torch transformers scipy tqdm
```

### Hardware Requirements

- **GPU Recommended**: CUDA-compatible GPU for faster processing
- **CPU Supported**: Works on CPU but slower (reduce batch_size to 16)
- **Memory**: At least 8GB RAM recommended for large datasets

## Usage

### 1. Data Preprocessing

#### Sort tweets by time:
```python
python sort_by_time.py
```
Converts `UTC_Time` to datetime and sorts tweets chronologically.

#### Remove duplicates:
```python
python remove_duplicate.py
```
Removes duplicate tweets based on `Post_ID`, `Tweet_Content`, and `Tweet_URL`.

#### Merge multiple CSV files:
```python
python pinTogether.py
```
Merges multiple tweet CSV files and sorts by timestamp.

### 2. Sentiment Analysis

#### Full Analysis:
```python
python sentiment_analysis.py
```

The script will:
- Load the specified CSV file (default: `HarrisTweetsMerged_sorted.csv`)
- Process tweets in batches (default: 32)
- Apply both political and general sentiment models
- Save results with sentiment scores and classifications

#### Quick Test (100 samples):
```python
# Uncomment in sentiment_analysis.py:
quick_analysis("your_tweets.csv", sample_size=100)
```

#### Custom Analysis:
```python
from sentiment_analysis import analyze_dataset

analyze_dataset(
    csv_path="your_tweets.csv",
    output_path="output_results.csv",
    batch_size=32  # 16 for CPU, 32-64 for GPU
)
```

**Output Columns:**
- `political_negative`, `political_neutral`, `political_positive`: Probability scores
- `general_negative`, `general_neutral`, `general_positive`: Probability scores
- `political_sentiment`: Dominant political sentiment (negative/neutral/positive)
- `general_sentiment`: Dominant general sentiment (negative/neutral/positive)

### 3. Daily Sentiment Aggregation

```python
python daily_sentiment.py
```

This script:
- Calculates sentiment score: `general_positive - general_negative`
- Applies engagement weights:
  - Reply Count: 1.5x
  - Repost Count: 2.0x
  - Like Count: 0.5x
  - View Count: 0.001x
  - Bookmark Count: 1.0x
- Aggregates weighted average sentiment by date

**Output:** CSV file with `date` and `daily_general_sentiment` columns

## Sentiment Models

### Political Sentiment Model
- **Model**: `cardiffnlp/xlm-twitter-politics-sentiment`
- **Purpose**: Specialized for political content on Twitter
- **Output**: Negative, Neutral, Positive probabilities

### General Sentiment Model
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Purpose**: General-purpose Twitter sentiment analysis
- **Output**: Negative, Neutral, Positive probabilities

## Data Format

### Input CSV Required Columns:
- `Tweet_Content`: The tweet text to analyze
- `UTC_Time`: Timestamp in datetime format
- `Reply_Count`, `Repost_Count`, `Like_Count`, `View_Count`, `Bookmark_Count`: Engagement metrics

### Example:
```csv
Tweet_Content,UTC_Time,Reply_Count,Repost_Count,Like_Count,View_Count,Bookmark_Count
"Example tweet text",2024-06-04 00:51:50+00:00,10,5,100,1000,2
```

## Performance

- **Processing Speed**: ~10-50 tweets/second (depending on hardware)
- **Batch Size Recommendations**:
  - CPU: 16
  - GPU: 32-64
- **Memory Management**: Automatic garbage collection every 5000 tweets

## Output Files

1. **`*_sentiment_results.csv`**: Full sentiment analysis with all probability scores
2. **`*_daily_sentiment.csv`**: Daily aggregated sentiment scores (engagement-weighted)

## Notes

- Models are automatically downloaded on first run (requires internet connection)
- Empty or invalid tweets are handled gracefully with neutral scores
- URLs are automatically removed from tweet text during preprocessing
- Processing time and statistics are displayed after completion

## Example Workflow

```bash
# 1. Sort your raw data
python sort_by_time.py

# 2. Remove duplicates
python remove_duplicate.py

# 3. Run sentiment analysis
python sentiment_analysis.py

# 4. Generate daily sentiment
python daily_general_sentiment.py
```

## License

This project uses pre-trained models from CardiffNLP. Please refer to their respective licenses for model usage.

## Author

NEU 6140 ML Course Project

