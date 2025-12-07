import pandas as pd

df = pd.read_csv("Harris_sentiment_results.csv")

#  Post_ID + Tweet_Content + Tweet_URL remove duplicate
df = df.drop_duplicates(
    subset=["Post_ID", "Tweet_Content", "Tweet_URL"],
    keep="first"
)

df.to_csv("HarrisTweets_cleaned.csv", index=False)
