import pandas as pd
import numpy as np

df = pd.read_csv("Harris_sentiment_results.csv")
df["UTC_Time"] = pd.to_datetime(df["UTC_Time"])

# ---- Only keep general sentiment ----
df["sentiment"] = df["general_positive"] - df["general_negative"]

# ---- Engagement Weight ----
weights = {
    'Reply_Count': 1.5,
    'Repost_Count': 2.0,
    'Like_Count': 0.5,
    'View_Count': 0.001,
    'Bookmark_Count': 1.0
}

df["weight"] = (
    df["Reply_Count"] * weights['Reply_Count'] +
    df["Repost_Count"] * weights['Repost_Count'] +
    df["Like_Count"] * weights['Like_Count'] +
    df["View_Count"] * weights['View_Count'] +
    df["Bookmark_Count"] * weights['Bookmark_Count']
)

# Avoid zero-weight tweets
df["weight"] = df["weight"].replace(0, 1)

# ---- Group by day ----
df["date"] = df["UTC_Time"].dt.date

daily = df.groupby("date").apply(
    lambda x: np.average(x["sentiment"], weights=x["weight"])
).reset_index(name="daily_general_sentiment")

daily.to_csv("Harris_daily_sentiment.csv", index=False)
print(daily.head())
