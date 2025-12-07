import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("TrumpTweets_cleaned.csv", low_memory=False)

df["UTC_Time"] = pd.to_datetime(df["UTC_Time"])

# Convert engagement columns to numeric
for col in ["Reply_Count", "Repost_Count", "Like_Count", "View_Count", "Bookmark_Count"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

# ==== Political Sentiment Score ====
df["political_score"] = df["political_positive"] - df["political_negative"]

# ==== Engagement Weight ====
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

df["weight"] = df["weight"].replace(0, 1)

# ==== Aggregate Daily ====
df["date"] = df["UTC_Time"].dt.date

daily_pol = df.groupby("date", group_keys=False).apply(
    lambda x: np.average(x["political_score"], weights=x["weight"])
).reset_index(name="daily_political_sentiment")

# ==== Fill missing dates with 0 (neutral) ====
daily_pol = daily_pol.set_index("date")
daily_pol = daily_pol.asfreq('D', fill_value=0)
daily_pol.reset_index(inplace=True)

daily_pol.to_csv("Trump_daily_political_sentiment.csv", index=False)

print(daily_pol.head())
