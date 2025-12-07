import pandas as pd

df = pd.read_csv("HarrisT1.csv")

# 假设列名是 UTC_Time
df["UTC_Time"] = pd.to_datetime(df["UTC_Time"])

# 按时间排序
df = df.sort_values(by="UTC_Time")

# 保存
df.to_csv("HarrisTweetsFinal.csv", index=False)
