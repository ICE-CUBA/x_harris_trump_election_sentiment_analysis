import pandas as pd

# 读取两个 CSV
df1 = pd.read_csv("HarrisTweetsFinal.csv")
df2 = pd.read_csv("HarrisTweets_only_1104.csv")

# 拼接
df = pd.concat([df1, df2], ignore_index=True)

# 转换时间格式（非常关键）
df["UTC_Time"] = pd.to_datetime(df["UTC_Time"])

# 按时间排序
df = df.sort_values(by="UTC_Time")

# 保存
df.to_csv("HarrisTweetsMerged_sorted.csv", index=False)
