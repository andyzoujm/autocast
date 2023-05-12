import os
import time
from datasets import load_dataset
from datetime import datetime as dt

time0 = time.time()

cc_news = load_dataset("cc_news", split="train")

print("=== Loaded all articles", time.time() - time0)
time0 = time.time()


def strptime_flexible(date_string, format="%Y-%m-%d %H:%M:%S"):
    try:
        return dt.strptime(date_string, format)
    except ValueError:
        return None


cc_news = cc_news.filter(
    lambda row: bool(strptime_flexible(row["date"]))
)  # drop articles with no date.
print("=== Done filtering")

# Map needs two distinct names to work correctly, so we rename `date` column to `date_string`
cc_news = cc_news.rename_column("date", "date_string")
cc_news = cc_news.map(
    lambda row: {"date": strptime_flexible(row["date_string"])},
    remove_columns="date_string",
)
cc_news = cc_news.sort("date")
print("=== Done mapping and sorting")

# Add an ID column.
ids = list(range(len(cc_news)))
cc_news = cc_news.add_column("id", ids)

print("=== Processed all articles", time.time() - time0)
time0 = time.time()

# Get the absolute path of the current script
script_path = os.path.abspath(__file__)

# Get the directory containing the script
script_dir = os.path.dirname(script_path)

# Construct a file path relative to the script's directory
data_dir = os.path.join(script_dir, "cc_news")

cc_news.save_to_disk(dataset_path=data_dir)
