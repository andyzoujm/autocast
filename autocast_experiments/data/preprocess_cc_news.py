import os
import json
import pandas as pd
import time
from datasets import load_dataset, Dataset, concatenate_datasets

# export HF_DATASETS_CACHE="/data/andyzou_jiaming/dataset_cache"

cc_news = load_dataset("cc_news", split="train")

base_dir = "/data/uid1785387"

for yr in range(2016, 2023):
    base_dirs = [
        os.path.join(f"{base_dir}/ccnews", d)
        for d in sorted(os.listdir(f"{base_dir}/ccnews"))
        if d.startswith("cc_download_articles") and str(yr) in d
    ]

    time0 = time.time()

    extra_news = []
    for base_dir in base_dirs:
        print(base_dir)
        count = 0
        exception_count = 0
        for source in os.listdir(base_dir):
            some_dir = os.path.join(base_dir, source)
            if not os.path.isdir(some_dir):
                continue
            for jfile in os.listdir(some_dir):
                try:
                    f_name = os.path.join(base_dir, source, jfile)
                    obj = json.load(open(f_name, "r"))

                    new_obj = {}
                    new_obj["title"] = (
                        obj["title"].strip() if obj["title"] is not None else ""
                    )
                    new_obj["text"] = (
                        obj["maintext"].strip() if obj["maintext"] is not None else ""
                    )
                    new_obj["domain"] = (
                        obj["source_domain"].strip()
                        if obj["source_domain"] is not None
                        else ""
                    )
                    new_obj["date"] = (
                        obj["date_publish"].strip()
                        if obj["date_publish"] is not None
                        else ""
                    )
                    if not new_obj["date"]:
                        continue
                    new_obj["description"] = (
                        obj["description"].strip()
                        if obj["description"] is not None
                        else ""
                    )
                    new_obj["url"] = (
                        obj["url"].strip() if obj["url"] is not None else ""
                    )
                    new_obj["image_url"] = (
                        obj["image_url"].strip() if obj["image_url"] is not None else ""
                    )

                    extra_news.append(new_obj)
                    count += 1
                except Exception as e:
                    exception_count += 1
                    # print(e)
                    # print("ERROR: " + f_name)
                    continue
        print("EXCEPTIONS")
        percentage = exception_count / count
        print(str(exception_count) + " " + str(count) + " " + str(percentage))

    print("=== Loaded all articles", time.time() - time0)
    time0 = time.time()
    all_news = cc_news

    all_news = all_news.filter(lambda example: example["date"] != "")
    print("=== Done filtering")

    all_news = all_news.sort("date")

    def process_date(example):
        example["date"], example["time"] = example["date"].split()
        return example

    all_news = all_news.map(process_date)
    print("=== Done mapping and sorting")

    print("=== Processed all articles", time.time() - time0)
    time0 = time.time()

    Dataset.save_to_disk(all_news, f"cc_news_{yr}")

news_ds = []
for yr in range(2016, 2023):
    ds = Dataset.load_from_disk(f"cc_news_{yr}")
    news_ds.append(ds)

ds = concatenate_datasets(news_ds)
Dataset.save_to_disk(ds, "cc_news")
