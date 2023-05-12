import json
import os
import pytz
from dateutil import parser
import pandas as pd


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

question_file = os.path.join(script_dir, "autocast/autocast_questions.json")
train_questions_file = os.path.join(script_dir, "train_questions.json")
test_questions_file = os.path.join(script_dir, "test_questions.json")
train_crowd_file = os.path.join(script_dir, "train_crowd.json")
test_crowd_file = os.path.join(script_dir, "test_crowd.json")

train_questions = {}
test_questions = {}
train_crowd = {}
test_crowd = {}

with open(question_file, "r") as infile:
    questions = json.load(infile)

for question in questions:
    # Filter for resolved questions.
    if question.get("status") != "Resolved":
        continue

    publish_time = parser.parse(question.get("publish_time"))
    close_time = parser.parse(question.get("close_time"))

    # Convert both datetimes to UTC
    publish_time_utc = publish_time.astimezone(pytz.UTC)
    close_time_utc = close_time.astimezone(pytz.UTC)

    # Filter for questions where we have articles from cc_news.
    if close_time_utc < pd.Timestamp("2017-01-01", tz="utc"):
        continue
    if publish_time_utc > pd.Timestamp("2018-08-01", tz="utc"):
        continue

    train_example = publish_time_utc < pd.Timestamp("2017-12-31", tz="utc")

    # Format values.
    if question.get("qtype") == "num":
        question["choices"] = ["num"]
        question["answer"] = float(question["answer"])
    elif question.get("qtype") == "t/f":
        question["choices"] = ["yes"]
        question["answer"] = question["answer"] == "yes"
    elif question.get("qtype") == "mc":
        question["answer"] = ord(question["answer"]) - ord("A")

    # Format the datetimes as strings without timezone information
    publish_time_str = publish_time_utc.strftime("%Y-%m-%d")
    close_time_str = close_time_utc.strftime("%Y-%m-%d")

    # Update the question with the new datetime strings
    question["publish_time"] = publish_time_str
    question["close_time"] = close_time_str

    # Calculate the crowd forecasts.
    forecasts = question.pop("crowd")
    crowd_forecast = (
        pd.DataFrame(
            data=[forecast["forecast"] for forecast in forecasts],
            index=pd.to_datetime(
                [forecast["timestamp"] for forecast in forecasts],
                format="ISO8601",
            ),
            columns=question["choices"],
        )
        .groupby(pd.Grouper(freq="D"))
        .mean()
        .fillna(method="ffill")
    )
    crowd_forecast.index = crowd_forecast.index.strftime("%Y-%m-%d")
    crowd_forecast = crowd_forecast.to_dict(orient="index")

    # Add question and crowd forecast to correct dataset.
    qid = question.pop("id")
    if train_example:
        train_questions[qid] = question
        train_crowd[qid] = crowd_forecast
    else:
        test_questions[qid] = question
        test_crowd[qid] = crowd_forecast

with open(train_questions_file, "w") as outfile:
    json.dump(train_questions, outfile, ensure_ascii=False, indent=4)
with open(test_questions_file, "w") as outfile:
    json.dump(test_questions, outfile, ensure_ascii=False, indent=4)
with open(train_crowd_file, "w") as outfile:
    json.dump(train_crowd, outfile, ensure_ascii=False, indent=4)
with open(test_crowd_file, "w") as outfile:
    json.dump(test_crowd, outfile, ensure_ascii=False, indent=4)
