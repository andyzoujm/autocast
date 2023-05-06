import json
import os
import pytz
from dateutil import parser


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
question_file = os.path.join(script_dir, "autocast_questions.json")

with open(question_file, "r") as infile:
    questions = json.load(infile)

processed_questions = []

for question in questions:
    # Filter out active questions.
    if question.get("status") == "Active":
        continue

    # Format values.
    if question.get("qtype") == "num":
        question["choices"] = ["num"]
    if question.get("qtype") == "t/f":
        question["choices"] = ["yes"]

    publish_time = parser.parse(question.get("publish_time"))
    close_time = parser.parse(question.get("close_time"))

    # Convert both datetimes to UTC
    publish_time_utc = publish_time.astimezone(pytz.UTC)
    close_time_utc = close_time.astimezone(pytz.UTC)

    # Format the datetimes as strings without timezone information
    publish_time_str = publish_time_utc.strftime("%Y-%m-%d")
    close_time_str = close_time_utc.strftime("%Y-%m-%d")

    # Update the question with the new datetime strings
    question["publish_time"] = publish_time_str
    question["close_time"] = close_time_str

    processed_questions.append(question)

with open(question_file, "w") as outfile:
    json.dump(processed_questions, outfile, ensure_ascii=False, indent=4)
