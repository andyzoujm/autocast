import json
import pytz
from dateutil import parser

input_file = "autocast_questions.json"

with open(input_file, "r") as infile:
    data = json.load(infile)

filtered_data = []

for question in data:
    if question.get("status") == "Resolved":
        publish_time = parser.parse(question.get("publish_time"))
        close_time = parser.parse(question.get("close_time"))

        # Convert both datetimes to UTC
        publish_time_utc = publish_time.astimezone(pytz.UTC)
        close_time_utc = close_time.astimezone(pytz.UTC)

        # Format the datetimes as strings without timezone information
        publish_time_str = publish_time_utc.strftime("%Y-%m-%dT%H:%M:%S")
        close_time_str = close_time_utc.strftime("%Y-%m-%dT%H:%M:%S")

        # Update the question with the new datetime strings
        question["publish_time"] = publish_time_str
        question["close_time"] = close_time_str

        filtered_data.append(question)

with open(input_file, "w") as outfile:
    json.dump(filtered_data, outfile, ensure_ascii=False, indent=4)
