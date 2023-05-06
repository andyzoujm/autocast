import pandas as pd
import json
import os


def load_questions(question_file):
    """Load questions from a JSON file."""
    with open(question_file, "r") as file:
        questions = json.load(file)
    return questions


def process_crowd_forecasts(questions):
    """Process crowd forecasts and return them as a dictionary."""
    crowd_forecasts = {}
    for question in questions:
        forecasts = question["crowd"]
        crowd_forecast = (
            pd.DataFrame(
                data=[forecast["forecast"] for forecast in forecasts],
                columns=question["choices"],
                index=pd.to_datetime(
                    [forecast["timestamp"] for forecast in forecasts]
                ),
            )
            .groupby(pd.Grouper(freq="D"))
            .mean()
        )
        crowd_forecast.index = crowd_forecast.index.strftime("%Y-%m-%d")
        crowd_forecasts[question["id"]] = crowd_forecast.to_dict(orient="index")
    return crowd_forecasts


def save_crowd_forecasts(crowd_forecasts, crowd_forecast_file):
    """Save crowd forecasts to a JSON file."""
    with open(crowd_forecast_file, "w") as outfile:
        json.dump(crowd_forecasts, outfile, ensure_ascii=False, indent=4)


def main():
    # Set up file paths
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    question_file = os.path.join(script_dir, "autocast_questions.json")
    crowd_forecast_file = os.path.join(script_dir, "crowd_forecast.json")
    
    # Load questions from the JSON file
    questions = load_questions(question_file)
    
    # Process crowd forecasts
    crowd_forecasts = process_crowd_forecasts(questions)
    
    # Save crowd forecasts to a JSON file
    save_crowd_forecasts(crowd_forecasts, crowd_forecast_file)


if __name__ == "__main__":
    main()

