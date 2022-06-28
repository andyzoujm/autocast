This folder contains reference implementation for FiD Static and FiD Temporal experiments.

FiD Temporal takes in train/test Json files where each item has the following format:
```json
    {
        "question_id": "question id",
        "question": "question (str)",
        "answers": "acceptable answers (List) (e.g., ['yes', 'Yes'] or ['C', 'c'] or [0.32])",
        "choices": "choices (List or Dict)",
        "targets":[
            {
                "date": "date (str)",
                "target": "crowd forecast (float or List[float])",
                "ctxs": [
                    {
                    "id": "retrieved context id",
                    "title": "context title",
                    "text": "context text",
                    "score": "retrieval relevance score"
                    }
                ]
            }
        ]
    }
```

Train/Test Json files for FiD Static can be converted from temporal data with `temporal_to_static_data_converter.py`.
