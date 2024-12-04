Run the script using the following command:
```bash
fastapi dev app.py
```

Make sure you're in the same directory as the `app.py` file. If not you can use the following command to navigate to the directory:
```bash
cd Function_4/T1_PTDA
```

See the API working by opening the following URL in your browser:
```bash
http://127.0.0.1:8000/docs
```

To make a POST request to the /predict/ endpoint, send a JSON body with the following structure:
```json
{
  "success_count": 2,
  "attempt_count": 10,
  "game_score_xp": 26,
  "game_level": 3,
  "engagement_time_mins": 7
}
```

Example Response:
```json
{
  "predicted_improvement_score": "56.73%"
}
```