Prerequisites

If error indicates that Torchaudio couldn't find an appropriate backend to handle the .wav file. This issue is often caused by missing dependencies or incorrect installation of the Torchaudio library, particularly for handling .wav files. Torchaudio relies on FFmpeg for handling .wav and other audio formats. Install [FFmpeg](https://ffmpeg.org/download.html) on your system.s

## Environment

1. Navigate to Your Project Directory `cd Function_2`
2. Create a Virtual Environment `python -m venv venv`
3. Activate Virtual Environment `source venv/Scripts/activate`
4. Deactivate Virtual Environment `deactivate` (Optional)

## Installation

1. Install Required Libraries `pip install -r requirements.txt`
2. Create requirements.txt `pip freeze > requirements.txt` (If needed)

## Run

Run application using `fastapi dev app.py` (default) command.

Task 01: `fastapi dev T1/app.py`
Task 02: `fastapi dev T2/app.py`
Task 03: `fastapi dev T3/app.py`
Task 04: `fastapi dev T4/app.py`