from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import os
import tempfile
import torchaudio

import torchaudio.transforms as T
import soundfile as sf

import google.generativeai as genai
import time
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Whisper model

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
WhisperModel = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Gemini API configuration
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

def upload_to_gemini(path, mime_type=None):
  file = genai.upload_file(path, mime_type=mime_type)
  print(f"Uploaded file '{file.display_name}' as: {file.uri}")
  return file

def wait_for_files_active(files):
  print("Waiting for file processing...")
  for name in (file.name for file in files):
    file = genai.get_file(name)
    while file.state.name == "PROCESSING":
      print(".", end="", flush=True)
      time.sleep(10)
      file = genai.get_file(name)
    if file.state.name != "ACTIVE":
      raise Exception(f"File {file.name} failed to process")
  print("...all files ready")
  print()

# FastAPI app
app = FastAPI()

@app.post("/transcribe_and_compare")
async def transcribe_and_compare_endpoint(audio_file: UploadFile = File(...), story_name: str = Form(...)):
    temp_path = f"{tempfile.mktemp()}.wav"
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            buffer.write(await audio_file.read())
        
        # Load the audio file
        waveform, sample_rate = torchaudio.load(temp_path)

        # Convert stereo to mono
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample to 16 kHz if necessary
        if sample_rate != 16000:
            resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Process waveform
        input_features = processor(waveform.squeeze(0), sampling_rate=sample_rate, return_tensors="pt").input_features

        # Run the Whisper model
        transcribed_text = WhisperModel.generate(input_features, num_beams=5)

        # Convert predicted text to string
        transcribed_text = processor.decode(transcribed_text[0])
        transcribed_text = transcribed_text.replace("<|startoftranscript|>", "").replace("<|notimestamps|>", "").strip()

        # Gemini API call (similar to your previous code)
        file_path_string = "../../models/F2T3/story.csv"  # Replace with actual path
        file = upload_to_gemini(file_path_string, mime_type='text/csv')
        files = [file]
        wait_for_files_active(files)

        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        chat_session = model.start_chat(
           history=[
                {
                "role": "user",
                "parts": [
                    files[0],
                    "Refer to this story.csv file and output an integer representing how the end of the story matches with \"transcribed_text\". ",
                ],
                },
                {
                "role": "model",
                "parts": [
                    "I will compare the ending of the story with the \"transcribed_text\" and output a similarity score.",
                ],
                },
            ]
        )

        response = chat_session.send_message(
            f"""
            Compare the ending of story name {story_name} in the provided story.csv
            with the following transcribed text: {transcribed_text}
            Output an integer representing as the similarity score.
            """
        )

        similarity_score = response.text  # Assuming response is an integer

        return {"transcription": transcribed_text, "similarity": similarity_score}

    except Exception as e:
        return {"error": str(e)}