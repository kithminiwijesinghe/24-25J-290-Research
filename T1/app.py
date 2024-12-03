from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import nltk
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import os
import tempfile
import torchaudio

import torchaudio.transforms as T
import soundfile as sf
from jiwer import wer

# Ensure NLTK data is available
nltk.download("punkt")
nltk.download('punkt_tab') 

# Initialize FastAPI
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI server!"}


# Initialize Whisper model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny.en")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")

# Scoring function
def scoring(words, transcriptions):
    words = words.lower()
    transcriptions = transcriptions.lower()

    unwanted = [".", ",", "/", "?", "-", ";", ":", "`", "@", "&", "%", "*"]

    clean_words = [i for i in nltk.word_tokenize(words) if i not in unwanted]
    clean_voices = [i for i in nltk.word_tokenize(transcriptions) if i not in unwanted]

    # Technique 1: Sentence and word comparison
    words_sent = nltk.sent_tokenize(words)
    voice_sent = nltk.sent_tokenize(transcriptions)

    write_sentences = []
    write_word = []
    missing_voice = []

    for i, j in enumerate(words_sent):
        for k, l in enumerate(voice_sent):
            if i == k:
                # Clean sentences
                j = " ".join([a for a in nltk.word_tokenize(j) if a not in unwanted])
                l = " ".join([b for b in nltk.word_tokenize(l) if b not in unwanted])

                if j == l:
                    write_sentences.append(l)
                else:
                    text_words = nltk.word_tokenize(j)
                    voice_words = nltk.word_tokenize(l)
                    for q, w in enumerate(text_words):
                        if q < len(voice_words) and w == voice_words[q]:
                            write_word.append(w)
                        else:
                            missing_voice.append(w)

    sentences_score1 = len(write_sentences) / len(words_sent) * 100 if words_sent else 0
    word_score1 = len(write_word) / len(clean_words) * 100 if clean_words else 0

    # Technique 2: Token comparison
    write_word2 = [i for i in clean_words if i in clean_voices]
    sentences_score2 = len(set(write_word2)) / len(set(clean_words)) * 100 if clean_words else 0

    # Final scores
    final_sent_score = max(sentences_score1, sentences_score2)
    final_word_score = max(word_score1, len(write_word2) / len(clean_words) * 100 if clean_words else 0)

    return final_sent_score, final_word_score, missing_voice

# API Models
class ScoringResponse(BaseModel):
    sentence_score: float
    word_score: float
    missing_words: list

# API Endpoints
@app.post("/upload-audio/")
async def upload_audio(original_text: str = Form(...), file: UploadFile = File(...)):
    temp_path = f"{tempfile.mktemp()}.wav"
    try:
        # Save uploaded file
        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())
        
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
        predicted_text = model.generate(input_features, num_beams=5)

        # Convert predicted text to string
        predicted_text = processor.decode(predicted_text[0])
        predicted_text = predicted_text.replace("<|startoftranscript|>", "").replace("<|notimestamps|>", "").strip()

        # Compute scoring (WER)
        wer_score = wer(original_text, predicted_text)

        return {"original_text": original_text, "predicted_text": predicted_text, "wer_score": wer_score}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process audio: {str(e)}"})

    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# Run server
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
