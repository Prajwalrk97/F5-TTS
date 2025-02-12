# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

from contextlib import asynccontextmanager
import io
import json
import os
import random
import sys
import tempfile
import time
import uuid
import asyncpg
import pandas as pd
import numpy as np
import soundfile as sf
import torchaudio
import uvicorn

from transformers import pipeline
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from datasets import Dataset
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

from f5_tts.infer.infer_gradio import load_custom, load_e2tts, load_f5tts
from f5_tts.infer.utils_infer import postgres_async_update
from f5_tts.model.utils import seed_everything
import torch
try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


from f5_tts.infer.utils_infer import (
    load_speech_types,
    load_vocoder,
    parse_speechtypes_text,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
    split_sentences,
)

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_TABLE = os.getenv("POSTGRES_TABLE")

DEFAULT_TTS_MODEL = "F5-TTS"
USING_SPACES = False
TTS_MODEL_CHOICE = DEFAULT_TTS_MODEL
SPEECH_TYPES_DIRECTORY = "ref_audio"
DEFAULT_TTS_MODEL_CFG = [
    "hf://prajwalrk/arsene-wenger-tts/model_160000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt",
    json.dumps(dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)),
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle and resource initialization."""
    try:
        app.speech_types = load_speech_types(SPEECH_TYPES_DIRECTORY)
        
        # load models
        app.vocoder = load_vocoder()
        app.F5TTS_ema_model = load_f5tts()
        app.E2TTS_ema_model = load_e2tts() if USING_SPACES else None
        app.pool = await asyncpg.create_pool(
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            database=POSTGRES_DB,
            host=POSTGRES_HOST,
        )
        yield
    finally:
        if hasattr(app, 'pool'):
            await app.pool.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TTSRequest(BaseModel):
    text: str
    message_id: uuid.UUID = uuid.uuid4()
    remove_silence: bool = False
    seed: int = -1

custom_ema_model, pre_custom_path = None, ""
chat_model_state = None
chat_tokenizer_state = None




def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.2,
    nfe_step=64,
    speed=1,
    show_info=print,
):
    if not ref_audio_orig:
        raise ValueError("Please provide reference audio.")

    if not gen_text.strip():
        raise ValueError("Please enter text to generate.")

    ref_audio, ref_text = preprocess_ref_audio_text(
        ref_audio_orig, ref_text, show_info=show_info
    )

    if model == "F5-TTS":
        ema_model = app.F5TTS_ema_model
    elif model == "E2-TTS":
        ema_model = app.E2TTS_ema_model
    elif isinstance(model, list) and model[0] == "Custom":
        global custom_ema_model, pre_custom_path
        if pre_custom_path != model[1]:
            show_info("Loading Custom TTS model...")
            custom_ema_model = load_custom(
                model[1], vocab_path=model[2], model_cfg=model[3]
            )
            pre_custom_path = model[1]
        ema_model = custom_ema_model
    else:
        raise ValueError("Invalid model specified.")

    final_wave, final_sample_rate, combined_spectrogram = infer_process(
        ref_audio,
        ref_text,
        gen_text,
        ema_model,
        app.vocoder,
        cross_fade_duration=cross_fade_duration,
        nfe_step=nfe_step,
        speed=speed,
        show_info=show_info,
    )

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, final_sample_rate)
            remove_silence_for_generated_wav(f.name)
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    return (final_sample_rate, final_wave), ref_text


async def generate_emotion_tags(text: str) -> str:
    sentences = split_sentences(text)
    df = pd.DataFrame({"text": sentences})
    dataset = Dataset.from_pandas(df)
    classifier = pipeline(
        "text-classification", 
        model="j-hartmann/emotion-english-distilroberta-base", 
        top_k=1,
        device=0    # cuda
    )
    
    def _process_batch(examples):
        outputs = classifier(examples["text"])
        return {"emotion": [output[0] for output in outputs]}
    
    results = dataset.map(
        _process_batch,
        batched=True,
        batch_size=50
    )
    torch.cuda.empty_cache()

    text_with_emotions = ""
    prev_emotion = ""
    for item in results:
        try:
            cur_emotion = str(item["emotion"]["label"]).strip()
            if cur_emotion.title() not in app.speech_types:
                cur_emotion = "neutral"
            if prev_emotion == cur_emotion:
                text_with_emotions += item["text"] + " "
            else:
                text_with_emotions += "{" + cur_emotion.title() + "} " + item["text"] + " "
            prev_emotion = cur_emotion
        except:
            text_with_emotions += "{Neutral} " + item["text"] + " "
            prev_emotion = cur_emotion
    
    return text_with_emotions


@app.post("/generate_tts/")
async def generate_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    torch.cuda.empty_cache()
    start_time = time.time()
    try:
        gen_text = request.text
        message_id = request.message_id
        remove_silence = request.remove_silence
        seed = request.seed
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)

        gen_text_with_emotions = await generate_emotion_tags(gen_text)
        print(gen_text_with_emotions)
        segments = parse_speechtypes_text(gen_text_with_emotions)

        generated_audio_segments = []
        current_style = "Neutral"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in app.speech_types:
                current_style = style
                if style == "Angry":
                    cross_fade_duration=0.2
                    speed=1
                if style == "Sadness":
                    cross_fade_duration=0.2
                    speed=1
                if style == "Laughing":
                    cross_fade_duration=0.1
                    speed=1
                if style == "Neutral":
                    cross_fade_duration=0.2
                    speed=1

            else:
                current_style = "Neutral"
                cross_fade_duration=0.2
                speed=1

            ref_audio = app.speech_types[current_style]["audio"]
            ref_text = app.speech_types[current_style].get("ref_text", "")

            audio_out, ref_text_out = infer(
                ref_audio,
                ref_text,
                text,
                TTS_MODEL_CHOICE,
                remove_silence=remove_silence,
                cross_fade_duration=cross_fade_duration,
                speed=speed,
                show_info=print,
            )
            sr, audio_data = audio_out
            generated_audio_segments.append(audio_data)
            app.speech_types[current_style]["ref_text"] = ref_text_out

        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)

            output_directory = "gen_audio"
            output_filename = f"{seed}.wav"
            output_path = os.path.join(output_directory, output_filename)

            os.makedirs(output_directory, exist_ok=True)
            sf.write(output_path, final_audio_data, sr)
            print(f"Seed: {seed}")
            print(f"Sample rate: {sr}")
            # background_tasks.add_task(
            #     postgres_async_update,
            #     app=app,
            #     message_id=message_id,
            #     message_audio=final_audio_data,
            #     seed=str(seed)
            # )
            return {
                "message": "TTS generation successful",
                "filepath": output_filename,
                "seed" : str(seed),
                "elapsed_time": time.time()-start_time
            }
        else:
            raise HTTPException(status_code=400, detail="No audio generated.")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/streaming_tts/")
async def streaming_tts(request: TTSRequest, background_tasks: BackgroundTasks):
    torch.cuda.empty_cache()
    try:
        gen_text = request.text
        message_id = request.message_id
        remove_silence = request.remove_silence
        seed = request.seed
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)

        gen_text_with_emotions = await generate_emotion_tags(gen_text)
        segments = parse_speechtypes_text(gen_text_with_emotions)

        generated_audio_segments = []
        current_style = "Neutral"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in app.speech_types:
                current_style = style
                if style == "Angry":
                    cross_fade_duration=0.2
                    speed=1
                if style == "Sadness":
                    cross_fade_duration=0.2
                    speed=0.9
                if style == "Laughing":
                    cross_fade_duration=0.1
                    speed=1
                if style == "Neutral":
                    cross_fade_duration=0.2
                    speed=0.9

            else:
                current_style = "Neutral"
                cross_fade_duration=0.2
                speed=0.9

            ref_audio = app.speech_types[current_style]["audio"]
            ref_text = app.speech_types[current_style].get("ref_text", "")

            audio_out, ref_text_out = infer(
                ref_audio,
                ref_text,
                text,
                TTS_MODEL_CHOICE,
                remove_silence=remove_silence,
                cross_fade_duration=cross_fade_duration,
                speed=speed,
                show_info=print,
            )
            sr, audio_data = audio_out
            generated_audio_segments.append(audio_data)
            app.speech_types[current_style]["ref_text"] = ref_text_out

        # Concatenate all segments
        final_audio = np.concatenate(generated_audio_segments)
        
        # Create final WAV buffer
        final_buffer = io.BytesIO()
        # sf.write(final_buffer, final_audio, sr, format='WAV')
        sf.write(final_buffer, final_audio, sr, format='WAV', subtype='PCM_16')
        final_buffer.seek(0)

        print(f"Seed: {seed}")
        print(f"Sample rate: {sr}")
        background_tasks.add_task(
                postgres_async_update,
                app=app,
                message_id=message_id,
                message_audio=final_audio,
                seed=str(seed)
            )
        return Response(
            content=final_buffer.getvalue(),
            media_type="audio/wav",
            headers={
                'Content-Type': 'audio/wav',
                'Content-Disposition': 'attachment; filename="audio.wav"'
            }
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
    )
