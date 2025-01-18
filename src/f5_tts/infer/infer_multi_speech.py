# ruff: noqa: E402
# Above allows ruff to ignore E402: module level import not at top of file

import json
import os
import random
import re
import sys
import tempfile
from collections import OrderedDict
import time

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel
import soundfile as sf
import torchaudio
from cached_path import cached_path
import uvicorn

from f5_tts.model.utils import seed_everything
import torch
try:
    import spaces

    USING_SPACES = True
except ImportError:
    USING_SPACES = False


def gpu_decorator(func):
    if USING_SPACES:
        return spaces.GPU(func)
    else:
        return func


from f5_tts.model import DiT, UNetT
from f5_tts.infer.utils_infer import (
    load_vocoder,
    load_model,
    preprocess_ref_audio_text,
    infer_process,
    remove_silence_for_generated_wav,
)


DEFAULT_TTS_MODEL = "F5-TTS"
tts_model_choice = DEFAULT_TTS_MODEL

DEFAULT_TTS_MODEL_CFG = [
    r"C:\F5-TTS\ckpts\arsene_wenger_tts\model_160000.safetensors",
    "hf://SWivid/F5-TTS/F5TTS_Base/vocab.txt",
    json.dumps(
        dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
    ),
]


# load models

vocoder = load_vocoder()


def load_f5tts(ckpt_path=r"C:\F5-TTS\ckpts\arsene_wenger_tts\model_160000.safetensors"):
    F5TTS_model_cfg = dict(
        dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
    )
    return load_model(DiT, F5TTS_model_cfg, ckpt_path)


def load_e2tts(ckpt_path=r"C:\F5-TTS\ckpts\arsene_wenger_tts\model_160000.safetensors"):
    E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)
    return load_model(UNetT, E2TTS_model_cfg, ckpt_path)


def load_custom(ckpt_path: str, vocab_path="", model_cfg=None):
    ckpt_path, vocab_path = ckpt_path.strip(), vocab_path.strip()
    if ckpt_path.startswith("hf://"):
        ckpt_path = str(cached_path(ckpt_path))
    if vocab_path.startswith("hf://"):
        vocab_path = str(cached_path(vocab_path))
    if model_cfg is None:
        model_cfg = dict(
            dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
        )
    return load_model(DiT, model_cfg, ckpt_path, vocab_file=vocab_path)


F5TTS_ema_model = load_f5tts()
E2TTS_ema_model = load_e2tts() if USING_SPACES else None
custom_ema_model, pre_custom_path = None, ""

chat_model_state = None
chat_tokenizer_state = None


# @gpu_decorator
# def generate_response(messages, model, tokenizer):
#     """Generate response using Qwen"""
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )

#     model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=512,
#         temperature=0.7,
#         top_p=0.95,
#     )

#     generated_ids = [
#         output_ids[len(input_ids) :]
#         for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
#     ]
#     return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def infer(
    ref_audio_orig,
    ref_text,
    gen_text,
    model,
    remove_silence,
    cross_fade_duration=0.15,
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
        ema_model = F5TTS_ema_model
    elif model == "E2-TTS":
        ema_model = E2TTS_ema_model
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
        vocoder,
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


def parse_speechtypes_text(gen_text):
    # Pattern to find {speechtype}
    pattern = r"\{(.*?)\}"

    # Split the text by the pattern
    tokens = re.split(pattern, gen_text)

    segments = []

    current_style = "Regular"

    for i in range(len(tokens)):
        if i % 2 == 0:
            # This is text
            text = tokens[i].strip()
            if text:
                segments.append({"style": current_style, "text": text})
        else:
            # This is style
            style = tokens[i].strip()
            current_style = style

    return segments


app = FastAPI()


class TTSRequest(BaseModel):
    text: str
    remove_silence: bool = False
    seed: int = -1


# Load predefined speech types from a directory
def load_speech_types(speech_types_dir):
    speech_types = OrderedDict()
    for filename in os.listdir(speech_types_dir):
        if filename.endswith(".wav"):
            speech_type_name = filename[:-4]  # Remove .wav extension
            filepath = os.path.join(speech_types_dir, filename)
            ref_text = open(os.path.join(speech_types_dir, f"{speech_type_name}.txt")).read()
            speech_types[speech_type_name] = {"audio": filepath, "ref_text": ref_text}
    return speech_types


# Replace with the actual directory containing your speech type audio files
SPEECH_TYPES_DIRECTORY = "ref_audio"
speech_types = load_speech_types(SPEECH_TYPES_DIRECTORY)


@app.post("/generate_tts/")
async def generate_tts(request: TTSRequest):
    torch.cuda.empty_cache()
    start_time = time.time()
    try:
        gen_text = request.text
        remove_silence = request.remove_silence
        seed = request.seed
        if seed == -1:
            seed = random.randint(0, sys.maxsize)
        seed_everything(seed)
        segments = parse_speechtypes_text(gen_text)

        generated_audio_segments = []
        current_style = "Regular"

        for segment in segments:
            style = segment["style"]
            text = segment["text"]

            if style in speech_types:
                current_style = style
                if style == "Angry":
                    cross_fade_duration=0.2
                    speed=1
                if style == "Sad":
                    cross_fade_duration=0.2
                    speed=1
                if style == "Laughing":
                    cross_fade_duration=0.1
                    speed=0.8
            else:
                current_style = "Regular"
                cross_fade_duration=0.3
                speed=1

            ref_audio = speech_types[current_style]["audio"]
            ref_text = speech_types[current_style].get("ref_text", "")

            audio_out, ref_text_out = infer(
                ref_audio,
                ref_text,
                text,
                tts_model_choice,
                remove_silence,
                cross_fade_duration=cross_fade_duration,
                speed=speed,
                show_info=print,
            )
            sr, audio_data = audio_out
            generated_audio_segments.append(audio_data)
            speech_types[current_style]["ref_text"] = ref_text_out

        if generated_audio_segments:
            final_audio_data = np.concatenate(generated_audio_segments)

            output_directory = "gen_audio"
            output_filename = f"{seed}.wav"
            output_path = os.path.join(output_directory, output_filename)

            os.makedirs(output_directory, exist_ok=True)
            sf.write(output_path, final_audio_data, sr)

            return {
                "message": "TTS generation successful",
                "filepath": output_filename,
                "seed" : int(seed),
                "elapsed_time": time.time()-start_time
            }
        else:
            raise HTTPException(status_code=400, detail="No audio generated.")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# "{Angry} During the match, i was very angry and frustrated with the referee for the tackle on Eduardo. It was a horrible tackle from the Birmingham player which broke his leg... I told the player that he should never be allowed on a football pitch again.\n{Sad} The match completely changed our season in the 2007 2008 season, as we were uhh... scarred, from watching Eduardo being stretchered off the pitch. We uhh lost our momentum after that. We were 6 points ahead of Manchester United in second but in the end, we finished third in May. I sometimes feel very sad you know? because, we had a real chance of winning the championship that season and for Eduardo as well, because uhh it was a horrific injury for a player to suffer.\n{Regular} But that's football, you know? You have to pick yourself up. You don't get time to feel sorry for yourself, you just have to go again the next game because that is the Premier League"