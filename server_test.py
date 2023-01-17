#%%

import dataclasses
import io
import json
import logging
import time
import typing as T
from pathlib import Path

import PIL

from riffusion.datatypes import InferenceInput, InferenceOutput
from riffusion.datatypes import InferenceInput, PromptInput
from riffusion.riffusion_pipeline import RiffusionPipeline
from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import base64_util
import numpy as np

from pythonosc import dispatcher
from pythonosc import osc_server, udp_client
import os, random

import whisper
model = whisper.load_model("base.en")


# %%

PIPELINE = RiffusionPipeline.load_checkpoint(
    checkpoint="riffusion/riffusion-model-v1",
    use_traced_unet=True,
    device="cuda",
)
# %%


# Load the seed image by ID
init_image_path = Path("./seed_images/", "og_beat.png")
init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

params = SpectrogramParams(
    min_frequency=0,
    max_frequency=10000,
)
converter = SpectrogramImageConverter(params=params, device=str(PIPELINE.device))


# client
client = udp_client.SimpleUDPClient('127.0.0.1', 10018)

def generate_with_prompt(address,prompt_a, prompt_b, denoising=0.75, iteration=50):
    seed_a = 1091 #np.random.randint(0, 2147483647)
#    seed_b = np.random.randint(0, 2147483647)

    start = PromptInput(prompt=prompt_a, seed=seed_a, denoising=denoising)
#    prompt_b = prompt_a
    alpha = 0
    end = PromptInput(prompt=prompt_b, seed=seed_a, denoising=denoising)

    num_segments = 8

    for i in range(num_segments + 1):

        alpha = 1.0 / num_segments * i
        riffusion_input = InferenceInput(
            start=start,
            end=end,
            alpha=alpha,
            num_inference_steps=iteration,
#            seed_image_id=None,
        )

        image = PIPELINE.riffuse(
            riffusion_input,
            init_image=init_image,
            mask_image=None,
            osc_client=client
        )

        print(image)

        segment = converter.audio_from_spectrogram_image(
            image,
            apply_filters=True,
        )

        randid = random.randint(0, 10000)
        imagepath = f'/tmp/img_{randid}.png'
    #    utils.save_image(sample, imagepath, nrow=1, normalize=True, range=(-1, 1))
        filepath = f'/tmp/gem_{randid}.wav'
        print(filepath)

        # Saving multi channel audio file
        # Export audio to MP3 bytes
        segment.export(filepath, format="wav")

        # Export image to JPEG bytes
        image.save(imagepath, exif=image.getexif(), format="JPEG")
        client.send_message("/generated", (filepath, imagepath, i)) 


def transcribe(address, path):
    print(path)
    result = model.transcribe(path)
    print(result)
    txt = result["text"]
    client.send_message("/whisper", (txt)) 

#    client.send_message("/generated", (filepath, imagepath)) 
# %%

# server
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/generate", generate_with_prompt)
dispatcher.map("/transcribe", transcribe)

server = osc_server.ThreadingOSCUDPServer(
    ('localhost', 10015), dispatcher)
print("Serving on {}".format(server.server_address))
server.serve_forever()


# %%
