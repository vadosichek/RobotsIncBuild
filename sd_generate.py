import io
import os
import sys
import warnings
from random import randint

from PIL import Image
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

import requests

os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = 'sk-6VgfTZ3vUhPOZmbIJ9v4xsxmsHXCMeFzuk8AIjESwrimkKEH'

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-512-v2-0", # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
    # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0
    # stable-diffusion-512-v2-1 stable-diffusion-768-v2-1 stable-inpainting-v1-0 stable-inpainting-512-v2-0
)
print("->client created")

img = Image.open("~/Documents/RovotsINC/SavedScreen.png")
img.load()

print("->img loaded")

# mask = Image.open("C:/Users/user/SavedScreenMask.png")
# mask.load()

# print("->mask loaded")

answers2 = stability_api.generate(
    prompt=[generation.Prompt(text=sys.argv[1]+", space station interior, pixel art, retro, snes, high resolution, 4k wallpaper, hd, 8k",parameters=generation.PromptParameters(weight=1)), 
    generation.Prompt(text="people, robots, creatures, humans, human, robot, android, watermark, character, author",parameters=generation.PromptParameters(weight=-1))],
    init_image=img,
    #mask_image=mask,
    start_schedule=0.6,
    seed=randint(111111111,999999999),
    steps=15,
    cfg_scale=17.0,
    width=1024,
    height=512,
    sampler=generation.SAMPLER_K_EULER_ANCESTRAL
)

print("->img generated")

# Set up our warning to print to the console if the adult content classifier is tripped.
# If adult content classifier is not tripped, display generated image.
for resp in answers2:
    for artifact in resp.artifacts:
        if artifact.finish_reason == generation.FILTER:
            warnings.warn(
                "Your request activated the API's safety filters and could not be processed."
                "Please modify the prompt and try again.")
        if artifact.type == generation.ARTIFACT_IMAGE:
            global img2
            img2 = Image.open(io.BytesIO(artifact.binary))
            img2.save("~/Documents/RovotsINC/SavedScreen-img2img.png")

print("->img saved")

r = requests.post(
    "https://api.deepai.org/api/torch-srgan",
    files={
        'image': open("~/Documents/RovotsINC/SavedScreen-img2img.png", 'rb'),
    },
    headers={'api-key': 'f8ed42bc-459f-49ec-ab3a-3c0887a95af2'}
)
response = requests.get(r.json()['output_url'])
open("~/Documents/RovotsINC/SavedScreen-img2img.png", "wb").write(response.content)

print("->img upscaled")

print("->finished")