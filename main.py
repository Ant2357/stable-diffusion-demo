import os
import re

from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline

load_dotenv()

access_token = os.environ["ACCESS_TOKEN"]
model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=access_token)
model.to("cpu")

prompt = "Word"

output_filename = re.sub(r'[\\/:*?"<>|,]+', "", prompt).replace(" ","_").lower()
image = model(prompt, num_inference_steps=100)["sample"][0]
image.save(f'output/{output_filename}.png')
