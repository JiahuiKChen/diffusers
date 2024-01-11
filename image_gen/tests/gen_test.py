import os
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

os.environ['HF_HOME'] = '/home/jc98685/hf_cache' # MIDI Boxes

img_txt_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
)
img_txt_pipe.to("cuda:1")

cond_img = load_image("bufo.jpeg")
prompt = "worried frog"
gen_image = img_txt_pipe(cond_img, prompt, dropout=True).images[0] 
gen_image.save("gen_bufo.jpg")