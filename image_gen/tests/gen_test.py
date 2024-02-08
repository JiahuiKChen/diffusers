import os
# os.environ['HF_HOME'] = '/mnt/zhang-nas/jiahuic/hf_cache' # MIDI Boxes
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache' # CS A40 box

import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
from torchvision.transforms import v2


############################# STABLE UNCLIP
img_txt_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
   "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
)
img_txt_pipe.to("cuda:0")
#
#cond_img = load_image("bufo.jpeg")
#prompt = "worried frog"
#gen_image = img_txt_pipe(cond_img, prompt, dropout=True).images[0] 
#gen_image.save("gen_bufo.jpg")


############################## CUTMIX/MIXUP 
# cutmix = v2.CutMix(num_classes=1)
# mixup = v2.MixUp(num_classes=1)
# preprocess = v2.Compose([
#     v2.PILToTensor(), 
#     v2.RandomResizedCrop(size=(224, 224), antialias=True),
#     v2.ToDtype(torch.float32, scale=True)
# ])

# img_1 = preprocess(load_image("/mnt/zhang-nas/jiahuic/diffusers/image_gen/tests/bufo.jpeg"))
# img_2 = preprocess(load_image("/mnt/zhang-nas/jiahuic/diffusers/image_gen/tests/gen_bufo.jpg"))
# dummy_images = torch.stack((img_1, img_2))
# dummy_labels = torch.zeros(size=(2,)).to(torch.int64)
# cutmixed_img, _ = cutmix(dummy_images, dummy_labels)
# mixuped_img, _ = mixup(dummy_images, dummy_labels)

# # saving just to see what they look like... 
# cutmixed_img = v2.functional.to_pil_image(cutmixed_img[0])
# mixuped_img = v2.functional.to_pil_image(mixuped_img[0])
# cutmixed_img.save("cutmixed.jpg")
# mixuped_img.save("mixuped.jpg")


############################## EMBEDDING SPACE CUTMIX/MIXUP 
img_1 = load_image("/mnt/zhang-nas/jiahuic/diffusers/image_gen/tests/bufo.jpeg")
img_2 = load_image("/mnt/zhang-nas/jiahuic/diffusers/image_gen/tests/gen_bufo.jpg")

# embedding mixup
gen_image = img_txt_pipe(
    img_2, 
    "worried frog", 
    num_images_per_prompt=4,
    dropout=False, 
    embed_cutmix=False, 
    embed_mixup=True,
    img_1 = img_1,
    img_2 = img_2
).images
for i in range(len(gen_image)):
    img = gen_image[i]
    img.save(f"embed_mixup_gen_test_{i}.jpg")

# embedding cutmix 
gen_image = img_txt_pipe(
    img_2, 
    "worried frog", 
    num_images_per_prompt=1,
    dropout=False, 
    embed_cutmix=True, 
    embed_mixup=False,
    img_1 = img_1,
    img_2 = img_2
).images[0]