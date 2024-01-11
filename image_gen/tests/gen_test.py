import os
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image
from torchvision.transforms import v2


############################# STABLE UNCLIP
#os.environ['HF_HOME'] = '/home/jc98685/hf_cache' # MIDI Boxes
#
#img_txt_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
#)
#img_txt_pipe.to("cuda:1")
#
#cond_img = load_image("bufo.jpeg")
#prompt = "worried frog"
#gen_image = img_txt_pipe(cond_img, prompt, dropout=True).images[0] 
#gen_image.save("gen_bufo.jpg")


############################## CUTMIX/MIXUP 
cutmix = v2.CutMix(num_classes=1)
mixup = v2.MixUp(num_classes=1)
preprocess = v2.Compose([
    v2.PILToTensor(), 
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

img_1 = preprocess(load_image("/mnt/zhang-nas/jiahuic/diffusers/image_gen/tests/bufo.jpeg"))
img_2 = preprocess(load_image("/mnt/zhang-nas/jiahuic/diffusers/image_gen/tests/gen_bufo.jpg"))
dummy_images = torch.stack((img_1, img_2))
dummy_labels = torch.zeros(size=(2,)).to(torch.int64)
cutmixed_img, _ = cutmix(dummy_images, dummy_labels)
mixuped_img, _ = mixup(dummy_images, dummy_labels)

# saving just to see what they look like... 
cutmixed_img = v2.functional.to_pil_image(cutmixed_img[0])
mixuped_img = v2.functional.to_pil_image(mixuped_img[0])
cutmixed_img.save("cutmixed.jpg")
mixuped_img.save("mixuped.jpg")