import os 
import random
import wandb
import torch
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

os.environ['HF_HOME'] = '/home/jc98685/hf_cache' # MIDI Boxes

wandb.init(
    project="StableUnclipImageGen"
)

PROMPT_FILE = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/imagenet_lt_balance_counts.txt"
TRAIN_DATA_TXT = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/ImageNet_LT_train.txt"
TRAIN_DATA_ROOT = "/mnt/zhang-nas/tensorflow_datasets/downloads/manual/imagenet2012"
OUTPUT_DIR = "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT"

img_txt_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
)

img_txt_pipe = img_txt_pipe.to("cuda:0")

# read in long-tail training data file, 
# construct dict: {int class label: <list of image paths>}
# used to get training data images to condition generative model on
TRAIN_DICT = {}
with open(TRAIN_DATA_TXT) as train_file:
    for line in train_file:
        info = line.split() 
        class_label = int(info[1])
        img_path = info[0]

        if class_label in TRAIN_DICT:
            TRAIN_DICT[class_label].append(img_path)
        else:
            TRAIN_DICT[class_label] = [img_path]

# randomly select one image in a class to condition generative model on
def get_cond_img(class_label):
    train_imgs = TRAIN_DICT[class_label]
    img_path = random.choice(train_imgs)
    img = load_image(os.path.join(TRAIN_DATA_ROOT, img_path))
    return img


# for each class, generate synthetic images with text label as prompt and randomly selected class image 
gen_data_txt = ""
with open(PROMPT_FILE) as gen_file:
    # each line of this file contains the label (text label is the prompt) and how many images need to be generated
    for line in gen_file:
        l = line.split("\"")
        int_label = int(l[0].strip()); txt_label = l[1].strip("\""); gen_count = int(l[2].strip())

        wandb.log({"label": txt_label, "gen_count": gen_count})
        for i in range(gen_count, 0, -1):
              cond_img = get_cond_img(int_label)
              print(f"Generating image {i} for \"{txt_label}\"")
              gen_image = img_txt_pipe(cond_img, prompt=txt_label).images[0]
              gen_img_name = f"{int_label}_{i}.jpg"
              gen_image.save(os.path.join(OUTPUT_DIR, gen_img_name))
              # write to output file: <path to image> <class label (int)>
              gen_data_txt += f"{gen_img_name} {int_label}\n"