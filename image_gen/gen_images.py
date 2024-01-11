import os 
import random
import wandb
import torch
from torchvision.transforms import v2
from accelerate import PartialState
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

############################################## SETUP
os.environ['HF_HOME'] = '/home/jc98685/hf_cache' # MIDI Boxes

wandb.init(
    project="StableUnclipImageGen",
    group="accelerate"
)

PROMPT_FILE = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/imagenet_lt_balance_counts_magpie.txt"
TRAIN_DATA_TXT = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/ImageNet_LT_train.txt"
TRAIN_DATA_ROOT = "/mnt/zhang-nas/tensorflow_datasets/downloads/manual/imagenet2012"
OUTPUT_DIR = "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/rand_img_cond"

img_txt_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
)
distributed_state = PartialState()
img_txt_pipe.to(distributed_state.device)

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


############################################################ METHODS
# randomly select one image of given class 
def get_cond_img(class_label):
    train_imgs = TRAIN_DICT[class_label]
    img_path = random.choice(train_imgs)
    img = load_image(os.path.join(TRAIN_DATA_ROOT, img_path))
    return img_path

# randomly select one train data image per class to condition generative model on
def rand_img_cond(dropout=False):
    # for each class, generate synthetic images with text label as prompt and randomly selected class image 
    with open(PROMPT_FILE) as gen_file:
        # each line of this file contains the label (text label is the prompt) and how many images need to be generated
        lines = [line.rstrip('\n') for line in gen_file]

    for line in lines:
        l = line.split("\"")
        int_label = int(l[0].strip()); txt_label = l[1].strip("\""); gen_count = int(l[2].strip())

        wandb.log({"label": int_label})
        # create dict to pass into distributed inference of {prompts: [], cond_imgs: []}
        total_prompts = [txt_label] * gen_count
        all_cond_imgs = [get_cond_img(int_label) for i in range(gen_count)]
        all_indices = [i for i in range(gen_count)]
        prompt_img_dict = {"prompts": total_prompts, "cond_imgs": all_cond_imgs, "indices": all_indices}
        with distributed_state.split_between_processes(prompt_img_dict) as prompt_imgs:
            # within each process, get all the prompts, images to condition on, and indices -- then generate image
            prompts = prompt_imgs["prompts"]; cond_imgs = prompt_imgs["cond_imgs"]; indices = prompt_imgs["indices"] 
            for i in range(len(prompts)):
                print(f"Generating image {indices[i]} of {int_label}: \"{prompts[i]}\"")
                gen_image = img_txt_pipe(cond_imgs[i], prompts[i], dropout=dropout).images[0] 
                gen_img_name = f"{int_label}_{indices[i]}.jpg"
                gen_image.save(os.path.join(OUTPUT_DIR, gen_img_name))