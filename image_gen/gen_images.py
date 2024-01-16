import os 
# os.environ['HF_HOME'] = '/home/jc98685/hf_cache' # MIDI Boxes
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache' # CS A40 box

import random
import wandb
import torch
from torchvision.transforms import v2
from accelerate import PartialState
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

##################################################### SETUP
wandb.init(
    project="StableUnclipImageGen",
    group="a40-dropout-test"
)

# MIDI BOXES
# PROMPT_FILE = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/imagenet_lt_balance_counts.txt"
# TRAIN_DATA_TXT = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/ImageNet_LT_train.txt"
# TRAIN_DATA_ROOT = "/mnt/zhang-nas/tensorflow_datasets/downloads/manual/imagenet2012"
# OUTPUT_DIR = "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/cutmix"
# testing output dir
# OUTPUT_DIR = "/mnt/zhang-nas/jiahuic/synth_LT_data/test"

# A40 
PROMPT_FILE = "/datastor1/jiahuikchen/diffusers/image_gen/imagenet_lt_balance_counts.txt"
TRAIN_DATA_TXT = "/datastor1/jiahuikchen/diffusers/image_gen/ImageNet_LT_train.txt"
TRAIN_DATA_ROOT = "/datastor1/imagenet2012_manual"
OUTPUT_DIR = "/datastor1/jiahuikchen/dropout"

# cutmix/mixup
cutmix = v2.CutMix(num_classes=1)
mixup = v2.MixUp(num_classes=1)
preprocess = v2.Compose([
    v2.PILToTensor(), 
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True)
])

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
def get_rand_img(class_label):
    train_imgs = TRAIN_DICT[class_label]
    img_path = random.choice(train_imgs)
    img = load_image(os.path.join(TRAIN_DATA_ROOT, img_path))
    return img

# randomly select 2 images from given class,
# perform cutmix on them and return the cutmixed image
def cutmix_or_mixup(class_label, use_cutmix=True, use_mixup=False):
    img_1 = preprocess(get_rand_img(class_label))
    img_2 = preprocess(get_rand_img(class_label))
    dummy_images = torch.stack((img_1, img_2))
    dummy_labels = torch.zeros(size=(2,)).to(torch.int64)
    cond_img = None
    if use_cutmix:
        cutmixed_img, _ = cutmix(dummy_images, dummy_labels)
        cond_img = cutmixed_img[0]
    elif use_mixup:
        mixuped_img, _ = mixup(dummy_images, dummy_labels)
        cond_img = mixuped_img[0]
    return v2.functional.to_pil_image(cond_img)


# randomly select one train data image per class to condition generative model on
def gen_imgs(dropout=False, use_cutmix=False, use_mixup=False):
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
        all_indices = [i for i in range(gen_count)]
        # image conditioning based on what's specified
        if use_cutmix:
            all_cond_imgs = [cutmix_or_mixup(int_label, use_cutmix=True, use_mixup=False) for i in range(gen_count)]
        elif use_mixup:
            all_cond_imgs = [cutmix_or_mixup(int_label, use_cutmix=False, use_mixup=True) for i in range(gen_count)]
        else:
            # randomly selecting an image from the same training class to generate conditioning image
            all_cond_imgs = [get_rand_img(int_label) for i in range(gen_count)]
        prompt_img_dict = {"prompts": total_prompts, "cond_imgs": all_cond_imgs, "indices": all_indices}
        
        with distributed_state.split_between_processes(prompt_img_dict) as prompt_imgs:
            # within each process, get all the prompts, images to condition on, and indices -- then generate image
            prompts = prompt_imgs["prompts"]; cond_imgs = prompt_imgs["cond_imgs"]; indices = prompt_imgs["indices"] 
            for i in range(len(prompts)):
                print(f"Generating image {indices[i]} of {int_label}: \"{prompts[i]}\"")
                gen_image = img_txt_pipe(cond_imgs[i], prompts[i], dropout=dropout).images[0] 
                gen_img_name = f"{int_label}_{indices[i]}.jpg"
                print(f"would save to {os.path.join(OUTPUT_DIR, gen_img_name)}")
                # gen_image.save(os.path.join(OUTPUT_DIR, gen_img_name))

# Gen images conditioned on mixup-ed random pairs with the same class 
# gen_imgs(dropout=False, use_cutmix=False, use_mixup=True)
                
# Gen images conditioned on cutmix-ed random paris with same class
# gen_imgs(dropout=False, use_cutmix=True, use_mixup=False)
                
# Gen images conditioned on randomly selected images with same class, with dropout applied
gen_imgs(dropout=True, use_cutmix=False, use_mixup=False)