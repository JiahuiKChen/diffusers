import os 
# os.environ['HF_HOME'] = '/mnt/zhang-nas/jiahuic/hf_cache' # NAS
# os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache' # datastor1

import random
import wandb
import torch
from torchvision.transforms import v2
from accelerate import PartialState
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

##################################################### SETUP

# NAS 
# PROMPT_FILE = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/imagenet_lt_balance_counts_589.txt"
# TRAIN_DATA_TXT = "/mnt/zhang-nas/jiahuic/diffusers/image_gen/ImageNet_LT_train.txt"
# TRAIN_DATA_ROOT = "/mnt/zhang-nas/tensorflow_datasets/downloads/manual/imagenet2012"
# OUTPUT_DIR = "/mnt/zhang-nas/jiahuic/synth_LT_data/ImageNetLT/dropout"
# testing output dir
# OUTPUT_DIR = "/mnt/zhang-nas/jiahuic/synth_LT_data/test"

# /datastor1 drive 
# PROMPT_FILE = "/datastor1/jiahuikchen/diffusers/image_gen/imagenet_lt_balance_counts_no90_391.txt"
# TRAIN_DATA_TXT = "/datastor1/jiahuikchen/diffusers/image_gen/ImageNet_LT_train.txt"
# TRAIN_DATA_ROOT = "/datastor1/imagenet2012_manual"
# OUTPUT_DIR = "/datastor1/jiahuikchen/synth_ImageNet/embed_mixup_dropout/"

# A100
PROMPT_FILE = "/home/karen/diffusers/image_gen/imagenet_lt_balance_counts_90.txt"
TRAIN_DATA_TXT = "/home/karen/diffusers/image_gen/ImageNet_LT_train.txt"
TRAIN_DATA_ROOT = "/home/karen/imagenet_raw"

CFG = 7.0
COND_METHOD = "embed_cutmix_dropout"
OUTPUT_DIR = f"/home/karen/synth_data/{COND_METHOD}90_{CFG}cfg/"

wandb.init(
    project="StableUnclipImageGen",
    group=OUTPUT_DIR.split('/')[-1],
    config={
        "GEN_IMG_OUT_DIR": OUTPUT_DIR,
        "CFG_Scale": CFG,
        "CONDITIONING_METHOD": COND_METHOD,
        "PROMPT_FILE": PROMPT_FILE
    }
)

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
# randomly select n image(s) of given class, defaults to 1
def get_rand_img(class_label, n=1):
    train_imgs = TRAIN_DICT[class_label]
    img_paths = random.sample(train_imgs, k=n)
    imgs = []
    for img_path in img_paths:
        img = load_image(os.path.join(TRAIN_DATA_ROOT, img_path))
        imgs.append(img)
    if len(imgs) == 1:
        return imgs[0]
    return imgs

# randomly select 2 images from given class,
# perform cutmix on them and return the cutmixed image
def cutmix_or_mixup(class_label, use_cutmix=True, use_mixup=False):
    imgs = get_rand_img(class_label=class_label, n=2)
    img_1 = preprocess(imgs[0])
    img_2 = preprocess(imgs[1])
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


def gen_imgs(dropout=False, 
             use_cutmix=False, 
             use_mixup=False, 
             use_embed_mixup=False, 
             use_embed_cutmix=False,
             guidance_scale=7.0,
             ):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PROMPT_FILE) as gen_file:
        # each line of this file contains the label (text label is the prompt) and how many images need to be generated
        lines = [line.rstrip('\n') for line in gen_file]

    for line in lines:
        l = line.split("\"")
        int_label = int(l[0].strip()); txt_label = l[1].strip("\""); gen_count = int(l[2].strip())

        wandb.log({"label": int_label})

        # create dict to pass into distributed inference of {prompts: [], cond_imgs: [], indices: []}
        total_prompts = [txt_label] * gen_count
        all_indices = [i for i in range(gen_count)]
        # image conditioning based on what's specified
        if use_cutmix:
            all_cond_imgs = [cutmix_or_mixup(int_label, use_cutmix=True, use_mixup=False) for i in range(gen_count)]
        elif use_mixup:
            all_cond_imgs = [cutmix_or_mixup(int_label, use_cutmix=False, use_mixup=True) for i in range(gen_count)]
        elif use_embed_mixup or use_embed_cutmix:
            # if doing embedding level mixup or cutmix, pass in tuples of 2 images 
            # that the model will encode and do mixup or cutmix on the CLIP image embeddings
            all_cond_imgs = []
            for _ in range(gen_count):
                imgs = get_rand_img(class_label=int_label, n=2)
                all_cond_imgs.append((imgs[0], imgs[1])) 
        else:
            # randomly selecting an image from the same training class to generate conditioning image
            all_cond_imgs = [get_rand_img(int_label) for i in range(gen_count)]
        prompt_img_dict = {"prompts": total_prompts, "cond_imgs": all_cond_imgs, "indices": all_indices}
        
        with distributed_state.split_between_processes(prompt_img_dict) as prompt_imgs:
            # within each process, get all the prompts, images to condition on, and indices -- then generate image
            prompts = prompt_imgs["prompts"]; cond_imgs = prompt_imgs["cond_imgs"]; indices = prompt_imgs["indices"] 
            for i in range(len(prompts)):
                print(f"Generating image {indices[i]} of {int_label}: \"{prompts[i]}\"")
                if not use_embed_cutmix and not use_embed_mixup:
                    gen_image = img_txt_pipe(cond_imgs[i], prompts[i], dropout=dropout, guidance_scale=guidance_scale).images[0] 
                else:
                    gen_image = img_txt_pipe(cond_imgs[i][0], 
                                                prompts[i], 
                                                guidance_scale=guidance_scale,
                                                dropout=dropout,
                                                embed_cutmix=use_embed_cutmix,
                                                embed_mixup=use_embed_mixup,
                                                img_1=cond_imgs[i][0],
                                                img_2=cond_imgs[i][1]
                                             ).images[0]  
                gen_img_name = f"{int_label}_{indices[i]}.jpg"
                gen_image.save(os.path.join(OUTPUT_DIR, gen_img_name))
    
    print(f"FINISHED -- IMAGES SAVED TO: {OUTPUT_DIR}")


# Given downsampled "many" class training txt file and balance counts,  
# generate images for these classes using ALL conditioning methods 
MULTI_OUTPUT_DIR_ROOT = "/datastor1/jiahuikchen/synth_ImageNet/30_many_to_median"
def gen_imgs_all_cond():
    with open(PROMPT_FILE) as gen_file:
        # each line of this file contains the label (text label is the prompt) and how many images need to be generated
        lines = [line.rstrip('\n') for line in gen_file]

    for line in lines:
        l = line.split("\"")
        int_label = int(l[0].strip()); txt_label = l[1].strip("\""); gen_count = int(l[2].strip())

        wandb.log({"label": int_label})

        # create dict to pass into distributed inference of {prompts: [], cond_imgs: [] (6 dicts, 1 for each method), indices: []}
        total_prompts = [txt_label] * gen_count
        all_indices = [i for i in range(gen_count)]
        # cutmix conditioning images
        cutmix_cond_imgs = [cutmix_or_mixup(int_label, use_cutmix=True, use_mixup=False) for i in range(gen_count)]
        # mixup conditioning images
        mixup_cond_imgs = [cutmix_or_mixup(int_label, use_cutmix=False, use_mixup=True) for i in range(gen_count)]
        # emebed-cutmix (tuples of 2 images the model will encode and do mixup or cutmix on the CLIP image embeddings)
        embed_cutmix_cond_imgs = []
        for _ in range(gen_count):
            imgs = get_rand_img(class_label=int_label, n=2)
            embed_cutmix_cond_imgs.append((imgs[0], imgs[1])) 
        # emebed-mixup (tuples of 2 images the model will encode and do mixup or cutmix on the CLIP image embeddings)
        embed_mixup_cond_imgs = [] 
        for _ in range(gen_count):
            imgs = get_rand_img(class_label=int_label, n=2)
            embed_mixup_cond_imgs.append((imgs[0], imgs[1])) 
        # random training of same class as conditioning image (used for dropout and rand_img_cond)
        rand_cond_imgs = [get_rand_img(int_label) for i in range(gen_count)]
        # distributed inference dict with ALL conditioning images for ALL methods
        prompt_img_dict = {"prompts": total_prompts, 
                           "cutmix_cond_imgs": cutmix_cond_imgs,
                           "mixup_cond_imgs": mixup_cond_imgs,
                           "embed_cutmix_cond_imgs": embed_cutmix_cond_imgs,
                           "embed_mixup_cond_imgs": embed_mixup_cond_imgs, 
                           "rand_cond_imgs": rand_cond_imgs, 
                           "indices": all_indices
                        }
        
        with distributed_state.split_between_processes(prompt_img_dict) as prompt_imgs:
            # within each process, get all the prompts, images to condition on, and indices -- then generate image
            prompts = prompt_imgs["prompts"]; indices = prompt_imgs["indices"] 
            cutmix_cond_imgs = prompt_imgs["cutmix_cond_imgs"]; mixup_cond_imgs = prompt_imgs["mixup_cond_imgs"]; rand_cond_imgs = prompt_imgs["rand_cond_imgs"] 
            embed_cutmix_cond_imgs = prompt_imgs["embed_cutmix_cond_imgs"]; embed_mixup_cond_imgs = prompt_imgs["embed_mixup_cond_imgs"]
            print(f"Generating images for {int_label}: {txt_label}")
            for i in range(len(prompts)):
                gen_img_name = f"{int_label}_{indices[i]}.jpg"
                # rand_img_cond
                gen_image = img_txt_pipe(rand_cond_imgs[i], prompts[i], dropout=False).images[0] 
                gen_image.save(os.path.join(MULTI_OUTPUT_DIR_ROOT, "rand_img_cond", gen_img_name))
                # dropout
                gen_image.save(os.path.join(MULTI_OUTPUT_DIR_ROOT, "dropout", gen_img_name))
                gen_image = img_txt_pipe(rand_cond_imgs[i], prompts[i], dropout=True).images[0] 
                # cutmix
                gen_image = img_txt_pipe(cutmix_cond_imgs[i], prompts[i], dropout=False).images[0] 
                gen_image.save(os.path.join(MULTI_OUTPUT_DIR_ROOT, "cutmix", gen_img_name))
                # embed_cutmix
                gen_image = img_txt_pipe(embed_cutmix_cond_imgs[i][0], 
                                            prompts[i], 
                                            dropout=False,
                                            embed_cutmix=True,
                                            embed_mixup=False,
                                            img_1=embed_cutmix_cond_imgs[i][0],
                                            img_2=embed_cutmix_cond_imgs[i][1]
                                            ).images[0]  
                gen_image.save(os.path.join(MULTI_OUTPUT_DIR_ROOT, "embed_cutmix", gen_img_name))
                # mixup
                gen_image = img_txt_pipe(mixup_cond_imgs[i], prompts[i], dropout=False).images[0] 
                gen_image.save(os.path.join(MULTI_OUTPUT_DIR_ROOT, "mixup", gen_img_name)) 
                # embed_mixup
                gen_image = img_txt_pipe(embed_mixup_cond_imgs[i][0], 
                                            prompts[i], 
                                            dropout=False,
                                            embed_cutmix=False,
                                            embed_mixup=True,
                                            img_1=embed_mixup_cond_imgs[i][0],
                                            img_2=embed_mixup_cond_imgs[i][1]
                                            ).images[0]  
                gen_image.save(os.path.join(MULTI_OUTPUT_DIR_ROOT, "embed_mixup", gen_img_name)) 

    print(f"FINISHED -- IMAGES SAVED TO: {MULTI_OUTPUT_DIR_ROOT}")

############################################################ RUNS

### INDIVIDUAL METHODS
# rand_img_cond 
# gen_imgs(dropout=False, use_cutmix=False, use_mixup=False)

# mixup
# gen_imgs(dropout=False, use_cutmix=False, use_mixup=True)
                
# mixup-dropout 
# gen_imgs(dropout=True, use_cutmix=False, use_mixup=True)
                
# cutmix
# gen_imgs(dropout=False, use_cutmix=True, use_mixup=False)
                
# cutmix-dropout 
# gen_imgs(dropout=True, use_cutmix=True, use_mixup=False)

# dropout
# gen_imgs(dropout=True, use_cutmix=False, use_mixup=False)
                
# embedding-space cutmix
# gen_imgs(dropout=False, use_cutmix=False, use_mixup=False, use_embed_mixup=False, use_embed_cutmix=True)    

# embedding-space cutmix dropout with more steps
# gen_imgs(guidance_scale=7.0, dropout=False, use_cutmix=False, use_mixup=False, use_embed_mixup=False, use_embed_cutmix=True)
                
# embedding-space mixup  
# gen_imgs(dropout=False, use_cutmix=False, use_mixup=False, use_embed_mixup=True, use_embed_cutmix=False)             
                
# embed-mixup-dropout  
# gen_imgs(dropout=True, use_cutmix=False, use_mixup=False, use_embed_mixup=True, use_embed_cutmix=False)             

# embed-cutmix-dropout
# gen_imgs(dropout=True, use_cutmix=False, use_mixup=False, use_embed_mixup=False, use_embed_cutmix=True)    

### ALL METHODS (downsampled classees)
# gen_imgs_all_cond()

if COND_METHOD == "embed_cutmix_dropout":
    # embedding-space cutmix dropout 
    gen_imgs(guidance_scale=CFG, dropout=True, use_cutmix=False, use_mixup=False, use_embed_mixup=False, use_embed_cutmix=True)