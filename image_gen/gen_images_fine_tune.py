import os 
# os.environ['HF_HOME'] = '/mnt/zhang-nas/jiahuic/hf_cache' # NAS
os.environ['HF_HOME'] = '/datastor1/jiahuikchen/hf_cache' # datastor1

import random
import wandb
import torch
from torchvision.transforms import v2
from accelerate import PartialState
from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

from datasets.coco import COCODataset
from datasets.pascal import PASCALDataset
from datasets.caltetch101 import CalTech101Dataset
from datasets.flowers102 import Flowers102Dataset

DATASET = "COCO"
GEN_IMG_OUT_DIR = f"/datastor1/jiahuikchen/synth_fine_tune/{DATASET}_synth_imgs"
if not os.path.exists(GEN_IMG_OUT_DIR):
    os.makedirs(GEN_IMG_OUT_DIR)

wandb.init(
    project="StableUnclipImageGen",
    group=f"{DATASET}",
    config={
        "DATASET": DATASET,
        "GEN_IMG_OUT_DIR": GEN_IMG_OUT_DIR
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

if DATASET == "COCO":
    dataset = COCODataset()
elif DATASET == "pascal":
    dataset = PASCALDataset()
elif DATASET == "caltech":
    dataset = CalTech101Dataset()
elif DATASET == "flowers":
    dataset = Flowers102Dataset()
else:
    raise ValueError("not a valid few-shot dataset")


# randomly select n image(s) of given class, defaults to 1
def get_rand_img(class_label, n=1):
    train_imgs = dataset.class_to_images[class_label]
    img_paths = random.sample(train_imgs, k=n)
    imgs = []
    for img_path in img_paths:
        img = load_image(img_path)
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

# Generate 16 images for each class with every conditioning method
# images are saved in subdirs corresponding to conditioning methods
def gen_imgs_all_cond():
    gen_count = 16
    for c in dataset.class_names:
        wandb.log({"label": c})
        prompt = f"a photo of a {c}"

        # create dict to pass into distributed inference of {prompts: [], cond_imgs: [] (6 dicts, 1 for each method), indices: []}
        total_prompts = [prompt] * gen_count
        all_indices = [i for i in range(gen_count)]
        # cutmix conditioning images
        cutmix_cond_imgs = [cutmix_or_mixup(c, use_cutmix=True, use_mixup=False) for i in range(gen_count)]
        # mixup conditioning images
        mixup_cond_imgs = [cutmix_or_mixup(c, use_cutmix=False, use_mixup=True) for i in range(gen_count)]
        # emebed-cutmix (tuples of 2 images the model will encode and do mixup or cutmix on the CLIP image embeddings)
        embed_cutmix_cond_imgs = []
        for _ in range(gen_count):
            imgs = get_rand_img(class_label=c, n=2)
            embed_cutmix_cond_imgs.append((imgs[0], imgs[1])) 
        # emebed-mixup (tuples of 2 images the model will encode and do mixup or cutmix on the CLIP image embeddings)
        embed_mixup_cond_imgs = [] 
        for _ in range(gen_count):
            imgs = get_rand_img(class_label=c, n=2)
            embed_mixup_cond_imgs.append((imgs[0], imgs[1])) 
        # random training of same class as conditioning image (used for dropout and rand_img_cond)
        rand_cond_imgs = [get_rand_img(c) for _ in range(gen_count)]
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
            print(f"Generating images for {c}")
            for i in range(len(prompts)):
                gen_img_name = f"{c}_{indices[i]}.jpg"
                # rand_img_cond
                gen_image = img_txt_pipe(rand_cond_imgs[i], prompts[i], dropout=False).images[0] 
                gen_image.save(os.path.join(GEN_IMG_OUT_DIR, "rand_img_cond", gen_img_name))
                # dropout
                gen_image.save(os.path.join(GEN_IMG_OUT_DIR, "dropout", gen_img_name))
                gen_image = img_txt_pipe(rand_cond_imgs[i], prompts[i], dropout=True).images[0] 
                # cutmix
                gen_image = img_txt_pipe(cutmix_cond_imgs[i], prompts[i], dropout=False).images[0] 
                gen_image.save(os.path.join(GEN_IMG_OUT_DIR, "cutmix", gen_img_name))
                # embed_cutmix
                gen_image = img_txt_pipe(embed_cutmix_cond_imgs[i][0], 
                                            prompts[i], 
                                            dropout=False,
                                            embed_cutmix=True,
                                            embed_mixup=False,
                                            img_1=embed_cutmix_cond_imgs[i][0],
                                            img_2=embed_cutmix_cond_imgs[i][1]
                                            ).images[0]  
                gen_image.save(os.path.join(GEN_IMG_OUT_DIR, "embed_cutmix", gen_img_name))
                # mixup
                gen_image = img_txt_pipe(mixup_cond_imgs[i], prompts[i], dropout=False).images[0] 
                gen_image.save(os.path.join(GEN_IMG_OUT_DIR, "mixup", gen_img_name)) 
                # embed_mixup
                gen_image = img_txt_pipe(embed_mixup_cond_imgs[i][0], 
                                            prompts[i], 
                                            dropout=False,
                                            embed_cutmix=False,
                                            embed_mixup=True,
                                            img_1=embed_mixup_cond_imgs[i][0],
                                            img_2=embed_mixup_cond_imgs[i][1]
                                        ).images[0]  
                gen_image.save(os.path.join(GEN_IMG_OUT_DIR, "embed_mixup", gen_img_name)) 

    print(f"FINISHED -- IMAGES SAVED TO: {GEN_IMG_OUT_DIR}")

gen_imgs_all_cond()