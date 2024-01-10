import wandb
from pipeline_stable_unclip_img2img import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

os.environ['HF_HOME'] = '/home/jc98685/hf_cache' # MIDI Boxes
IMAGENET_DATA_ROOT = "/mnt/zhang-nas/tensorflow_datasets/downloads/manual/imagenet2012/train"

img_txt_pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, 
)
img_txt_pipe.set_progress_bar_config(disable=True)

img_txt_pipe = img_txt_pipe.to("cuda:2")

with open("imagenet_lt_balance_counts.txt") as gen_file:
    for line in gen_file:
        l = line.split
        int_label = l[0]; txt_label = l[1]; gen_count = l[2]


# text and image generation
# prompts_and_imgs is list of tuples (prompt, image)
def gen_images_from_text_and_img(prompts_and_imgs):
    for tup in prompts_and_imgs:
        prompt = tup[0]
        img = tup[1]
        images = img_txt_pipe(img, prompt=prompt, num_images_per_prompt=4).images