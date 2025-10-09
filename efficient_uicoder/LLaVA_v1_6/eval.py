import os
import sys
import re
import random
import argparse
import warnings
from io import BytesIO
import torch
import numpy as np
from PIL import Image
import requests
from tqdm import tqdm
from os.path import join as pjoin

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
uied_root = os.path.join(project_root, "UIED")
sys.path.insert(0, uied_root)
sys.path.insert(0, project_root)

warnings.filterwarnings('ignore')
def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2026)

# ------------------- Prompt -------------------
query = """You are an expert web developer who specializes in HTML and CSS.
    A user will provide you with a screenshot of a webpage.
    You need to return a single html file that uses HTML and CSS to reproduce the given website.
    Include all CSS code in the HTML file itself.
    If the webpage contains any images, use a placeholder image from https://placehold.co.
    You must determine the appropriate width and height of the placeholder based on the visual size of the image in the screenshot.
    Format the image URLs like this: https://placehold.co/600x400/EEEEEE/EEEEEE, adjusting the dimensions accordingly.
    If any images in the screenshot are replaced with a gray rectangle, treat them as image placeholders and apply the same https://placehold.co rule.
    Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
    Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
    Respond with the content of the HTML+CSS file:"""

# ------------------- Import UIED -------------------
import detect_text.text_detection as text
import detect_compo.ip_region_proposal as ip
import detect_merge.merge as merge

# ------------------- Import LLaVA -------------------
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_anyres_image,
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

from .dense_patch_labels import extract_dense_patch_labels, visualize_dense_patch_labels
from .main import EfficientUICoder
from .sampling import my_greedy_search
from ..convert import clean_generated_html

# ------------------- Utility Functions -------------------
def load_image(image_file):
    if image_file.startswith("http"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def load_images(image_files):
    return [load_image(image_file) for image_file in image_files]


def generate_html_from_screenshot(args):
    exp_mode = args.exp_mode
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=model_name
    )
    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if model.config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if model.config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(
            "[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(
                conv_mode, args.conv_mode, args.conv_mode
            )
        )
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    eval_dir = args.eval_dir
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)
    png_files = [f for f in os.listdir(eval_dir) if f.endswith('.png')]
    png_files_full_path = [os.path.join(eval_dir, f) for f in png_files]

    model.eval()
    for image_file in tqdm(png_files_full_path, total=len(png_files_full_path), desc="Processing Images"):
        name = image_file.split("/")[-1].split(".")[0]
        image = load_image(image_file)

        if exp_mode == 1:
            keep_indices_dir = os.path.join(args.keep_indices_dir, name)
            os.makedirs(keep_indices_dir, exist_ok=True)

            os.makedirs(pjoin(keep_indices_dir, 'ocr'), exist_ok=True)
            text.text_detection(image_file, keep_indices_dir, show=False, method='easyocr')
            
            os.makedirs(pjoin(keep_indices_dir, 'ip'), exist_ok=True)
            classifier = None
            key_params = {'min-grad':3, 'ffl-block':5, 'min-ele-area':5,
                        'merge-contained-ele':True, 'merge-line-to-paragraph':True, 'remove-bar':True}
            ip.compo_detection(image_file, keep_indices_dir, key_params,
                               classifier=classifier, resize_by_height=None, show=False)
            
            os.makedirs(pjoin(keep_indices_dir, 'merge'), exist_ok=True)
            compo_path = pjoin(keep_indices_dir, 'ip', f'{name}.json')
            ocr_path = pjoin(keep_indices_dir, 'ocr', f'{name}.json')
            _, components = merge.merge_EfficientUI(image_file, compo_path, ocr_path, pjoin(keep_indices_dir, 'merge'))
            
            dense_labels, images = extract_dense_patch_labels(
                image, image_processor, model.config.image_grid_pinpoints, components
            )
            # Optional: uncomment to visualize the selected tokens (dense patch labels)
            # visualize_dense_patch_labels(images, dense_labels, save_dir=keep_indices_dir)
            model = EfficientUICoder(model, dense_labels=dense_labels, token_selection_ratio=0.1)

        image_files = image_file.split(args.sep)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]

        image_tensor = process_images(
            images,
            image_processor,
            model.config
        ).to(model.device, dtype=torch.float16)
        input_ids = (
            tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
                .unsqueeze(0)
                .cuda()
        )
        if exp_mode == 1:
            model.tokenizer = tokenizer
            model.decay_factor = args.decay_factor
            model.penalty_step = args.penalty_step
            from transformers.generation.utils import GenerationMixin
            GenerationMixin.greedy_search = my_greedy_search
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
            )
        output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        output = clean_generated_html(output)
        with open(os.path.join(result_dir, f"{name}.html"), "w") as fs:
            fs.write(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default="llava_v1", choices=["llava_v1", "chatml_direct"],
                        help="'llava_v1' is used for LLaVA-v1.6-7B/13B, 'chatml_direct' is used for LLaVA-v1.6-34B")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--eval-dir", type=str, required=True, help="Directory containing original screenshots")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory to save generated HTML results")
    parser.add_argument("--keep-indices-dir", type=str, required=True, help="Directory to save intermediate UIED outputs")
    parser.add_argument("--exp-mode", type=int, default=1, choices=[0, 1], help="0=end-to-end, 1=EfficientUICoder")
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--penalty-step", type=int, default=3)

    args = parser.parse_args()

    generate_html_from_screenshot(args)