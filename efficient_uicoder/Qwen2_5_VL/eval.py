import os
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from types import MethodType
from os.path import join as pjoin

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
uied_root = os.path.join(project_root, "UIED")
sys.path.insert(0, uied_root)

def set_seed(seed=2026):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(2026)

# ------------------- Prompt -------------------
prompt = """You are an expert web developer who specializes in HTML and CSS.
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

# ------------------- Import Qwen -------------------
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration as Qwen2_5_VLForConditionalGeneration_end2end

from .dense_patch_labels import extract_dense_patch_labels, visualize_dense_patch_labels
from .modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration_EfficientUICoder
from .sampling import new_sample
from ..convert import clean_generated_html


def generate_html_from_screenshot(args):
    if args.exp_mode == 0:
        Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration_end2end
    else:
        Qwen2_5_VLForConditionalGeneration = Qwen2_5_VLForConditionalGeneration_EfficientUICoder

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    os.makedirs(args.result_dir, exist_ok=True)
    png_files = [f for f in os.listdir(args.eval_dir) if f.endswith('.png')]
    png_files_full_path = [os.path.join(args.eval_dir, f) for f in png_files]

    for image_file in tqdm(png_files_full_path, total=len(png_files_full_path), desc="Processing Images"):
        name = os.path.splitext(os.path.basename(image_file))[0]
        img = Image.open(image_file)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_file},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        if args.exp_mode == 1:
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

            dense_patch_labels = extract_dense_patch_labels(img, components=components)
            keep_indices = torch.tensor(dense_patch_labels)
            # Optional: uncomment to visualize the selected tokens (dense patch labels)
            # visualize_dense_patch_labels(image_inputs[0], dense_patch_labels, save_dir=keep_indices_dir)

        inputs = processor(
            text=[txt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        if args.exp_mode == 1:
            inputs["keep_indices"] = keep_indices.to(model.device)
            model.tokenizer = tokenizer
            model.decay_factor = args.decay_factor
            model.penalty_step = args.penalty_step
            model._sample = MethodType(new_sample, model)

        inputs = inputs.to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        ans = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        ans = clean_generated_html(ans)

        with open(os.path.join(args.result_dir, f"{name}.html"), "w") as fs:
            fs.write(ans)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model")
    parser.add_argument("--eval-dir", type=str, required=True, help="Directory containing original screenshots")
    parser.add_argument("--result-dir", type=str, required=True, help="Directory to save generated HTML results")
    parser.add_argument("--keep-indices-dir", type=str, required=True, help="Directory to save intermediate UIED outputs")
    parser.add_argument("--exp-mode", type=int, default=1, choices=[0, 1], help="0=end-to-end, 1=EfficientUICoder")
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--decay-factor", type=float, default=0.5)
    parser.add_argument("--penalty-step", type=int, default=1)
    args = parser.parse_args()

    generate_html_from_screenshot(args)