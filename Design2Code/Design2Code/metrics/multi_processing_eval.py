from Design2Code.metrics.visual_score import visual_eval_v3_multi
from multiprocessing import Pool
import contextlib, joblib
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import json
import os
import csv
import shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def print_multi_score(multi_score):
    _, final_size_score, final_matched_text_score, final_position_score, final_text_color_score, final_clip_score = multi_score
    print()
    print("Block-Match: ", final_size_score)
    print("Text: ", final_matched_text_score)
    print("Position: ", final_position_score)
    print("Color: ", final_text_color_score)
    print("CLIP: ", final_clip_score)
    print("--------------------------------\n")



if __name__ == "__main__":
    debug = False
    multiprocessing = True

    orig_reference_dir = "../testset_final"
    eval_name = "testset_final"

    ## copy the original reference directory to a new directory
    ## because we will be creating new screenshots
    reference_dir = "../testset_final_" + eval_name
    os.makedirs(reference_dir, exist_ok=True)
    for filename in os.listdir(orig_reference_dir):
        if filename.endswith(".html") or filename == "rick.jpg":
            shutil.copy(os.path.join(orig_reference_dir, filename), os.path.join(reference_dir, filename))
    print ("copied original reference directory to ", reference_dir)

    test_dirs = {
        "gpt4v_direct_prompting": "../predictions_final/gpt4v_direct_prompting",
        "gemini_direct_prompting": "../predictions_final/gemini_direct_prompting"
    }
    file_name_list = []

    ## check if the file is in all prediction directories
    for filename in os.listdir(reference_dir):
        if filename.endswith(".html"):
            if all([os.path.exists(os.path.join(test_dirs[key], filename)) for key in test_dirs]):
                file_name_list.append(filename)

    print ("total #egs: ", len(file_name_list))

    input_lists = []
    for filename in file_name_list:

        input_pred_list = [os.path.join(test_dirs[key], filename) for key in test_dirs]
        original = os.path.join(reference_dir, filename)

        input_list = [input_pred_list, original]
        input_lists.append(input_list)

    if multiprocessing:
        with tqdm_joblib(tqdm(total=len(input_lists))) as progress_bar:
            return_score_lists = list(tqdm(Parallel(n_jobs=8)(delayed(visual_eval_v3_multi)(input_list, debug=debug) for input_list in input_lists), total=len(input_lists)))
    else:
        return_score_lists = []
        for input_list in tqdm(input_lists):
            return_score_list = visual_eval_v3_multi(input_list, debug=debug)
            return_score_lists.append(return_score_list)

    per_key_data = {key: [] for key in test_dirs}

    for i, filename in enumerate(file_name_list):
        idx = 0
        return_score_list = return_score_lists[i]

        if return_score_list:
            for key in test_dirs:
                if multiprocessing:
                    matched, final_score, multi_score = return_score_list[idx]
                else:
                    matched = return_score_list[idx][0]
                    final_score = return_score_list[idx][1]
                    multi_score = return_score_list[idx][2]
                idx += 1

                block_match = multi_score[0]
                text_match = multi_score[1]
                position_match = multi_score[2]
                text_color_match = multi_score[3]
                clip_score = multi_score[4]
                
                nemd = multi_score[5]
                mae = multi_score[6]
                ssim = multi_score[7]
                clip = multi_score[8]

                bleu        = multi_score[9]
                rouge       = multi_score[10]
                tree_bleu   = multi_score[11]
                tree_rouge1 = multi_score[12]

                if block_match == 0 and text_match == 0 and position_match == 0 and text_color_match == 0:
                    clip_score = 0
                    nemd = 0
                    mae = 0
                    ssim = 0
                    clip = 0
                    bleu = 0
                    rouge = 0
                    tree_bleu = 0
                    tree_rouge1 = 0
                current_score = [
                    filename,
                    block_match,       
                    text_match,
                    position_match,
                    text_color_match,
                    clip_score,
                    nemd,
                    mae,
                    ssim,  
                    clip,
                    bleu,
                    rouge,
                    tree_bleu,
                    tree_rouge1
                ]
                per_key_data[key].append(current_score)
        else:
            print(f"{filename} didn't get a score")

    for key in test_dirs:
        output_csv = f"metrics/{key}_{eval_name}.csv"
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "filename",
                "block_match",
                "text_match",
                "position_match",
                "text_color_match",
                "clip_score",
                "nemd",
                "mae",
                "ssim",
                "clip",
                "bleu",
                "rouge",
                "tree_bleu",
                "tree_rouge1"
            ])
            writer.writerows(per_key_data[key])

        data = np.array(per_key_data[key], dtype=object)
        if len(data) == 0:
            print(f"--- No valid data for {key} ---")
            print("-----------------------------\n")
            continue

        block_match_avg      = np.mean(data[:, 1].astype(float))
        text_match_avg       = np.mean(data[:, 2].astype(float))
        position_match_avg   = np.mean(data[:, 3].astype(float))
        text_color_match_avg = np.mean(data[:, 4].astype(float))
        clip_score_avg       = np.mean(data[:, 5].astype(float))
        nemd_avg             = np.mean(data[:, 6].astype(float))
        mae_avg              = np.mean(data[:, 7].astype(float))
        ssim_avg             = np.mean(data[:, 8].astype(float))
        clip_avg             = np.mean(data[:, 9].astype(float))
        bleu_avg             = np.mean(data[:, 10].astype(float))
        rouge_avg            = np.mean(data[:, 11].astype(float))
        tree_bleu_avg        = np.mean(data[:, 12].astype(float))
        tree_rouge1_avg      = np.mean(data[:, 13].astype(float))

        print(f"--- Average for {key} ---")
        print(f"block_match: {block_match_avg:.4f}")
        print(f"text_match: {text_match_avg:.4f}")
        print(f"position_match: {position_match_avg:.4f}")
        print(f"text_color_match: {text_color_match_avg:.4f}")
        print(f"clip_score: {clip_score_avg:.4f}")
        print(f"nemd: {nemd_avg:.4f}")
        print(f"mae: {mae_avg:.4f}")
        print(f"ssim: {ssim_avg:.4f}")
        print(f"clip: {clip_avg:.4f}")
        print(f"bleu: {bleu_avg:.4f}")
        print(f"rouge: {rouge_avg:.4f}")
        print(f"tree_bleu: {tree_bleu_avg:.4f}")
        print(f"tree_rouge1: {tree_rouge1_avg:.4f}")
        print("-----------------------------\n")