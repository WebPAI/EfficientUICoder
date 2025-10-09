import sys
import cv2
import clip
import torch
import numpy as np
from PIL import Image
from bs4 import BeautifulSoup
from skimage.metrics import structural_similarity
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge
from Design2Code.metrics.html_tree import *
smoothie = SmoothingFunction().method4

CLIP_MODEL, CLIP_PREPROCESS = None,None
def clip_encode(ims,device='cuda'):
    global CLIP_MODEL
    global CLIP_PREPROCESS
    if not CLIP_MODEL:        
        CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=device)
    with torch.no_grad():
        img_tmps = torch.stack([CLIP_PREPROCESS(im) for im in ims]).to(device)
        img_feas = CLIP_MODEL.encode_image(img_tmps).cpu()
    return img_feas   

def clip_sim(im1, im2, device='cuda'):
    feas = clip_encode([im1 , im2], device)
    return torch.nn.functional.cosine_similarity(feas[0], feas[1], dim=0).item()

def ssim(img1: np.ndarray, img2: np.ndarray):
    """
    [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
        (2004). Image quality assessment: From error visibility to
        structural similarity. IEEE Transactions on Image Processing,
        13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        :DOI:`10.1109/TIP.2003.819861`
    """
    # img2 = img2.resize(img1.size, Image.LANCZOS)
    assert (
        img2.shape == img1.shape
    ), "to caculate the SSIM, two images should have the same shape."
    ssim_value = structural_similarity(
        img1,
        img2,
        multichannel=True,
        channel_axis=2,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
        data_range=255.0,
    )
    return ssim_value

def bleu_rouge(original: str, generated: str):
    # Parse HTML and extract plain text, then split into word lists
    soup1 = BeautifulSoup(original, "html.parser")
    soup2 = BeautifulSoup(generated, "html.parser")
    original = soup1.get_text().split()
    generated = soup2.get_text().split()
    # If either hypothesis or reference is empty, return 0 scores
    if not original or not generated:
        return 0.0, 0.0
    # Compute BLEU score (1-gram)
    bleu = bleu_score.sentence_bleu(
        [original], generated,
        weights=(1.0, 0, 0, 0),
        smoothing_function=bleu_score.SmoothingFunction().method4
    )
    # Compute ROUGE score
    rouge = Rouge()
    rouge_scores = rouge.get_scores(" ".join(generated), " ".join(original))
    # Return BLEU and ROUGE-1 recall
    return bleu, rouge_scores[0]["rouge-1"]["r"]

def dom_sim(ref:str, cand:str):
    """
    Perform subtree matching, partly following the method from [1]
    match_score = count(cand matched in ref)/count(subtrees number of ref)
    
    [1] Ren, Shuo et al. “CodeBLEU: a Method for Automatic Evaluation of Code Synthesis.” ArXiv abs/2009.10297 (2020): n. pag.
    """
    ref_tree_nodes = html2tree(ref)
    cand_tree_nodes = html2tree(cand)

    if len(ref_tree_nodes) == 0 or len(cand_tree_nodes) == 0:
        return 0, 0
    
    def collect_all_subtrees(nodes, height=1): 
        subtrees = []
        for node in nodes:
            if len(node.childs) == 0:
                continue
            names = [node.name.strip().lower()]
            for child in node.childs:
                names.append(child.name.strip().lower())            
            subtrees.append('_'.join(names))
        return subtrees
    
    ref_subtree_seqs = collect_all_subtrees(ref_tree_nodes)
    cand_subtree_seqs = collect_all_subtrees(cand_tree_nodes)
    
    if len(ref_subtree_seqs) == 0 or len(cand_subtree_seqs) == 0:
        return 0.0, 0.0

    match_count = 0
    for seq in set(cand_subtree_seqs):
        if seq in set(ref_subtree_seqs):
            match_count += 1
    
    tree_rouge_1 = match_count/len(set(ref_subtree_seqs))
    
    match_count = 0
    for seq in cand_subtree_seqs:
        if seq in set(ref_subtree_seqs):
            match_count += 1
        
    tree_bleu = match_count/len(cand_subtree_seqs)
    
    return tree_bleu, tree_rouge_1


def image_sim_scores(img1_path, img2_path):

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img2 = cv2.resize(
        img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_LANCZOS4
    )    

    ssim_value = ssim(img1, img2)
    clip_sim_value = clip_sim(Image.open(img1_path), Image.open(img2_path), 'cpu')

    return ssim_value, clip_sim_value

def html_sim_scores(html1_path, html2_path):   
    with open(html1_path, "r") as f:
         html1 = f.read()
    with open(html2_path, "r") as f:
         html2 = f.read()
    assert len(html1) >0 and len(html2)>0, "The html must not be empty!"
    sys.setrecursionlimit(6000)
    bleu, rouge = bleu_rouge(html1, html2)
    tree_bleu, tree_rouge_1 = dom_sim(html1, html2)
 
    return (bleu, rouge, tree_bleu, tree_rouge_1)