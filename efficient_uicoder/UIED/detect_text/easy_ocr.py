import cv2
import easyocr
import numpy as np
from PIL import Image
def merge_bboxs(results, threshold=50):
    while True:
        flag = True
        for idx, item in enumerate(results):
            bbox, text, _ = item
            left, top, right, bottom = bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]
            for idx2, item2 in enumerate(results):
                if idx2 == idx:
                    continue
                bbox2, text2, _ = item2
                left2, top2, right2, bottom2 = bbox2[0][0], bbox2[0][1], bbox2[2][0], bbox2[2][1]
                if not (left2 > right + threshold or right2 < left - threshold or bottom2 < top - threshold or top2 > bottom + threshold):
                    left3, top3, right3, bottom3 = min(left, left2), min(top, top2), max(right, right2), max(bottom, bottom2)
                    results[idx] = ([[left3, top3], [right3, top3], [right3, bottom3], [left3, bottom3]], f'{text}\n{text2}', None)
                    results = results[:idx2]+results[idx2+1:]
                    flag = False
                    break
            if not flag:
                break
        if flag:
            return results

def ocr_with_easyocr(pil_image, lang_list=['en', 'ch_sim'], merge_threshold=50):
    reader = easyocr.Reader(lang_list, gpu=False) 
    image_np = np.array(pil_image)
    results = reader.readtext(image_np)
    results = merge_bboxs(results, threshold=merge_threshold)

    for item in results:
        bbox, text, _ = item
        top_left = tuple(map(int, bbox[0]))
        top_right = tuple(map(int, bbox[1]))
        bottom_right = tuple(map(int, bbox[2]))
        bottom_left = tuple(map(int, bbox[3]))

        cv2.polylines(image_np, [np.array([top_left, top_right, bottom_right, bottom_left])], isClosed=True, color=(0, 255, 0), thickness=2)

    result_image = Image.fromarray(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))

    return results, result_image
