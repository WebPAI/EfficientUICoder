from .clip_encoder import CLIPVisionTower_EfficientUICoder
from .llava_arch import encode_images_visionzip, encode_images_visionzip_multi
from .llava_arch import restore_image_features_sorted_EfficientUICoder,prepare_inputs_labels_for_multimodal_EfficientUICoder

def EfficientUICoder(model, dense_labels=None, token_selection_ratio=0):
    model.model.vision_tower.vision_tower._info = {
        "dense_labels": dense_labels,
        "token_selection_ratio": token_selection_ratio,
    }

    from llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
    CLIPVisionTower.forward = CLIPVisionTower_EfficientUICoder.forward

    from llava.model.llava_arch import LlavaMetaForCausalLM
    if hasattr(LlavaMetaForCausalLM, 'prepare_inputs_labels_for_multimodal'):
        LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_EfficientUICoder
        LlavaMetaForCausalLM.restore_image_features_sorted = restore_image_features_sorted_EfficientUICoder
        LlavaMetaForCausalLM.encode_images_visionzip_multi = encode_images_visionzip_multi
        LlavaMetaForCausalLM.encode_images_visionzip = encode_images_visionzip
    return model