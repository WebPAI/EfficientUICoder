import re
import os
from typing import List, Optional, Union
import tinycss2
from html.parser import HTMLParser
import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateNonBeamOutput,
    GenerateEncoderDecoderOutput,
    GenerateDecoderOnlyOutput
)


def normalize_selector(selector):
    # Replace all nth-child/nth-of-type with generic (*)
    selector = re.sub(
        r":nth-(child|of-type|last-child|last-of-type|first-child|first-of-type)\([^)]*\)",
        r":nth-\1(*)",
        selector,
    )
    # Remove redundant spaces
    selector = re.sub(r"\s+", " ", selector).strip()
    # Split by comma, normalize and sort
    parts = [s.strip() for s in selector.split(",")]
    parts = sorted(parts)
    return ",".join(parts)


def parse_css_counts(css_text):
    """
    Parse a CSS fragment using tinycss2.
    Return a dict where keys are in the format "normalized_selector::property::value".
    """
    counts = {}
    try:
        rules = tinycss2.parse_rule_list(css_text, skip_whitespace=True, skip_comments=True)
        for rule in rules:
            # Only process qualified rules (selector { declarations })
            if getattr(rule, "type", None) == "qualified-rule":
                selector_str = "".join([tok.serialize() for tok in rule.prelude]).strip()
                if not selector_str:
                    continue
                norm_sel = normalize_selector(selector_str)

                decl_source = "".join([tok.serialize() for tok in getattr(rule, "content", [])])
                if not decl_source:
                    continue
                decls = tinycss2.parse_declaration_list(decl_source, skip_whitespace=True, skip_comments=True)

                for decl in decls:
                    if getattr(decl, "type", None) == "declaration":
                        prop_name = decl.name.strip()
                        prop_val = "".join([tok.serialize() for tok in decl.value]).strip()
                        key = f"{norm_sel}::{prop_name}::{prop_val}"
                        counts[key] = counts.get(key, 0) + 1
    except Exception as e:
        # Fallback on parse error (print debug info but don't raise)
        print("parse_css_counts (tinycss2) exception:", e)
    return counts


def css_rule_signatures(css_text: str) -> List[str]:
    """
    Normalize CSS qualified-rule(s) into signatures for duplication detection.
    Signature = normalized selector list + sorted "prop" list.
    """
    sigs = []
    try:
        rules = tinycss2.parse_rule_list(css_text, skip_whitespace=True, skip_comments=True)
        for rule in rules:
            if getattr(rule, "type", None) == "qualified-rule":
                selector_str = "".join(tok.serialize() for tok in rule.prelude).strip()
                if not selector_str:
                    continue
                norm_sel = normalize_selector(selector_str)

                decl_source = "".join(tok.serialize() for tok in getattr(rule, "content", []))
                if not decl_source:
                    continue
                decls = tinycss2.parse_declaration_list(decl_source, skip_whitespace=True, skip_comments=True)

                items = []
                for decl in decls:
                    if getattr(decl, "type", None) == "declaration":
                        prop_name = decl.name.strip()
                        prop_val = "".join(tok.serialize() for tok in decl.value).strip()
                        # Only use property name in signature
                        items.append(f"{prop_name}")

                if items:
                    sig = f"{norm_sel}::" + "|".join(sorted(items))
                    sigs.append(sig)
    except Exception as e:
        print("css_rule_signatures exception:", e)
    return sigs


def feed_css_incremental(generation_state, new_text, decay_factor=0.5, step=1):
    generation_state["css_buffer"] += new_text

    while "}" in generation_state["css_buffer"]:
        close_idx = generation_state["css_buffer"].find("}") + 1
        open_idx = generation_state["css_buffer"].rfind("{", 0, close_idx)
        if open_idx == -1:
            generation_state["css_buffer"] = generation_state["css_buffer"][close_idx:]
            continue

        prev_close = generation_state["css_buffer"].rfind("}", 0, open_idx)
        start_idx = prev_close + 1 if prev_close != -1 else 0
        complete_rule = generation_state["css_buffer"][start_idx:close_idx].strip()
        generation_state["css_buffer"] = generation_state["css_buffer"][close_idx:]

        if not complete_rule:
            continue

        sigs = css_rule_signatures(complete_rule)
        for sig in sigs:
            # === Global accumulation ===
            css_counts = generation_state.setdefault("css_counts", {})
            css_counts[sig] = css_counts.get(sig, 0) + 1
            # print(f"css_counts {sig}: {css_counts[sig]}")

            if css_counts[sig] >= 3:
                trigger_count = generation_state.setdefault("css_penalty_trigger_count", {}).get(sig, 0) + 1
                generation_state["css_penalty_trigger_count"][sig] = trigger_count
                factor = decay_factor ** trigger_count
                penalty_steps = generation_state.get("css_penalty_steps_remaining", 0)
                generation_state["css_penalty_steps_remaining"] = step
                generation_state["css_penalty_factor"] = factor
                # print(
                #     f"[Penalty-Init] CSS sig {sig} trigger {trigger_count}, "
                #     f"factor={factor:.3f}, steps={generation_state['css_penalty_steps_remaining']}"
                # )


class MainHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack = []     # Track tag hierarchy
        self.elements = []  # Store parsed elements as (tag, attrs, text)

    def handle_starttag(self, tag, attrs):
        # Record tag and attributes
        attr_dict = dict(attrs)
        # Sort attributes so order doesn't affect deduplication
        sorted_attrs = tuple(sorted(attr_dict.items()))
        self.stack.append((tag, sorted_attrs, ""))

    def handle_data(self, data):
        if self.stack:
            tag, attrs, text = self.stack.pop()
            text += data.strip()
            self.stack.append((tag, attrs, text))

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1][0] == tag:
            element = self.stack.pop()
            self.elements.append(element)


def feed_html_incremental(generation_state, new_text, decay_factor=0.5, step=1):
    generation_state["html_buffer"] += new_text
    temp_parser = MainHTMLParser()
    try:
        temp_parser.feed(generation_state["html_buffer"])
    except Exception:
        return

    if temp_parser.elements:
        for tag, attrs, text in temp_parser.elements:
            attr_str = ";".join(f"{k}={v}" for k, v in attrs)
            text_norm = re.sub(r"\s+", " ", text.strip())[:50]
            key = f"{tag}::{attr_str}::{text_norm}"

            html_counts = generation_state.setdefault("html_counts", {})
            html_counts[key] = html_counts.get(key, 0) + 1
            # print(f"html_counts {key}: {html_counts[key]}")

            if html_counts[key] >= 3:
                trigger_count = generation_state.setdefault("html_penalty_trigger_count", {}).get(key, 0) + 1
                generation_state["html_penalty_trigger_count"][key] = trigger_count
                factor = decay_factor ** trigger_count
                penalty_steps = generation_state.get("html_penalty_steps_remaining", 0)
                generation_state["html_penalty_steps_remaining"] = step
                generation_state["html_penalty_factor"] = factor
                # print(
                #     f"[Penalty-Init] HTML key {key} trigger {trigger_count}, "
                #     f"factor={factor:.3f}, steps={generation_state['html_penalty_steps_remaining']}"
                # )

        generation_state["html_buffer"] = ""


def detect_content_repetition_after_symbol(
    text, scan_tail_len=500, min_word_repeats=3, min_phrase_len=8
):
    """
    Detect repeated words or phrases after the latest '>' or '}' symbol.
    """
    tail_text = text[-scan_tail_len:]
    last_pos_gt = tail_text.rfind(">")
    last_pos_brace = tail_text.rfind("}")
    last_pos = max(last_pos_gt, last_pos_brace)
    segment = tail_text[last_pos + 1 :] if last_pos != -1 else tail_text
    # Remove placeholder URLs
    segment = re.sub(r'https?://placehold\.co/\d+x\d+/\w+/\w+', '', segment)

    # Word repetition detection
    word_pattern = re.compile(r"(\w+|[.#]\w+|\+\s*\w+)")
    matches = word_pattern.findall(segment)
    consecutive_count = 1
    if len(matches) > 1:
        for i in range(1, len(matches)):
            w = matches[i]
            if w.isdigit():
                consecutive_count = 1
                continue
            if w == matches[i - 1] and not matches[i - 1].isdigit():
                consecutive_count += 1
                if consecutive_count >= min_word_repeats:
                    # print(f"[Detected] Short word '{w}' repeated {consecutive_count} times")
                    return True
            else:
                consecutive_count = 1

    # Phrase repetition detection
    phrase_pattern = re.compile(r"(\S{" + str(min_phrase_len) + r",}?)\1")
    m = phrase_pattern.search(segment)
    if m:
        phrase = m.group(1)
        # print(f"[Detected] Long phrase repetition: {phrase[:80]}{'...' if len(phrase) > 80 else ''}")
        return True

    return False


def update_content_repetition_state(combined_text, generation_state, decay_factor=0.5, extra_penalty_steps=1):
    is_repeat = detect_content_repetition_after_symbol(combined_text)
    prev_active = generation_state.get("content_penalty_active", False)
    if is_repeat:
        generation_state["content_penalty_active"] = True
        trigger_count = generation_state.get("content_penalty_trigger_count", 0) + 1
        generation_state["content_penalty_trigger_count"] = trigger_count
        factor = decay_factor ** trigger_count
        generation_state["content_penalty_factor"] = factor
        generation_state["content_penalty_steps_remaining"] = -1  # -1 means "persistent mode"
        # print(f"[Penalty-Init] content repetition trigger {trigger_count}, factor={factor:.3f}")
        if not prev_active:
            # print("[ContentRepeat] detected -> start content penalty")
            pass
    else:
        if prev_active:
            # Transition from repeating -> non-repeating, enter cooldown
            generation_state["content_penalty_active"] = False
            generation_state["content_penalty_steps_remaining"] = extra_penalty_steps
            # print(f"[ContentRepeat] no longer repeating -> entering cooldown ({extra_penalty_steps} steps)")



def new_sample(
    self,
    input_ids: torch.LongTensor,
    logits_processor: LogitsProcessorList,
    stopping_criteria: StoppingCriteriaList,
    generation_config: GenerationConfig,
    synced_gpus: bool,
    streamer: Optional["BaseStreamer"],
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
    can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.
        generation_config ([`~generation.GenerationConfig`]):
            The generation configuration to be used as parametrization of the decoding method.
        synced_gpus (`bool`):
            Whether to continue running the while loop until max_length (needed to avoid deadlocking with
            `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
            an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
        A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.
    """
    # init values
    pad_token_id = generation_config._pad_token_tensor
    output_attentions = generation_config.output_attentions
    output_hidden_states = generation_config.output_hidden_states
    output_scores = generation_config.output_scores
    output_logits = generation_config.output_logits
    return_dict_in_generate = generation_config.return_dict_in_generate
    has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
    do_sample = generation_config.do_sample

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
    raw_logits = () if (return_dict_in_generate and output_logits) else None
    decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
    cross_attentions = () if (return_dict_in_generate and output_attentions) else None
    decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

    # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
    if return_dict_in_generate and self.config.is_encoder_decoder:
        encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
        encoder_hidden_states = (
            model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
        )

    # keep track of which sequences are already finished
    batch_size, cur_len = input_ids.shape
    this_peer_finished = False
    unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
    model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

    model_forward = self.__call__
    if isinstance(model_kwargs.get("past_key_values"), Cache):
        is_compileable = model_kwargs["past_key_values"].is_compileable and self._supports_static_cache
        if getattr(self, "hf_quantizer", None) is not None:
            is_compileable &= self.hf_quantizer.is_compileable
        is_compileable = is_compileable and not generation_config.disable_compile
        if is_compileable and (
            self.device.type == "cuda" or generation_config.compile_config._compile_all_devices
        ):
            os.environ["TOKENIZERS_PARALLELISM"] = "0"
            model_forward = self.get_compiled_call(generation_config.compile_config)

    if generation_config.prefill_chunk_size is not None:
        model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
        is_prefill = False
    else:
        is_prefill = True

    decay_factor = getattr(self, "decay_factor", 0.5)
    penalty_step = getattr(self, "penalty_step", 1)
    generation_state = {
        "mode": None,  # None / 'css' / 'html'
        "css_buffer": "",
        "html_buffer": "",
        "css_counts": {},     # Global CSS rule counts
        "html_counts": {},    # Global HTML element counts
        "lookbehind_buffer": "",
        "css_penalty_steps_remaining": 0,
        "html_penalty_steps_remaining": 0,
        "content_penalty_steps_remaining": 0,
        "css_penalty_factor": 1.0,     # Default: no penalty
        "html_penalty_factor": 1.0,    
        "content_penalty_factor": 1.0,
        "prev_len": input_ids.shape[1],
        "css_penalty_trigger_count": {},    # Trigger counts for each CSS signature
        "html_penalty_trigger_count": {},   # Trigger counts for each HTML key
        "content_penalty_trigger_count": 0, # Content repetition trigger count
        "html_seen": False,      # Whether <html> has already appeared
        "force_suffix_ids": [],  # Forced token sequence (backup)
        "force_suffix_pos": 0,   # Position of forced token output
    }

    while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # prepare variable output controls (note: some models won't accept all output controls)
        model_inputs.update({"output_attentions": output_attentions} if output_attentions else {})
        model_inputs.update({"output_hidden_states": output_hidden_states} if output_hidden_states else {})

        if is_prefill:
            outputs = self(**model_inputs, return_dict=True)
            is_prefill = False
        else:
            outputs = model_forward(**model_inputs, return_dict=True)

        # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=self.config.is_encoder_decoder,
        )
        if synced_gpus and this_peer_finished:
            continue

        # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
        # (the clone itself is always small)
        next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

        # pre-process distribution
        next_token_scores = logits_processor(input_ids, next_token_logits)



        # penalty
        if hasattr(self, "tokenizer"):
            tokenizer = self.tokenizer

            # Extract new tokens since last step
            new_tokens = input_ids[0][generation_state.get("prev_len", 0):]
            new_text = tokenizer.decode(new_tokens, clean_up_tokenization_spaces=False)
            generation_state["prev_len"] = len(input_ids[0])

            # Maintain lookbehind buffer (sliding window)
            lookbehind_limit = 500
            generation_state["lookbehind_buffer"] = (
                generation_state["lookbehind_buffer"] + new_text
            )[-lookbehind_limit:]
            combined_text = generation_state["lookbehind_buffer"]

            # === CSS mode detection ===
            if re.search(r"<\s*style\b", combined_text, re.IGNORECASE) and generation_state.get("mode") != "css":
                generation_state["mode"] = "css"
                generation_state["css_buffer"] = ""
                generation_state["css_wait_for_tag_close"] = True
                generation_state["lookbehind_buffer"] = ""

            # === HTML mode detection ===
            if re.search(r"<\s*body\b", combined_text, re.IGNORECASE) and generation_state.get("mode") != "html":
                generation_state["mode"] = "html"
                generation_state["html_buffer"] = ""
                generation_state["html_wait_for_tag_close"] = True
                generation_state["lookbehind_buffer"] = ""
                generation_state["html_seen"] = True 

            # === Incremental CSS buffer parsing ===
            if generation_state.get("mode") == "css":
                if generation_state.get("css_wait_for_tag_close", False):
                    pos = new_text.find(">")
                    if pos != -1:
                        new_text = new_text[pos + 1:]
                        generation_state["css_wait_for_tag_close"] = False
                    else:
                        new_text = ""

                new_text = new_text.lstrip("\n\r\t")
                feed_css_incremental(generation_state, new_text, decay_factor=decay_factor, step=penalty_step)

                # Apply CSS penalty if active
                if generation_state.get("css_penalty_steps_remaining", 0) > 0:
                    top1_val, top1_idx = torch.max(next_token_scores, dim=-1)
                    factor = generation_state.get("css_penalty_factor", decay_factor)
                    next_token_scores[0, top1_idx] *= factor
                    generation_state["css_penalty_steps_remaining"] -= 1
                    # print(f"[Penalty-Step] CSS applying factor={factor:.3f}, "
                    #     f"remaining={generation_state['css_penalty_steps_remaining']}")

                # Exit CSS mode when </style> appears
                if re.search(r"</\s*style\s*>", combined_text, re.IGNORECASE):
                    generation_state["mode"] = None
                    generation_state["lookbehind_buffer"] = ""
                    generation_state["css_wait_for_tag_close"] = False

            # === Incremental HTML buffer parsing ===
            if generation_state.get("mode") == "html":
                if generation_state.get("html_wait_for_tag_close", False):
                    pos = new_text.find(">")
                    if pos != -1:
                        new_text = new_text[pos + 1:]
                        generation_state["html_wait_for_tag_close"] = False
                    else:
                        new_text = ""

                new_text = new_text.lstrip("\n\r\t")
                feed_html_incremental(generation_state, new_text, decay_factor=decay_factor, step=penalty_step)

                # Apply HTML penalty if active
                if generation_state.get("html_penalty_steps_remaining", 0) > 0:
                    top1_val, top1_idx = torch.max(next_token_scores, dim=-1)
                    factor = generation_state.get("html_penalty_factor", decay_factor)
                    next_token_scores[0, top1_idx] *= factor
                    generation_state["html_penalty_steps_remaining"] -= 1
                    # print(f"[Penalty-Step] HTML applying factor={factor:.3f}, "
                    #     f"remaining={generation_state['html_penalty_steps_remaining']}")

                # Exit HTML mode when </body> appears
                if re.search(r"</\s*body\s*>", combined_text, re.IGNORECASE):
                    generation_state["mode"] = None
                    generation_state["lookbehind_buffer"] = ""
                    generation_state["html_wait_for_tag_close"] = False

            # === Handle content repetition state ===
            update_content_repetition_state(
                combined_text,
                generation_state,
                decay_factor=decay_factor,
                extra_penalty_steps=penalty_step
            )

            steps = generation_state.get("content_penalty_steps_remaining", 0)

            if steps == -1:  # Persistent repetition mode
                top1_val, top1_idx = torch.max(next_token_scores, dim=-1)
                factor = generation_state.get("content_penalty_factor", decay_factor)
                next_token_scores[0, top1_idx] *= factor
                # print(f"[Penalty-Step] content (repeating) factor={factor:.3f}")

            elif steps > 0:  # Cooldown phase
                top1_val, top1_idx = torch.max(next_token_scores, dim=-1)
                factor = generation_state.get("content_penalty_factor", decay_factor)
                next_token_scores[0, top1_idx] *= factor
                generation_state["content_penalty_steps_remaining"] -= 1
                remaining = generation_state["content_penalty_steps_remaining"]
                # print(f"[Penalty-Step] content (cooldown) factor={factor:.3f}, remaining={remaining}")

                if remaining == 0:
                    # Cooldown finished -> reset penalty state
                    generation_state["content_penalty_factor"] = 1.0
                    generation_state["content_penalty_trigger_count"] = 0
                    # print("[ContentRepeat] cooldown finished -> penalty cleared")

         

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_token_scores,)
            if output_logits:
                raw_logits += (next_token_logits,)
            if output_attentions:
                decoder_attentions += (
                    (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                )
                if self.config.is_encoder_decoder:
                    cross_attentions += (outputs.cross_attentions,)

            if output_hidden_states:
                decoder_hidden_states += (
                    (outputs.decoder_hidden_states,)
                    if self.config.is_encoder_decoder
                    else (outputs.hidden_states,)
                )

        # token selection
        if do_sample:
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            
            # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            next_tokens = torch.argmax(next_token_scores, dim=-1)

        # finished sentences should have their next token be a padding token
        if has_eos_stopping_criteria:
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
        this_peer_finished = unfinished_sequences.max() == 0
        cur_len += 1

        # This is needed to properly delete outputs.logits which may be very large for first iteration
        # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
        del outputs

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                encoder_attentions=encoder_attentions,
                encoder_hidden_states=encoder_hidden_states,
                decoder_attentions=decoder_attentions,
                cross_attentions=cross_attentions,
                decoder_hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
        else:
            return GenerateDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                logits=raw_logits,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids