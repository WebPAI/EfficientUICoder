import re
import time
import warnings
from html.parser import HTMLParser
from typing import List, Optional, Union
import tinycss2
import torch
import torch.distributed as dist
from transformers.generation.logits_process import LogitsProcessorList
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
    validate_stopping_criteria,
)
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)


def normalize_selector(selector):
    # Replace nth-child/nth-of-type related selectors with a normalized form (*)
    selector = re.sub(
        r":nth-(child|of-type|last-child|last-of-type|first-child|first-of-type)\([^)]*\)",
        r":nth-\1(*)",
        selector,
    )
    # Remove redundant spaces
    selector = re.sub(r"\s+", " ", selector).strip()
    # Split by comma, normalize, and sort for consistency
    parts = [s.strip() for s in selector.split(",")]
    parts = sorted(parts)
    return ",".join(parts)


def parse_css_counts(css_text):
    """
    Parse a complete CSS snippet (e.g., selector { ... }) using tinycss2.
    Return a dict of counts with keys in the format:
    "normalized_selector::property::value"
    """
    counts = {}
    try:
        # Parse rule list (allow single rule as well)
        rules = tinycss2.parse_rule_list(css_text, skip_whitespace=True, skip_comments=True)
        for rule in rules:
            # Only handle qualified rules (selector { declarations })
            if getattr(rule, "type", None) == "qualified-rule":
                # Extract selector string
                selector_str = "".join([tok.serialize() for tok in rule.prelude]).strip()
                if not selector_str:
                    continue
                norm_sel = normalize_selector(selector_str)

                # Extract declarations from rule content
                decl_source = "".join([tok.serialize() for tok in getattr(rule, "content", [])])
                if not decl_source:
                    continue
                decls = tinycss2.parse_declaration_list(
                    decl_source, skip_whitespace=True, skip_comments=True
                )
                for decl in decls:
                    if getattr(decl, "type", None) == "declaration":
                        prop_name = decl.name.strip()
                        # Serialize token list into property value string
                        prop_val = "".join([tok.serialize() for tok in decl.value]).strip()
                        key = f"{norm_sel}::{prop_name}::{prop_val}"
                        counts[key] = counts.get(key, 0) + 1
    except Exception as e:
        # Fallback when parsing fails (log but do not raise)
        print("parse_css_counts (tinycss2) exception:", e)
    return counts


def css_rule_signatures(css_text: str) -> List[str]:
    """
    Normalize a full CSS qualified-rule into a signature (for duplicate detection).
    Signature = normalized selector(s) + sorted "prop" list.
    Returns a list of possible signatures (usually one per qualified-rule).
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
                decls = tinycss2.parse_declaration_list(
                    decl_source, skip_whitespace=True, skip_comments=True
                )

                items = []
                for decl in decls:
                    if getattr(decl, "type", None) == "declaration":
                        prop_name = decl.name.strip()
                        prop_val = "".join(tok.serialize() for tok in decl.value).strip()
                        # Keep only property names (ignore values for signature)
                        items.append(f"{prop_name}")

                if items:
                    sig = f"{norm_sel}::" + "|".join(sorted(items))
                    sigs.append(sig)
    except Exception as e:
        print("css_rule_signatures exception:", e)
    return sigs


def feed_css_incremental(generation_state, new_text, decay_factor=0.5, step=3):
    """
    Incrementally process CSS text fragments and update repetition penalties.
    """
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
        complete_rule = complete_rule.replace("<0x0A>", "\n")

        sigs = css_rule_signatures(complete_rule)
        for sig in sigs:
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
        self.stack = []      # Track tag nesting
        self.elements = []   # Store parsed elements as (tag, attrs, text)

    def handle_starttag(self, tag, attrs):
        # Record current tag and attributes
        attr_dict = dict(attrs)
        # Sort attributes to ensure order does not affect deduplication
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


def feed_html_incremental(generation_state, new_text, decay_factor=0.5, step=3):
    """
    Incrementally process HTML fragments and update repetition penalties.
    """
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
    Detect content repetition after the last '>' or '}' symbol.
    Supports both short word repetition and longer phrase repetition.
    """
    tail_text = text[-scan_tail_len:]
    last_pos_gt = tail_text.rfind(">")
    last_pos_brace = tail_text.rfind("}")
    last_pos = max(last_pos_gt, last_pos_brace)
    segment = tail_text[last_pos + 1 :] if last_pos != -1 else tail_text
    segment = re.sub(r'https?://placehold\.co/\d+x\d+/\w+/\w+', '', segment)

    # Detect repeated short words
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
                    # print(f"[Detected] short word '{w}' repeated {consecutive_count} times consecutively")
                    return True
            else:
                consecutive_count = 1

    # Detect repeated long phrases
    phrase_pattern = re.compile(r"(\S{" + str(min_phrase_len) + r",}?)\1")
    m = phrase_pattern.search(segment)
    if m:
        phrase = m.group(1)
        # print(f"[Detected] long phrase repetition: {phrase[:80]}{'...' if len(phrase) > 80 else ''}")
        return True

    return False


def update_content_repetition_state(
    combined_text, generation_state, decay_factor=0.5, extra_penalty_steps=3
):
    """
    Update generation state with repetition penalty if repeated content is detected.
    """
    is_repeat = detect_content_repetition_after_symbol(combined_text)
    prev_active = generation_state.get("content_penalty_active", False)
    if is_repeat:
        generation_state["content_penalty_active"] = True
        trigger_count = generation_state.get("content_penalty_trigger_count", 0) + 1
        generation_state["content_penalty_trigger_count"] = trigger_count
        factor = decay_factor ** trigger_count
        generation_state["content_penalty_factor"] = factor
        generation_state["content_penalty_steps_remaining"] = -1  # -1 = persistent penalty mode
        # print(f"[Penalty-Init] content repetition trigger {trigger_count}, factor={factor:.3f}")
        if not prev_active:
            # print("[ContentRepeat] detected -> start content penalty")
            pass
    else:
        if prev_active:
            # Transition from repeating -> non-repeating: enter cooldown
            generation_state["content_penalty_active"] = False
            generation_state["content_penalty_steps_remaining"] = extra_penalty_steps
            # print(
            #     f"[ContentRepeat] no longer repeating -> entering cooldown "
            #     f"({extra_penalty_steps} steps)"
            # )


def my_greedy_search(
    self,
    input_ids: torch.LongTensor,
    logits_processor: Optional[LogitsProcessorList] = None,
    stopping_criteria: Optional[StoppingCriteriaList] = None,
    max_length: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[Union[int, List[int]]] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_scores: Optional[bool] = None,
    return_dict_in_generate: Optional[bool] = None,
    synced_gpus: bool = False,
    streamer: Optional["BaseStreamer"] = None,
    **model_kwargs,
) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
    r"""
    Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
    used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

    <Tip warning={true}>

    In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
    instead. For an overview of generation strategies and code examples, check the [following
    guide](../generation_strategies).

    </Tip>


    Parameters:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        logits_processor (`LogitsProcessorList`, *optional*):
            An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
            used to modify the prediction scores of the language modeling head applied at each generation step.
        stopping_criteria (`StoppingCriteriaList`, *optional*):
            An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
            used to tell if the generation loop should stop.

        max_length (`int`, *optional*, defaults to 20):
            **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
            tokens. The maximum length of the sequence to be generated.
        pad_token_id (`int`, *optional*):
            The id of the *padding* token.
        eos_token_id (`Union[int, List[int]]`, *optional*):
            The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
        output_attentions (`bool`, *optional*, defaults to `False`):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more details.
        output_hidden_states (`bool`, *optional*, defaults to `False`):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
            for more details.
        output_scores (`bool`, *optional*, defaults to `False`):
            Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
        return_dict_in_generate (`bool`, *optional*, defaults to `False`):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        synced_gpus (`bool`, *optional*, defaults to `False`):
            Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
        streamer (`BaseStreamer`, *optional*):
            Streamer object that will be used to stream the generated sequences. Generated tokens are passed
            through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
        model_kwargs:
            Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
            If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

    Return:
        [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or
        `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
        [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
        `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
        `model.config.is_encoder_decoder=True`.

    Examples:

    ```python
    >>> from transformers import (
    ...     AutoTokenizer,
    ...     AutoModelForCausalLM,
    ...     LogitsProcessorList,
    ...     MinLengthLogitsProcessor,
    ...     StoppingCriteriaList,
    ...     MaxLengthCriteria,
    ... )

    >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
    >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

    >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
    >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

    >>> input_prompt = "It might be possible to"
    >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

    >>> # instantiate logits processors
    >>> logits_processor = LogitsProcessorList(
    ...     [
    ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
    ...     ]
    ... )
    >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

    >>> outputs = model.greedy_search(
    ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
    ... )

    >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
    ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
    ```"""
    # init values
    logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    if max_length is not None:
        warnings.warn(
            "`max_length` is deprecated in this function, use"
            " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
            UserWarning,
        )
        stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
    pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
    eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
    if isinstance(eos_token_id, int):
        eos_token_id = [eos_token_id]
    eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
    output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
    output_attentions = (
        output_attentions if output_attentions is not None else self.generation_config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
    )
    return_dict_in_generate = (
        return_dict_in_generate
        if return_dict_in_generate is not None
        else self.generation_config.return_dict_in_generate
    )

    # init attention / hidden states / scores tuples
    scores = () if (return_dict_in_generate and output_scores) else None
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
    unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

    this_peer_finished = False  # used by synced_gpus only



    decay_factor = getattr(self, "decay_factor", 0.5)
    penalty_step = getattr(self, "penalty_step", 3)
    generation_state = {
        "mode": None,                     # None / 'css' / 'html'
        "css_buffer": "",
        "html_buffer": "",
        "css_counts": {},                 # Global CSS rule counts
        "html_counts": {},                # Global HTML element counts
        "lookbehind_buffer": "",
        "css_penalty_steps_remaining": 0,
        "html_penalty_steps_remaining": 0,
        "content_penalty_steps_remaining": 0,
        "css_penalty_factor": 1.0,        # Default no penalty
        "html_penalty_factor": 1.0,       
        "content_penalty_factor": 1.0,
        "prev_len": input_ids.shape[1],   
        "css_penalty_trigger_count": {},  # Trigger counts for each CSS signature
        "html_penalty_trigger_count": {}, # Trigger counts for each HTML key
        "content_penalty_trigger_count": 0, # Content repetition trigger count
        "html_seen": False,               # Whether <html> has already appeared
        "force_suffix_ids": [],           # Forced token sequence (backup)
        "force_suffix_pos": 0,            # Position of forced token output
    }


    while True:
        if synced_gpus:
            # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
            # The following logic allows an early break if all peers finished generating their sequence
            this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
            # send 0.0 if we finished, 1.0 otherwise
            dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
            # did all peers finish? the reduced sum will be 0.0 then
            if this_peer_finished_flag.item() == 0.0:
                break

        # prepare model inputs
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # forward pass to get next token
        outputs = self(
            **model_inputs,
            return_dict=True,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if synced_gpus and this_peer_finished:
            continue  # don't waste resources running the code we don't need

        next_token_logits = outputs.logits[:, -1, :]

        # pre-process distribution
        next_tokens_scores = logits_processor(input_ids, next_token_logits)

        # Store scores, attentions and hidden_states when required
        if return_dict_in_generate:
            if output_scores:
                scores += (next_tokens_scores,)
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



        # penalty
        if hasattr(self, "tokenizer"):
            tokenizer = self.tokenizer

            def tokens_to_text(tokens):
                return "".join(tok.replace("▁", " ") for tok in tokens)

            # Extract new tokens since last step
            new_tokens = input_ids[0][generation_state.get("prev_len", 0):]
            generation_state["prev_len"] = len(input_ids[0])
            tokens = tokenizer.convert_ids_to_tokens(new_tokens)
            new_text = tokens_to_text(tokens)

            # Update lookbehind buffer (limit size to 500 chars)
            lookbehind_limit = 500
            generation_state["lookbehind_buffer"] = (
                generation_state["lookbehind_buffer"] + new_text
            )[-lookbehind_limit:]
            combined_text = generation_state["lookbehind_buffer"]

            # === Detect CSS mode ===
            if re.search(r"<\s*style\b", combined_text, re.IGNORECASE) and generation_state.get("mode") != "css":
                generation_state["mode"] = "css"
                generation_state["css_buffer"] = ""
                generation_state["css_wait_for_tag_close"] = True
                generation_state["lookbehind_buffer"] = ""

            # === Detect HTML mode ===
            if re.search(r"<\s*body\b", combined_text, re.IGNORECASE) and generation_state.get("mode") != "html":
                generation_state["mode"] = "html"
                generation_state["html_buffer"] = ""
                generation_state["html_wait_for_tag_close"] = True
                generation_state["lookbehind_buffer"] = ""
                generation_state["html_seen"] = True 

            # === CSS buffer incremental parsing ===
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

                # Apply penalty for CSS repetition
                if generation_state.get("css_penalty_steps_remaining", 0) > 0:
                    top1_val, top1_idx = torch.max(next_tokens_scores, dim=-1)
                    factor = generation_state.get("css_penalty_factor", decay_factor)
                    next_tokens_scores[0, top1_idx] *= factor
                    generation_state["css_penalty_steps_remaining"] -= 1
                    # print(f"[Penalty-Step] CSS applying factor={factor:.3f}, "
                    #     f"remaining={generation_state['css_penalty_steps_remaining']}")

                # Exit CSS mode when </style> is detected
                if re.search(r"</\s*style\s*>", combined_text, re.IGNORECASE):
                    generation_state["mode"] = None
                    generation_state["lookbehind_buffer"] = ""
                    generation_state["css_wait_for_tag_close"] = False

            # === HTML buffer incremental parsing ===
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

                # Apply penalty for HTML repetition
                if generation_state.get("html_penalty_steps_remaining", 0) > 0:
                    top1_val, top1_idx = torch.max(next_tokens_scores, dim=-1)
                    factor = generation_state.get("html_penalty_factor", decay_factor)
                    next_tokens_scores[0, top1_idx] *= factor
                    generation_state["html_penalty_steps_remaining"] -= 1
                    # print(f"[Penalty-Step] HTML applying factor={factor:.3f}, "
                    #     f"remaining={generation_state['html_penalty_steps_remaining']}")

                # Exit HTML mode when </body> is detected
                if re.search(r"</\s*body\s*>", combined_text, re.IGNORECASE):
                    generation_state["mode"] = None
                    generation_state["lookbehind_buffer"] = ""
                    generation_state["html_wait_for_tag_close"] = False

            # === Content repetition detection and penalty ===
            update_content_repetition_state(
                combined_text, generation_state, decay_factor=decay_factor, extra_penalty_steps=penalty_step
            )
            steps = generation_state.get("content_penalty_steps_remaining", 0)

            if steps == -1:  # Continuous repetition
                top1_val, top1_idx = torch.max(next_tokens_scores, dim=-1)
                factor = generation_state.get("content_penalty_factor", decay_factor)
                next_tokens_scores[0, top1_idx] *= factor
                # print(f"[Penalty-Step] content (repeating) factor={factor:.3f}")

            elif steps > 0:  # Cooldown period
                top1_val, top1_idx = torch.max(next_tokens_scores, dim=-1)
                factor = generation_state.get("content_penalty_factor", decay_factor)
                next_tokens_scores[0, top1_idx] *= factor
                generation_state["content_penalty_steps_remaining"] -= 1
                remaining = generation_state["content_penalty_steps_remaining"]
                # print(f"[Penalty-Step] content (cooldown) factor={factor:.3f}, remaining={remaining}")
                if remaining == 0:
                    # End cooldown, reset content penalty
                    generation_state["content_penalty_factor"] = 1.0
                    generation_state["content_penalty_trigger_count"] = 0
                    # print("[ContentRepeat] cooldown finished -> penalty cleared")

        # --- Argmax decoding step ---
        next_tokens = torch.argmax(next_tokens_scores, dim=-1)

        # --- Force <html> output when EOS is about to be generated and <html> is missing ---
        if not generation_state.get("html_seen", False):
            eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
            next_token_id = torch.argmax(next_tokens_scores, dim=-1).item()

            # Initialize forced sequence only when EOS is predicted
            if next_token_id in eos_ids and not generation_state.get("force_suffix_ids"):
                generation_state["force_suffix_ids"] = self.tokenizer.encode("```html", add_special_tokens=False)
                generation_state["force_suffix_pos"] = 0

            # If already in forced mode, output forced tokens step by step
            if generation_state.get("force_suffix_ids"):
                # print("[ForceHTML] Triggered forced <html> insertion")
                fp_ids = generation_state["force_suffix_ids"]
                fp_pos = generation_state["force_suffix_pos"]

                if fp_pos < len(fp_ids):
                    forced_id = fp_ids[fp_pos]
                    next_tokens = torch.full_like(next_tokens, forced_id)
                    generation_state["force_suffix_pos"] = fp_pos + 1

                # Once forced sequence is done, mark <html> as seen and clear state
                if generation_state["force_suffix_pos"] >= len(fp_ids):
                    generation_state["html_seen"] = True
                    generation_state.pop("force_suffix_ids", None)
                    generation_state.pop("force_suffix_pos", None)


        # finished sentences should have their next token be a padding token
        if eos_token_id is not None:
            if pad_token_id is None:
                raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

        # update generated ids, model inputs, and length for next step
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        if streamer is not None:
            streamer.put(next_tokens.cpu())
        model_kwargs = self._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
        )

        # if eos_token was found in one sentence, set sentence to finished
        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                this_peer_finished = True

        # stop if we exceed the maximum length
        if stopping_criteria(input_ids, scores):
            this_peer_finished = True

        if this_peer_finished and not synced_gpus:
            break

    if streamer is not None:
        streamer.end()

    if return_dict_in_generate:
        if self.config.is_encoder_decoder:
            return GenerateEncoderDecoderOutput(
                sequences=input_ids,
                scores=scores,
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
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
                past_key_values=model_kwargs.get("past_key_values"),
            )
    else:
        return input_ids