import torch
import torch.nn as nn


class CLIPVisionTower_EfficientUICoder(nn.Module):
    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            raise NotImplementedError("Batch list mode not supported in this version.")

        # Dense labels: List[(H, W)] for each image
        dense_labels = self.vision_tower._info.get("dense_labels", None)
        token_selection_ratio = float(self.vision_tower._info.get("token_selection_ratio", 0))
        assert dense_labels is not None, "dense_labels must be provided in vision_tower._info"

        # Forward pass through vision tower (with hidden states + attentions)
        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True
        )

        attn_weights = image_forward_outs.attentions[-2]      # [B, Heads, Tokens, Tokens]
        hidden_states = image_forward_outs.hidden_states[-2]  # [B, Tokens, D]
        # print(f"attn_weights: {attn_weights.size()}, hidden_states: {hidden_states.size()}, token_selection_ratio={token_selection_ratio}")

        B, T, D = hidden_states.shape
        all_indices_batch = []

        BASE_PATCH = 576
        target_replace = int(round(BASE_PATCH * token_selection_ratio))

        for b in range(B):
            labels = dense_labels[b]  # (H, W) numpy
            labels_flat = torch.as_tensor(labels.flatten(), device=hidden_states.device, dtype=torch.int32)

            # --- Label masks ---
            comp_mask = (labels_flat == 1)      # component / connection foreground
            text_mask = (labels_flat == 0)      # text tokens
            bg_mask   = (labels_flat == -1)     # real background only
            # padding (-2) is ignored completely (not included in any mask)

            # --- Indices (+1 to skip CLS=0) ---
            comp_indices = torch.nonzero(comp_mask, as_tuple=False).squeeze(1) + 1
            text_indices = torch.nonzero(text_mask, as_tuple=False).squeeze(1) + 1
            bg_indices   = torch.nonzero(bg_mask,   as_tuple=False).squeeze(1) + 1

            c_count = comp_indices.numel() + text_indices.numel()  # total count of labels 0/1
            b_count = bg_indices.numel()                           # total count of background (-1)
            a = min(target_replace, c_count, b_count)

            def attn_score_for(indices: torch.Tensor):
                """Compute attention scores for given token indices (sum across heads from CLS token)."""
                if indices.numel() == 0:
                    return None
                return attn_weights[b, :, 0, indices].sum(dim=0)

            if a == 0:
                # No replacement needed: keep all 0/1 tokens, drop all background and padding
                keep_nonbg = (
                    torch.cat([comp_indices, text_indices])
                    if c_count > 0 else torch.empty(0, dtype=torch.long, device=hidden_states.device)
                )
                final_indices = torch.cat([
                    torch.tensor([0], device=hidden_states.device),  # always keep CLS
                    torch.sort(keep_nonbg).values
                ])
                all_indices_batch.append(final_indices)
                continue

            # --- Step 1: prune 'a' lowest-attention tokens from label==1 (components/connections) ---
            removed_comp = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            k1 = min(a, comp_indices.numel())
            if k1 > 0:
                scores_comp = attn_score_for(comp_indices)  # [Nc]
                prune_idx_1 = torch.topk(scores_comp, k1, largest=False).indices
                removed_comp = comp_indices[prune_idx_1]
            remaining = a - removed_comp.numel()

            # --- Step 2: if still remaining, prune lowest-attention tokens from label==0 (text) ---
            removed_text = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            if remaining > 0 and text_indices.numel() > 0:
                k2 = min(remaining, text_indices.numel())
                if k2 > 0:
                    scores_text = attn_score_for(text_indices)  # [N0]
                    prune_idx_2 = torch.topk(scores_text, k2, largest=False).indices
                    removed_text = text_indices[prune_idx_2]

            removed_total = removed_comp.numel() + removed_text.numel()

            # --- Step 3: add the same number of background (-1) tokens with highest attention ---
            added_bg = torch.empty(0, dtype=torch.long, device=hidden_states.device)
            if removed_total > 0 and bg_indices.numel() > 0:
                scores_bg = attn_score_for(bg_indices)  # [Nb]
                k_bg = min(removed_total, bg_indices.numel())
                if k_bg > 0:
                    top_bg = torch.topk(scores_bg, k_bg, largest=True).indices
                    added_bg = bg_indices[top_bg]

            # --- Step 4: construct final keep set ---
            # keep = (all 0/1 tokens) - (removed_comp ∪ removed_text) + added_bg
            drop = (
                torch.cat([removed_comp, removed_text])
                if removed_total > 0 else torch.empty(0, dtype=torch.long, device=hidden_states.device)
            )
            drop_set = set(drop.tolist())

            all_nonbg = torch.cat([comp_indices, text_indices])
            kept_nonbg = [idx for idx in all_nonbg.tolist() if idx not in drop_set]
            kept_nonbg = torch.as_tensor(kept_nonbg, device=hidden_states.device, dtype=torch.long)

            keep_indices_parts = [torch.tensor([0], device=hidden_states.device)]  # always keep CLS
            if kept_nonbg.numel() > 0:
                keep_indices_parts.append(kept_nonbg)
            if added_bg.numel() > 0:
                keep_indices_parts.append(added_bg)

            final_indices = torch.cat(keep_indices_parts)
            final_indices = torch.sort(final_indices).values
            all_indices_batch.append(final_indices)

        hidden_states_save = hidden_states
        return hidden_states_save, all_indices_batch
