# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
from climax.arch import ClimaX

class RegionalClimaX(ClimaX):
    def __init__(self, default_vars, img_size=..., patch_size=2, embed_dim=1024, depth=8, decoder_depth=2, num_heads=16, mlp_ratio=4, drop_path=0.1, drop_rate=0.1):
        super().__init__(default_vars, img_size, patch_size, embed_dim, depth, decoder_depth, num_heads, mlp_ratio, drop_path, drop_rate)
        # self.encoder.gradient_checkpointing = True

    def forward_encoder(self, x: torch.Tensor, lead_times: torch.Tensor, variables, region_info):
        # x: `[B, V, H, W]` shape.

        if isinstance(variables, list):
            variables = tuple(variables)

        # tokenize each variable separately
        embeds = []
        var_ids = self.get_var_ids(variables, x.device)
        for i in range(len(var_ids)):
            id = var_ids[i]
            embeds.append(self.token_embeds[id](x[:, i : i + 1]))
        x = torch.stack(embeds, dim=1)  # B, V, L, D

        # add variable embedding
        var_embed = self.get_var_emb(self.var_embed, variables)
        x = x + var_embed.unsqueeze(2)  # B, V, L, D

        # get the patch ids corresponding to the region
        min_h, max_h = region_info["min_h"], region_info["max_h"]
        min_w, max_w = region_info["min_w"], region_info["max_w"]

        # Assume input was [B, V, H, W] and you patchify it by patch_size x patch_size
        H, W = self.img_size  # or x.shape[-2:] if available
        ph = H // self.patch_size
        pw = W // self.patch_size

        # Patchify image coordinates → token ids
        patch_ids = []
        for i in range(min_h // self.patch_size, (max_h + 1) // self.patch_size):
            for j in range(min_w // self.patch_size, (max_w + 1) // self.patch_size):
                patch_id = i * pw + j
                patch_ids.append(patch_id)

        region_patch_ids = torch.tensor(patch_ids, device=x.device)

        x = x[:, :, region_patch_ids, :]

        # variable aggregation
        x = self.aggregate_variables(x)  # B, L, D

        # add pos embedding
        x = x + self.pos_embed[:, region_patch_ids, :]

        # add lead time embedding
        lead_time_emb = self.lead_time_embed(lead_times.unsqueeze(-1))  # B, D
        lead_time_emb = lead_time_emb.unsqueeze(1)
        x = x + lead_time_emb  # B, L, D

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, x, y, lead_times, variables, out_variables, metric, lat, region_info):
        """Forward pass through the model.

        Args:
            x: `[B, Vi, H, W]` shape. Input weather/climate variables
            y: `[B, Vo, H, W]` shape. Target weather/climate variables
            lead_times: `[B]` shape. Forecasting lead times of each element of the batch.
            region_info: Containing the region's information

        Returns:
            loss (list): Different metrics.
            preds (torch.Tensor): `[B, Vo, H, W]` shape. Predicted weather/climate variables.
        """
        out_transformers = self.forward_encoder(x, lead_times, variables, region_info)  # B, L, D
        preds = self.head(out_transformers)  # B, L, V*p*p

        min_h, max_h = region_info['min_h'], region_info['max_h']
        min_w, max_w = region_info['min_w'], region_info['max_w']
        preds = self.unpatchify(preds, h = max_h - min_h + 1, w = max_w - min_w + 1)
        out_var_ids = self.get_var_ids(tuple(out_variables), preds.device)
        preds = preds[:, out_var_ids]

        y = y[:, :, min_h:max_h+1, min_w:max_w+1]
        lat = lat[min_h:max_h+1]

        if metric is None:
            loss = None
        else:
            loss = [m(preds, y, out_variables, lat) for m in metric]

        return loss, preds

def evaluate(self, x, y, lead_times, variables, out_variables, transform, metrics, lat, clim, log_postfix, region_info):
    _, preds = self.forward(x, y, lead_times, variables, out_variables, metric=None, lat=lat, region_info=region_info)

    min_h, max_h = region_info['min_h'], region_info['max_h']
    min_w, max_w = region_info['min_w'], region_info['max_w']
    y = y[:, :, min_h:max_h+1, min_w:max_w+1]
    lat = lat[min_h:max_h+1]
    clim = clim[:, min_h:max_h+1, min_w:max_w+1]

    # Ensure lat and clim are tensors on the same device as preds
    device = preds.device

    if not isinstance(lat, torch.Tensor):
        lat = torch.tensor(lat, device=device)
    else:
        lat = lat.to(device)

    if not isinstance(clim, torch.Tensor):
        clim = torch.tensor(clim, device=device)
    else:
        clim = clim.to(device)

    # Now metrics can safely use lat and clim on correct device
    return [m(preds, y, transform, out_variables, lat, clim, log_postfix) for m in metrics]