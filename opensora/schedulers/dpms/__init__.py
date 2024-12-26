from functools import partial

import torch

from opensora.registry import SCHEDULERS

from .dpm_solver import DPMS


@SCHEDULERS.register_module("dpm-solver")
class DPM_SOLVER:
    def __init__(self, num_sampling_steps=None, cfg_scale=4.0):
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        progress=True,
    ):
        if mask is not None:
            print("[WARNING] mask is not supported in dpm-solver, it will be ignored")
        n = len(prompts)
        if additional_args is not None and "positive_embeds" in additional_args and "positive_attention_mask" in additional_args:
            model_args = {"y": additional_args["positive_embeds"], "mask": additional_args["positive_attention_mask"]}
        elif additional_args is not None and "positive_embeds" in additional_args:
            model_args = {"y": additional_args["positive_embeds"]}
        else:
            model_args = text_encoder.encode(prompts)
        y = model_args.pop("y").to(device)
        if additional_args is not None and "negative_embeds" in additional_args:
            null_y = additional_args["negative_embeds"].to(device)
        else:
            null_y = text_encoder.null(n).to(device)
        if additional_args is not None:
            model_args.update({k:v for k,v in additional_args.items() if k!="positive_embeds" and k!="negative_embeds" and k!="positive_attention_mask"})

        dpms = DPMS(
            partial(forward_with_dpmsolver, model),
            condition=y,
            uncondition=null_y,
            cfg_scale=self.cfg_scale,
            model_kwargs=model_args,
        )
        samples = dpms.sample(
            z,
            steps=self.num_sampling_steps,
            order=2,
            skip_type="time_uniform",
            method="multistep",
            progress=progress,
        )
        return samples


def forward_with_dpmsolver(self, x, timestep, y, **kwargs):
    """
    dpm solver donnot need variance prediction
    """
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    model_out = self.forward(x, timestep, y, **kwargs)
    return model_out.chunk(2, dim=1)[0]
