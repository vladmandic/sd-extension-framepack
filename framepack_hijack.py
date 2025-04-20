def set_progress_bar_config(bar_format: str = None, ncols: int = 80, colour: str = None):
    import functools
    from tqdm.auto import trange as trange_orig
    from diffusers_helper.k_diffusion import uni_pc_fm
    uni_pc_fm.trange = functools.partial(trange_orig, bar_format=bar_format, ncols=ncols, colour=colour)


def set_prompt_template():
    from diffusers_helper import hunyuan
    hunyuan.DEFAULT_PROMPT_TEMPLATE = {
        "template": (
            "<|start_header_id|>system<|end_header_id|>\nDescribe the video by detailing the following aspects: \n"
            "1. main content and theme of the video\n"
            "2. color, shape, size, quantity, text, and spatial relationships of the objects\n"
            "3. actions, events, behaviors, temporal relationships, physical movement, and changes of the objects\n"
            "4. background environment, light, style, atmosphere\n"
            "5. camera angles, movements, transitions used in the video\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
        ),
        "crop_start": 93,
    }
