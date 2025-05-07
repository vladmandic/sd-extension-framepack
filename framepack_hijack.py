DEFAULT_PROMPT_TEMPLATE = { # hunyuanvideo reference prompt template
    "template": (
        "<|start_header_id|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
        "1. The main content and theme of the video."
        "2. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
        "3. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
        "4. camera angles, movements, and transitions used in the video:<|eot_id|>"
        "5. background environment, light, style and atmosphere."
        "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
    ),
    "crop_start": 95,
}


def set_progress_bar_config():
    from diffusers_helper.k_diffusion import uni_pc_fm

    def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'): # pylint: disable=unused-argument
        return uni_pc_fm.FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(noise, sigmas=sigmas, callback=callback, disable_pbar=True)

    uni_pc_fm.sample_unipc = sample_unipc


def set_prompt_template(prompt, system_prompt:str=None):
    from modules import shared
    from diffusers_helper import hunyuan
    if system_prompt is None or len(system_prompt) == 0:
        system_prompt = (
            "Describe the video by detailing the following aspects:\n"
            "1. Main content and theme of the video.\n"
            "2. Actions, events, behaviors, temporal relationships, physical movement, and changes of the subjects or objects.\n"
            "3. Camera angles, camera movements, and transitions used in the video.\n"
            "4. Details of the scene and background environment, light, style, and atmosphere.\n"
        )
    # system_prompt = DEFAULT_PROMPT_TEMPLATE["template"]
    inputs = shared.sd_model.tokenizer(system_prompt, max_length=256, truncation=True, return_tensors="pt", return_length=True, return_overflowing_tokens=False, return_attention_mask=False)
    tokens_system = inputs['length'].item() - int(shared.sd_model.tokenizer.bos_token_id is not None) - int(shared.sd_model.tokenizer.eos_token_id is not None)
    inputs = shared.sd_model.tokenizer(prompt, max_length=256, truncation=True, return_tensors="pt", return_length=True, return_overflowing_tokens=False, return_attention_mask=False)
    hunyuan.DEFAULT_PROMPT_TEMPLATE = {
        "template": (
            f"<|start_header_id|>system<|end_header_id|>{system_prompt}\n<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
        ),
        "crop_start": tokens_system,
    }
    # tokens_user = inputs['length'].item() - int(shared.sd_model.tokenizer.bos_token_id is not None) - int(shared.sd_model.tokenizer.eos_token_id is not None)
    # shared.log.trace(f'FramePack prompt: system={tokens_system} user={tokens_user}')
