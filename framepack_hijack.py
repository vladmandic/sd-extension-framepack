

def set_progress_bar_config():
    # import functools
    # from tqdm.auto import trange as trange_orig
    # from diffusers_helper.k_diffusion import uni_pc_fm
    # uni_pc_fm.trange = functools.partial(trange_orig, bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m', ncols=80, colour='#327fba')
    # uni_pc_fm.trange = functools.partial(trange_orig, disable=True)
    from diffusers_helper.k_diffusion import uni_pc_fm

    def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'): # pylint: disable=unused-argument
        return uni_pc_fm.FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(noise, sigmas=sigmas, callback=callback, disable_pbar=True)

    uni_pc_fm.sample_unipc = sample_unipc


def set_prompt_template(system_prompt:str=None):
    from diffusers_helper import hunyuan
    if system_prompt is None or len(system_prompt) == 0:
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
    else: # TODO framepack: custom prompt template and calculate crop_start from tokenizer
        hunyuan.DEFAULT_PROMPT_TEMPLATE = {
            "template": (
                f"<|start_header_id|>system<|end_header_id|>\n{system_prompt}\n"
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>{}<|eot_id|>"
            ),
            "crop_start": 0,
        }
