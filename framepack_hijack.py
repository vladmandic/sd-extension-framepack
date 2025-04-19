def set_progress_bar_config(bar_format: str = None, ncols: int = 80, colour: str = None):
    import functools
    from tqdm.auto import trange as trange_orig
    from diffusers_helper.k_diffusion import uni_pc_fm
    uni_pc_fm.trange = functools.partial(trange_orig, bar_format=bar_format, ncols=ncols, colour=colour)
