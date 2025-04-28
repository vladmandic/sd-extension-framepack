import os
import sys
import random
import threading
import gradio as gr
from modules import shared, processing, timer, paths, extra_networks, progress
import framepack_install
import framepack_load
import framepack_worker
import framepack_hijack


tmp_dir = os.path.join(paths.data_path, 'tmp', 'framepack')
git_dir = os.path.join(os.path.dirname(__file__), 'framepack')
git_repo = 'https://github.com/lllyasviel/framepack'
git_commit = 'a875c8b58691c7ba98f93ad6623994a4e69df8ef'
queue_lock = threading.Lock()


def check_av():
    try:
        import av
    except Exception as e:
        shared.log.error(f'av package: {e}')
        return False
    return av


def get_codecs():
    av = check_av()
    if av is None:
        return []
    codecs = []
    for codec in av.codecs_available:
        try:
            c = av.Codec(codec, mode='w')
            if c.type == 'video' and c.is_encoder and len(c.video_formats) > 0:
                if not any(c.name == ca.name for ca in codecs):
                    codecs.append(c)
        except Exception:
            pass
    hw_codecs = [c for c in codecs if (c.capabilities & 0x40000 > 0) or (c.capabilities & 0x80000 > 0)]
    sw_codecs = [c for c in codecs if c not in hw_codecs]
    shared.log.debug(f'Video codecs: hardware={len(hw_codecs)} software={len(sw_codecs)}')
    # for c in hw_codecs:
    #     shared.log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=True')
    # for c in sw_codecs:
    #     shared.log.trace(f'codec={c.name} cname="{c.canonical_name}" decs="{c.long_name}" intra={c.intra_only} lossy={c.lossy} lossless={c.lossless} capabilities={c.capabilities} hw=False')
    return ['none'] + [c.name for c in hw_codecs + sw_codecs]


def prepare_image(image, resolution):
    from diffusers_helper.utils import resize_and_center_crop
    buckets = [
        (416, 960), (448, 864), (480, 832), (512, 768), (544, 704), (576, 672), (608, 640),
        (640, 608), (672, 576), (704, 544), (768, 512), (832, 480), (864, 448), (960, 416),
    ]
    h, w, _c = image.shape
    min_metric = float('inf')
    scale_factor = resolution / 640.0
    scaled_h, scaled_w = h, w
    for (bucket_h, bucket_w) in buckets:
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            scaled_h = round(bucket_h * scale_factor / 16) * 16
            scaled_w = round(bucket_w * scale_factor / 16) * 16

    image = resize_and_center_crop(image, target_height=scaled_h, target_width=scaled_w)
    h0, w0, _c = image.shape
    shared.log.debug(f'FramePack prepare: input="{w}x{h}" resized="{w0}x{h0}" resolution={resolution} scale={scale_factor}')
    return image


def prepare_prompt(p, system_prompt):
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    shared.prompt_styles.apply_styles_to_extra(p)
    p.prompts, p.network_data = extra_networks.parse_prompts([p.prompt])
    p.prompt = p.prompts[0]
    extra_networks.activate(p)
    framepack_hijack.set_prompt_template(p.prompt, system_prompt)


def load_model(attention):
    if shared.sd_model_type != 'hunyuanvideo':
        yield gr.update(), gr.update(), 'Verifying FramePack'
        framepack_install.install_requirements(attention)
        framepack_install.git_clone(git_repo=git_repo, git_dir=git_dir, tmp_dir=tmp_dir)
        framepack_install.git_update(git_dir=git_dir, git_commit=git_commit)
        sys.path.append(git_dir)
        framepack_hijack.set_progress_bar_config()
        yield gr.update(), gr.update(), 'Model loading...', ''
        loaded = framepack_load.load_model()
        if loaded:
            return gr.update(), gr.update(), 'Model loaded'
        else:
            return gr.update(), gr.update(), 'Model load failed'


def unload_model():
    shared.log.debug('FramePack unload')
    framepack_load.unload_model()
    return gr.update(), gr.update(), 'Model unloaded'


def run_framepack(task_id, init_image, end_image, start_weight, end_weight, vision_weight, prompt, system_prompt, section_prompt, negative_prompt, styles, seed, resolution, duration, latent_ws, steps, cfg_scale, cfg_distilled, cfg_rescale, shift, use_teacache, use_cfgzero, use_preview, mp4_fps, mp4_codec, mp4_sf, mp4_video, mp4_frames, mp4_opt, mp4_ext, mp4_interpolate, attention, vae_type):
    if init_image is None:
        shared.log.error('FramePack: no input image')
        return gr.update(), gr.update(), 'No input image'
    av = check_av()
    if av is None:
        return gr.update(), gr.update(), 'AV package not installed'

    progress.add_task_to_queue(task_id)
    with queue_lock:
        progress.start_task(task_id)

        yield from load_model(attention)
        if shared.sd_model_type != 'hunyuanvideo':
            progress.finish_task(task_id)
            return gr.update(), gr.update(), 'Model load failed'

        yield gr.update(), gr.update(), 'Generate starting...'
        from diffusers_helper.thread_utils import AsyncStream, async_run
        framepack_worker.stream = AsyncStream()

        if seed is None or seed == '' or seed == -1:
            random.seed()
            seed = random.randrange(4294967294)
        seed = int(seed)
        mode = 'i2v' if end_image is None else 'flf2v'
        num_sections = len(framepack_worker.get_latent_paddings(mp4_fps, mp4_interpolate, latent_ws, duration))
        num_frames = (latent_ws * 4 - 3) * num_sections + 1
        shared.log.info(f'FramePack start: mode={mode} frames={num_frames} sections={num_sections} resolution={resolution} seed={seed} duration={duration} teacache={use_teacache} cfgzero={use_cfgzero}')
        init_image = prepare_image(init_image, resolution)
        if end_image is not None:
            end_image = prepare_image(end_image, resolution)
        w, h, _c = init_image.shape
        p = processing.StableDiffusionProcessingVideo(
            sd_model=shared.sd_model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            styles=styles,
            steps=steps,
            seed=seed,
            width=w,
            height=h,
        )
        prepare_prompt(p, system_prompt)

        async_run(
            framepack_worker.worker,
            init_image, end_image,
            start_weight, end_weight, vision_weight,
            p.prompt, section_prompt, p.negative_prompt,
            seed,
            duration,
            latent_ws,
            p.steps,
            cfg_scale, cfg_distilled, cfg_rescale,
            shift,
            use_teacache, use_cfgzero, use_preview,
            mp4_fps, mp4_codec, mp4_sf, mp4_video, mp4_frames, mp4_opt, mp4_ext, mp4_interpolate,
            vae_type,
        )

        output_filename = None
        while True:
            flag, data = framepack_worker.stream.output_queue.next()
            if flag == 'file':
                output_filename = data
                yield output_filename, gr.update(), gr.update()
            if flag == 'progress':
                preview, text = data
                summary = timer.process.summary(min_time=0.25, total=False).replace('=', ' ')
                memory = shared.mem_mon.summary()
                stats = f"<div class='performance'><p>{summary} {memory}</p></div>"
                yield gr.update(), gr.update(value=preview), f'{text} {stats}'
            if flag == 'end':
                yield output_filename, gr.update(value=None), gr.update()
                break

        progress.finish_task(task_id)
    return gr.update(), gr.update(), 'Generate finished'
