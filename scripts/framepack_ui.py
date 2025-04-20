import os
import sys
import random
import threading
import gradio as gr
import framepack_install
import framepack_load
import framepack_worker
import framepack_hijack
from modules import shared, processing, paths, script_callbacks, extra_networks, progress, ui_sections, ui_common, generation_parameters_copypaste


tmp_dir = os.path.join(paths.data_path, 'tmp', 'framepack')
git_dir = os.path.join(os.path.dirname(__file__), '..', 'framepack')
git_repo = 'https://github.com/lllyasviel/framepack'
git_commit = '743657ef2355920fb2f1f934a34647ccd0f916c7'
queue_lock = threading.Lock()
resolutions = [
    'Auto',
    '416 x 960',
    '448 x 864',
    '480 x 832',
    '512 x 768',
    '544 x 704',
    '576 x 672',
    '608 x 640',
    '640 x 608',
    '672 x 576',
    '704 x 544',
    '768 x 512',
    '832 x 480',
    '864 x 448',
    '960 x 416',
]


def get_codecs():
    try:
        import av
    except Exception as e:
        shared.log.error(f'av package: {e}')
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
    return [c.name for c in hw_codecs + sw_codecs]


def prepare_image(image, resolution):
    from diffusers_helper.utils import resize_and_center_crop
    h, w, _c = image.shape
    if resolution == 'Auto':
        min_metric = float('inf')
        buckets = [map(int, resolution.split(' x ')) for resolution in resolutions[1:]]
        for (bucket_h, bucket_w) in buckets:
            metric = abs(h * bucket_w - w * bucket_h)
            if metric <= min_metric:
                min_metric = metric
                best_h, best_w = bucket_h, bucket_w
    else:
        best_h, best_w = map(int, resolution.split(' x '))
    image = resize_and_center_crop(image, target_width=best_w, target_height=best_h)
    h0, w0, _c = image.shape
    shared.log.debug(f'FramePack: input="{h}x{w}" resized="{h0}x{w0}"')
    return image


def prepare_prompt(p):
    p.prompt = shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)
    p.negative_prompt = shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)
    shared.prompt_styles.apply_styles_to_extra(p)
    p.prompts, p.network_data = extra_networks.parse_prompts([p.prompt])
    p.prompt = p.prompts[0]
    extra_networks.activate(p)


def load_model(offload_native, attention):
    if shared.sd_model_type != 'hunyuanvideo':
        yield gr.update(), gr.update(), 'Installing Framepack'
        framepack_install.install_requirements(attention)
        framepack_install.git_clone(git_repo=git_repo, git_dir=git_dir, tmp_dir=tmp_dir)
        framepack_install.git_update(git_dir=git_dir, git_commit=git_commit)
        sys.path.append(git_dir)
        framepack_hijack.set_progress_bar_config(bar_format='Progress {rate_fmt}{postfix} {bar} {percentage:3.0f}% {n_fmt}/{total_fmt} {elapsed} {remaining} ' + '\x1b[38;5;71m', ncols=80, colour='#327fba')
        yield gr.update(), gr.update(), 'Model loading...', ''
        loaded = framepack_load.load_model(offload_native)
        if loaded:
            return gr.update(), gr.update(), 'Model loaded'
        else:
            return gr.update(), gr.update(), 'Model load failed'


def unload_model():
    shared.log.debug('FramePack unload')
    framepack_load.unload_model()
    return gr.update(), gr.update(), 'Model unloaded'


def run_framepack(task_id, input_image, prompt, negative_prompt, styles, seed, resolution, duration, latent_ws, steps, cfg_scale, cfg_distilled, cfg_rescale, shift, gpu_preserved, offload_native, use_teacache, mp4_fps, mp4_codec, mp4_opt, mp4_ext, mp4_interpolate, attention):
    if input_image is None:
        shared.log.error('FramePack: no input image')
        return gr.update(), gr.update(), 'No input image'

    progress.add_task_to_queue(task_id)
    with queue_lock:
        progress.start_task(task_id)

        yield from load_model(offload_native, attention)
        if shared.sd_model_type != 'hunyuanvideo':
            progress.finish_task(task_id)
            return gr.update(), gr.update(), 'Model load failed'

        yield gr.update(), gr.update(), 'Generate starting...'
        from diffusers_helper.thread_utils import AsyncStream, async_run
        framepack_worker.stream = AsyncStream()

        if seed is None or seed == '' or seed == -1:
            random.seed()
            seed = int(random.randrange(4294967294))
        shared.log.info(f'FramePack start: {task_id} resolution="{resolution}" seed={seed} duration={duration} teacache={use_teacache} fps={mp4_fps} codec={mp4_codec} ext="{mp4_ext}" options="{mp4_opt}"')
        input_image = prepare_image(input_image, resolution)
        w, h, _c = input_image.shape
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
        prepare_prompt(p)

        async_run(
            framepack_worker.worker,
            input_image,
            p.prompt,
            p.negative_prompt,
            p.seed,
            duration,
            latent_ws,
            p.steps,
            cfg_scale,
            cfg_distilled,
            cfg_rescale,
            shift,
            gpu_preserved,
            offload_native,
            use_teacache,
            mp4_fps,
            mp4_codec,
            mp4_opt,
            mp4_ext,
            mp4_interpolate,
        )

        output_filename = None
        while True:
            flag, data = framepack_worker.stream.output_queue.next()
            if flag == 'file':
                output_filename = data
                yield output_filename, gr.update(), gr.update()
            if flag == 'progress':
                preview, desc = data
                yield gr.update(), gr.update(value=preview), desc
            if flag == 'end':
                yield output_filename, gr.update(value=None), gr.update()
                break

        progress.finish_task(task_id)
    return gr.update(), gr.update(), 'Generate finished'


def process_end():
    if framepack_worker.stream is not None:
        framepack_worker.stream.input_queue.push('end')
    return gr.update(), gr.update(), 'Interrupted...'


def create_ui():
    with gr.Blocks(analytics_enabled=False) as ui:
        prompt, styles, negative, generate, _reprocess, paste, _networks_button, _token_counter, _token_button, _token_counter_negative, _token_button_negative = ui_sections.create_toprow(is_img2img=False, id_part="framepack", negative_visible=False, reprocess_visible=False)
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(sources='upload', type="numpy", label="Image", height=512, interactive=True, tool="editor", image_mode='RGB', elem_id="framepack_input_image")
                with gr.Row():
                    btn_load = gr.Button(value="Load model", elem_id="framepack_btn_load", interactive=True)
                    btn_unload = gr.Button(value="Unload model", elem_id="framepack_btn_unload", interactive=True)
                with gr.Row():
                    resolution = gr.Dropdown(label="Resolution", choices=resolutions, value='Auto', type='value', elem_id="framepack_resolution")
                    duration = gr.Slider(label="Video duration", minimum=1, maximum=120, value=4, step=0.1)
                    mp4_fps = gr.Slider(label="FPS", minimum=1, maximum=60, value=24, step=1)
                    mp4_interpolate = gr.Slider(label="Interpolation", minimum=0, maximum=10, value=0, step=1)
                with gr.Accordion(label="Video codec", open=False):
                    with gr.Row():
                        mp4_codec = gr.Dropdown(label="Codec", choices=['libx264'], value='libx264', type='value')
                        ui_common.create_refresh_button(mp4_codec, get_codecs)
                        mp4_ext = gr.Textbox(label="Format", value='mp4', elem_id="framepack_mp4_ext")
                        mp4_opt = gr.Textbox(label="Options", value='crf:16', elem_id="framepack_mp4_ext")
                with gr.Accordion(label="Advanced", open=False):
                    seed = ui_sections.create_seed_inputs('control', reuse_visible=False, subseed_visible=False, accordion=False)[0]
                    latent_ws = gr.Slider(label="Latent window size", minimum=1, maximum=33, value=9, step=1)
                    with gr.Row():
                        steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1)
                        shift = gr.Slider(label="Sampler shift", minimum=0, maximum=10, value=0, step=1)
                    with gr.Row():
                        cfg_scale = gr.Slider(label="CFG scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01)
                        cfg_distilled = gr.Slider(label="Distilled CFG scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01)
                        cfg_rescale = gr.Slider(label="CFG re-scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01)
                with gr.Accordion(label="Processing", open=False):
                    offload_native = gr.Checkbox(label='Native offloading', value=True)
                    use_teacache = gr.Checkbox(label='Enable TeaCache', value=True)
                    attention = gr.Dropdown(label="Attention", choices=['Default', 'Xformers', 'FlashAttention', 'SageAttention'], value='Default', type='value')
                    gpu_preserved = gr.Slider(label="Preserved memory", minimum=6, maximum=128, value=6, step=0.1)
                override_settings = ui_common.create_override_inputs('framepack')

            with gr.Column():
                with gr.Tabs():
                    with gr.TabItem("Video"):
                        result_video = gr.Video(label="Video", autoplay=True, show_share_button=False, height=512, loop=True)
                    with gr.Tab("Preview"):
                        preview_image = gr.Image(label="Current", height=512)
                progress_desc = gr.Markdown('')

            outputs = [result_video, preview_image, progress_desc]
            btn_load.click(fn=load_model, inputs=[offload_native, attention], outputs=outputs)
            btn_unload.click(fn=unload_model, outputs=outputs)
            task_id = gr.Textbox(visible=False, value='')
            generate.click(
                fn=run_framepack,
                _js="submit_framepack",
                inputs=[task_id, input_image, prompt, negative, styles, seed, resolution, duration, latent_ws, steps, cfg_scale, cfg_distilled, cfg_rescale, shift, gpu_preserved, offload_native, use_teacache, mp4_fps, mp4_codec, mp4_opt, mp4_ext, mp4_interpolate, attention],
                outputs=outputs,
            )

            paste_fields = [(prompt, "Prompt")]
            generation_parameters_copypaste.add_paste_fields("video", None, paste_fields, override_settings)
            bindings = generation_parameters_copypaste.ParamBinding(paste_button=paste, tabname="framepack", source_text_component=prompt, source_image_component=None)
            generation_parameters_copypaste.register_paste_params_button(bindings)


    return [(ui, "FramePack", "framepack_tab")]


script_callbacks.on_ui_tabs(create_ui)
