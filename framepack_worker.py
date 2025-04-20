import os
import time
import datetime
import torch
import torchvision
import numpy as np
import einops
from modules import shared, errors ,devices, sd_models, timer, memstats


stream = None # AsyncStream


def save_video(pixels, mp4_fps, mp4_codec, mp4_opt, mp4_ext):
    if pixels is None:
        return
    t_save = time.time()
    try:
        n, _c, t, h, w = pixels.shape
        x = torch.clamp(pixels.float(), -1., 1.) * 127.5 + 127.5
        x = x.detach().cpu().to(torch.uint8)
        x = einops.rearrange(x, '(m n) c t h w -> t (m h) (n w) c', n=n)
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        output_filename = os.path.join(shared.opts.outdir_video, f'{timestamp}-{mp4_codec}-f{t}.{mp4_ext}')
        options = {}
        for option in [option.strip() for option in mp4_opt.split(',')]:
            if '=' in option:
                key, value = option.split('=', 1)
            elif ':' in option:
                key, value = option.split(':', 1)
            else:
                continue
            options[key.strip()] = value.strip()
        torchvision.io.write_video(output_filename, video_array=x, fps=mp4_fps, video_codec=mp4_codec, options=options)
        timer.process.add('save', time.time()-t_save)
        shared.log.info(f'FramePack video: file="{output_filename}" codec={mp4_codec} frames={t} width={w} height={h} fps={mp4_fps} options={options}')
        stream.output_queue.push(('file', output_filename))
    except Exception as e:
        shared.log.error(f'FramePack video: {e}')
        errors.display(e, 'FramePack video')


@torch.no_grad()
def worker(input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, shift, gpu_memory_preservation, offload_native, use_teacache, mp4_fps, mp4_codec, mp4_opt, mp4_ext):
    timer.process.reset()
    memstats.reset_stats()
    if stream is None or shared.state.interrupted or shared.state.skipped:
        shared.log.error('FramePack: stream is None')
        stream.output_queue.push(('end', None))
        return

    from diffusers_helper import hunyuan
    from diffusers_helper import utils
    from diffusers_helper import memory
    from diffusers_helper import clip_vision
    from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan

    calculated_frames = 0
    total_latent_sections = (total_second_length * mp4_fps) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))
    shared.state.begin('Video')
    shared.state.job_count = 1

    try:
        text_encoder = shared.sd_model.text_encoder
        text_encoder_2 = shared.sd_model.text_encoder_2
        tokenizer = shared.sd_model.tokenizer
        tokenizer_2 = shared.sd_model.tokenizer_2
        vae = shared.sd_model.vae
        feature_extractor = shared.sd_model.feature_extractor
        image_encoder = shared.sd_model.image_processor
        transformer = shared.sd_model.transformer

        shared.state.textinfo = 'Start'
        stream.output_queue.push(('progress', (None, 'Starting..')))
        if offload_native:
            sd_models.apply_balanced_offload(shared.sd_model)
        else:
            memory.unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)

        t0 = time.time()

        # Text encoding
        shared.state.textinfo = 'Text encode'
        stream.output_queue.push(('progress', (None, 'Text encoding...')))
        if offload_native:
            sd_models.apply_balanced_offload(shared.sd_model)
            sd_models.move_model(text_encoder, devices.device, force=True) # required as hunyuan.encode_prompt_conds checks device before calling model
            sd_models.move_model(text_encoder_2, devices.device, force=True)
        else:
            memory.fake_diffusers_current_device(text_encoder, devices.device)
            memory.load_model_as_complete(text_encoder_2, target_device=devices.device)
        llama_vec, clip_l_pooler = hunyuan.encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = hunyuan.encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
        llama_vec, llama_attention_mask = utils.crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = utils.crop_or_pad_yield_mask(llama_vec_n, length=512)
        t1 = time.time()
        timer.process.add('prompt', t1-t0)

        # Processing input image
        shared.state.textinfo = 'Image process'
        stream.output_queue.push(('progress', (None, 'Image processing...')))
        height, width, _C = input_image.shape
        input_image_pt = torch.from_numpy(input_image).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

        # VAE encoding
        stream.output_queue.push(('progress', (None, 'VAE encoding...')))
        if offload_native:
            sd_models.apply_balanced_offload(shared.sd_model)
            sd_models.move_model(vae, devices.device, force=True)
        else:
            memory.load_model_as_complete(vae, target_device=devices.device)
        start_latent = hunyuan.vae_encode(input_image_pt, vae)
        t2 = time.time()
        timer.process.add('encode', t2-t1)

        # CLIP Vision
        shared.state.textinfo = 'Vision encode'
        stream.output_queue.push(('progress', (None, 'Vision encoding...')))
        if offload_native:
            sd_models.apply_balanced_offload(shared.sd_model)
            sd_models.move_model(feature_extractor, devices.device, force=True)
            sd_models.move_model(image_encoder, devices.device, force=True)
        else:
            memory.load_model_as_complete(image_encoder, target_device=devices.device)
        image_encoder_output = clip_vision.hf_clip_vision_encode(input_image, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        t3 = time.time()
        timer.process.add('vision', t3-t2)

        # Dtype
        llama_vec = llama_vec.to(transformer.dtype)
        llama_vec_n = llama_vec_n.to(transformer.dtype)
        clip_l_pooler = clip_l_pooler.to(transformer.dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)

        # Sampling
        shared.state.textinfo = 'Sample'
        stream.output_queue.push(('progress', (None, 'Start sampling...')))
        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3
        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0
        latent_paddings = reversed(range(total_latent_sections))
        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        lattent_padding_loop = 0
        lattent_padding_loops = total_latent_sections if total_latent_sections <= 4 else len(latent_paddings)
        for latent_padding in latent_paddings:
            lattent_padding_loop += 1
            shared.log.debug(f'FramePack: op=sample section={lattent_padding_loop}/{lattent_padding_loops} frames={total_generated_latent_frames}/{num_frames}')
            is_last_section = latent_padding == 0
            latent_padding_size = latent_padding * latent_window_size
            if stream.input_queue.top() == 'end' or shared.state.interrupted or shared.state.skipped:
                stream.output_queue.push(('end', None))
                return
            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, _blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            if offload_native:
                sd_models.apply_balanced_offload(shared.sd_model)
            else:
                memory.unload_complete_models()
                memory.move_model_to_device_with_memory_preservation(transformer, target_device=devices.device, preserved_memory_gb=gpu_memory_preservation)
            if use_teacache:
                transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
            else:
                transformer.initialize_teacache(enable_teacache=False)

            def callback(d):
                if stream.input_queue.top() == 'end' or shared.state.interrupted or shared.state.skipped:
                    stream.output_queue.push(('progress', (None, 'Interrupted...')))
                    stream.output_queue.push(('end', None))
                    raise AssertionError('Interrupted...')
                if shared.state.paused:
                    shared.log.debug('Sampling paused')
                    while shared.state.paused:
                        if shared.state.interrupted or shared.state.skipped:
                            raise AssertionError('Interrupted...')
                        time.sleep(0.1)
                nonlocal calculated_frames
                t_preview = time.time()
                preview = d['denoised']
                preview = hunyuan.vae_decode_fake(preview)
                preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')
                current_step = d['i'] + 1
                shared.state.textinfo = ''
                shared.state.sampling_step = ((lattent_padding_loop-1) * steps) + current_step # noqa: B023
                shared.state.sampling_steps = steps * lattent_padding_loops
                progress = shared.state.sampling_step / shared.state.sampling_steps
                calculated_frames = int(max(0, total_generated_latent_frames * 4 - 3)) # noqa: B023
                desc = f'Section: {lattent_padding_loop}/{lattent_padding_loops} Step: {current_step}/{steps} Frames: {calculated_frames} Progress: {progress:.2%}' # noqa: B023
                stream.output_queue.push(('progress', (preview, desc)))
                timer.process.add('preview', time.time()-t_preview)
                return

            t_sample = time.time()
            generated_latents = sample_hunyuan(
                transformer=transformer,
                sampler='unipc',
                width=width,
                height=height,
                frames=num_frames,
                real_guidance_scale=cfg,
                distilled_guidance_scale=gs,
                guidance_rescale=rs,
                shift=shift if shift > 0 else None,
                num_inference_steps=steps,
                generator=rnd,
                prompt_embeds=llama_vec,
                prompt_embeds_mask=llama_attention_mask,
                prompt_poolers=clip_l_pooler,
                negative_prompt_embeds=llama_vec_n,
                negative_prompt_embeds_mask=llama_attention_mask_n,
                negative_prompt_poolers=clip_l_pooler_n,
                device=devices.device,
                dtype=devices.dtype,
                image_embeddings=image_encoder_last_hidden_state,
                latent_indices=latent_indices,
                clean_latents=clean_latents,
                clean_latent_indices=clean_latent_indices,
                clean_latents_2x=clean_latents_2x,
                clean_latent_2x_indices=clean_latent_2x_indices,
                clean_latents_4x=clean_latents_4x,
                clean_latent_4x_indices=clean_latent_4x_indices,
                callback=callback,
            )
            timer.process.add('sample', time.time()-t_sample)

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            t_vae = time.time()
            if offload_native:
                sd_models.apply_balanced_offload(shared.sd_model)
                sd_models.move_model(vae, devices.device, force=True)
            else:
                memory.offload_model_from_device_for_memory_preservation(transformer, target_device=devices.device, preserved_memory_gb=8)
                memory.load_model_as_complete(vae, target_device=devices.device)
            if history_pixels is None:
                history_pixels = hunyuan.vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                current_pixels = hunyuan.vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = utils.soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)
            timer.process.add('vae', time.time()-t_vae)

            if is_last_section:
                save_video(history_pixels, mp4_fps, mp4_codec, mp4_opt, mp4_ext)
                break
    except AssertionError:
        shared.log.info('FramePack: interrupted')
        if shared.opts.keep_incomplete:
            save_video(history_pixels, mp4_fps, mp4_codec, mp4_opt, mp4_ext)
    except Exception as e:
        shared.log.error(f'FramePack: {e}')
        errors.display(e, 'FramePack')

    if offload_native:
        sd_models.apply_balanced_offload(shared.sd_model)
    else:
        memory.unload_complete_models()
    stream.output_queue.push(('end', None))
    t1 = time.time()
    shared.log.info(f'Processed: frames={calculated_frames} its={(steps*calculated_frames)/(t1-t0):.2f} time={t1-t0:.2f} timers={timer.process.dct()} memory={memstats.memory_stats()}')
    shared.state.end()
