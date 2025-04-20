from modules import shared, devices, errors, sd_models, model_quant


repo_hunyuan = 'hunyuanvideo-community/HunyuanVideo'
repo_encoder = 'lllyasviel/flux_redux_bfl'
repo_transformer = 'lllyasviel/FramePackI2V_HY'


def load_model(offload_native: bool = True):
    shared.state.begin('Load')
    try:
        import diffusers
        from diffusers import HunyuanVideoImageToVideoPipeline, AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
        from diffusers_helper.memory import DynamicSwapInstaller

        class FramepackHunyuanVideoPipeline(HunyuanVideoImageToVideoPipeline): # inherit and override
            def __init__(
                self,
                text_encoder: LlamaModel,
                tokenizer: LlamaTokenizerFast,
                text_encoder_2: CLIPTextModel,
                tokenizer_2: CLIPTokenizer,
                vae: AutoencoderKLHunyuanVideo,
                feature_extractor: SiglipImageProcessor,
                image_processor: SiglipVisionModel,
                transformer: HunyuanVideoTransformer3DModelPacked,
                scheduler,
            ):
                super().__init__(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    transformer=transformer,
                    image_processor=image_processor,
                    scheduler=scheduler,
                )
                self.register_modules(
                    text_encoder=text_encoder,
                    tokenizer=tokenizer,
                    text_encoder_2=text_encoder_2,
                    tokenizer_2=tokenizer_2,
                    vae=vae,
                    feature_extractor=feature_extractor,
                    image_processor=image_processor,
                    transformer=transformer,
                    scheduler=scheduler,
                )

        sd_models.unload_model_weights()

        shared.log.debug(f'FramePack load: module=llm repo="{repo_hunyuan}"')
        load_args, quant_args = model_quant.get_dit_args({}, module='LLM', device_map=True, allow_quant=offload_native)
        text_encoder = LlamaModel.from_pretrained(repo_hunyuan, subfolder='text_encoder', cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args)
        tokenizer = LlamaTokenizerFast.from_pretrained(repo_hunyuan, subfolder='tokenizer', cache_dir=shared.opts.hfcache_dir)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        sd_models.move_model(text_encoder, devices.cpu)

        shared.log.debug(f'FramePack load: module=te repo="{repo_hunyuan}"')
        text_encoder_2 = CLIPTextModel.from_pretrained(repo_hunyuan, subfolder='text_encoder_2', torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
        tokenizer_2 = CLIPTokenizer.from_pretrained(repo_hunyuan, subfolder='tokenizer_2', cache_dir=shared.opts.hfcache_dir)
        text_encoder_2.requires_grad_(False)
        text_encoder_2.eval()
        sd_models.move_model(text_encoder_2, devices.cpu)

        shared.log.debug(f'FramePack load: module=vae repo="{repo_hunyuan}"')
        vae = AutoencoderKLHunyuanVideo.from_pretrained(repo_hunyuan, subfolder='vae', torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
        vae.requires_grad_(False)
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()
        sd_models.move_model(vae, devices.cpu)

        shared.log.debug(f'FramePack load: module=encoder repo="{repo_encoder}"')
        feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor', cache_dir=shared.opts.hfcache_dir)
        image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
        image_encoder.requires_grad_(False)
        image_encoder.eval()
        sd_models.move_model(image_encoder, devices.cpu)

        shared.log.debug(f'FramePack load: module=transformer repo="{repo_transformer}"')
        load_args, quant_args = model_quant.get_dit_args({}, module='Video', device_map=True, allow_quant=offload_native)
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args)
        transformer.high_quality_fp32_output_for_inference = True
        transformer.requires_grad_(False)
        transformer.eval()
        sd_models.move_model(transformer, devices.cpu)

        shared.sd_model = FramepackHunyuanVideoPipeline(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            vae=vae,
            feature_extractor=feature_extractor,
            image_processor=image_encoder,
            transformer=transformer,
            scheduler=None,
        )

        shared.sd_model = model_quant.do_post_load_quant(shared.sd_model)

        diffusers.loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING['HunyuanVideoTransformer3DModelPacked'] = lambda model_cls, weights: weights # pylint: disable=protected-access
        shared.log.info(f'FramePack load: model={shared.sd_model.__class__.__name__} type={shared.sd_model_type} offload={"native" if offload_native else "lllyasviel"}')
        if offload_native:
            sd_models.apply_balanced_offload(shared.sd_model)
        else:
            DynamicSwapInstaller.install_model(transformer, device=devices.device)
            DynamicSwapInstaller.install_model(text_encoder, device=devices.device)
        devices.torch_gc(force=True)

    except Exception as e:
        shared.log.error(f'FramePack load: {e}')
        errors.display(e, 'FramePack')
        shared.state.end()
        return False

    shared.state.end()
    return True


def unload_model():
    sd_models.unload_model_weights()
