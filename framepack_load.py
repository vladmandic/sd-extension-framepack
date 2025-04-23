import time
from modules import shared, devices, errors, sd_models, model_quant


default_model = {
    'pipeline': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': '' },
    'vae': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'vae' },
    'text_encoder': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'text_encoder' },
    'tokenizer': {'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'tokenizer' },
    # 'text_encoder': { 'repo': 'Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'subfolder': '' },
    # 'tokenizer': { 'repo': 'Kijai/llava-llama-3-8b-text-encoder-tokenizer', 'subfolder': '' },
    'text_encoder_2': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'text_encoder_2' },
    'tokenizer_2': { 'repo': 'hunyuanvideo-community/HunyuanVideo', 'subfolder': 'tokenizer_2' },
    'feature_extractor': { 'repo': 'lllyasviel/flux_redux_bfl', 'subfolder': 'feature_extractor' },
    'image_encoder': { 'repo': 'lllyasviel/flux_redux_bfl', 'subfolder': 'image_encoder' },
    'transformer': { 'repo': 'lllyasviel/FramePackI2V_HY', 'subfolder': '' },
}
model = default_model.copy()


def split_url(url):
    if url.count('/') == 1:
        url += '/'
    if url.count('/') != 2:
        raise ValueError(f'Invalid URL: {url}')
    url = [section.strip() for section in url.split('/')]
    return { 'repo': f'{url[0]}/{url[1]}', 'subfolder': url[2] }


def set_model(receipe: str=None):
    if receipe is None or receipe == '':
        return
    lines = [line.strip() for line in receipe.split('\n') if line.strip() != '' and ':' in line]
    for line in lines:
        k, v = line.split(':', 1)
        k = k.strip()
        if k not in default_model.keys():
            shared.log.warning(f'FramePack receipe: key={k} invalid')
        model[k] = split_url(v)
        shared.log.debug(f'FramePack receipe: set {k}={model[k]}')


def get_model():
    receipe = ''
    for k, v in model.items():
        receipe += f'{k}: {v["repo"]}/{v["subfolder"]}\n'
    return receipe.strip()


def reset_model():
    global model # pylint: disable=global-statement
    model = default_model.copy()
    shared.log.debug('FramePack receipe: reset')
    return ''


def load_model(pipeline:str=None, text_encoder:str=None, text_encoder_2:str=None, feature_extractor:str=None, image_encoder:str=None, transformer:str=None):
    shared.state.begin('Load')
    if pipeline is not None:
        model['pipeline'] = split_url(pipeline)
    if text_encoder is not None:
        model['text_encoder'] = split_url(text_encoder)
    if text_encoder_2 is not None:
        model['text_encoder_2'] = split_url(text_encoder_2)
    if feature_extractor is not None:
        model['feature_extractor'] = split_url(feature_extractor)
    if image_encoder is not None:
        model['image_encoder'] = split_url(image_encoder)
    if transformer is not None:
        model['transformer'] = split_url(transformer)
    shared.log.trace(f'FramePack load: {model}')

    try:
        import diffusers
        from diffusers import HunyuanVideoImageToVideoPipeline, AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
        from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked

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
        t0 = time.time()

        shared.log.debug(f'FramePack load: module=llm {model["text_encoder"]}')
        load_args, quant_args = model_quant.get_dit_args({}, module='LLM', device_map=True)
        text_encoder = LlamaModel.from_pretrained(model["text_encoder"]["repo"], subfolder=model["text_encoder"]["subfolder"], cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args)
        tokenizer = LlamaTokenizerFast.from_pretrained(model["tokenizer"]["repo"], subfolder=model["tokenizer"]["subfolder"], cache_dir=shared.opts.hfcache_dir)
        text_encoder.requires_grad_(False)
        text_encoder.eval()
        sd_models.move_model(text_encoder, devices.cpu)

        shared.log.debug(f'FramePack load: module=te {model["text_encoder_2"]}')
        text_encoder_2 = CLIPTextModel.from_pretrained(model["text_encoder_2"]["repo"], subfolder=model["text_encoder_2"]["subfolder"], torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
        tokenizer_2 = CLIPTokenizer.from_pretrained(model["pipeline"]["repo"], subfolder='tokenizer_2', cache_dir=shared.opts.hfcache_dir)
        text_encoder_2.requires_grad_(False)
        text_encoder_2.eval()
        sd_models.move_model(text_encoder_2, devices.cpu)

        shared.log.debug(f'FramePack load: module=vae {model["vae"]}')
        vae = AutoencoderKLHunyuanVideo.from_pretrained(model["vae"]["repo"], subfolder=model["vae"]["subfolder"], torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
        vae.requires_grad_(False)
        vae.eval()
        vae.enable_slicing()
        vae.enable_tiling()
        sd_models.move_model(vae, devices.cpu)

        shared.log.debug(f'FramePack load: module=encoder {model["feature_extractor"]} model={model["image_encoder"]}')
        feature_extractor = SiglipImageProcessor.from_pretrained(model["feature_extractor"]["repo"], subfolder=model["feature_extractor"]["subfolder"], cache_dir=shared.opts.hfcache_dir)
        print('HERE', model["image_encoder"]["repo"], model["image_encoder"]["subfolder"])
        image_encoder = SiglipVisionModel.from_pretrained(model["image_encoder"]["repo"], subfolder=model["image_encoder"]["subfolder"], torch_dtype=devices.dtype, cache_dir=shared.opts.hfcache_dir)
        image_encoder.requires_grad_(False)
        image_encoder.eval()
        sd_models.move_model(image_encoder, devices.cpu)

        shared.log.debug(f'FramePack load: module=transformer {model["transformer"]}')
        load_args, quant_args = model_quant.get_dit_args({}, module='Video', device_map=True)
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(model["transformer"]["repo"], subfolder=model["transformer"]["subfolder"], cache_dir=shared.opts.hfcache_dir, **load_args, **quant_args)
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
        t1 = time.time()

        diffusers.loaders.peft._SET_ADAPTER_SCALE_FN_MAPPING['HunyuanVideoTransformer3DModelPacked'] = lambda model_cls, weights: weights # pylint: disable=protected-access
        shared.log.info(f'FramePack load: model={shared.sd_model.__class__.__name__} type={shared.sd_model_type} time={t1-t0:.2f}')
        sd_models.apply_balanced_offload(shared.sd_model)
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
