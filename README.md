# SD.Next extension for HunyuanVideo FramePack

Implementation of **Lllyasviel** [FramePack](https://lllyasviel.github.io/frame_pack_gitpage/) for **Tencent** [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) I2V  
With some major differences and improvements:
- *i2v & flf2v support, complex actions with multi-prompts*,  
- *multiple video codecs, raw export, frame export, frame interpolation,*  
- *quantization support, new offloading, more configuration options, cross-platform...*  

> [!NOTE]
> At the moment implemented as [SD.Next](https://github.com/vladmandic/sdnext) extension,  
> but will be fully integrated into the main codebase in the future  
> Reason is to avoid breaking changes as upstream changes are made to the codebase  

> [!IMPORTANT]
> Video support requires `ffmpeg` to be installed and available in the `PATH`  

## Install

Exension repository URL: <https://github.com/vladmandic/sd-extension-framepack>

### Via UI  

Enter repository URL in *SD.Next -> Extensions -> Manual Install* and select *Install*  
Extension will appear as a top-level tab after server restart  

## Via CLI  

Clone repository into SD.Next `/extensions` folder

## Differences

- Supports both **I2V** (image-to-video) and **FLF2V** (frame-last-frame-to-video) modes  
  You can choose if you want to provide end frame or not  
- Supports resolution scaling: from 240p to 960p  
  Input image will be resized to closest aspect ratio supported by HV and scaled to desired resolution  
  *Note*: resolution is directly proportional to VRAM usage, so if you have low VRAM, use lower resolution  
- Support for per-section prompt suffix which if provided uses separate prompt encodings for each prompt section  
  Can be used to achieve complex actions that change over time  
- Replace **lllyasviel offloading** with SD.Next **Balanced offloading**  
  Balanced offload will use more resources, but unless you have a low-end GPU, it should also be much faster  
  especially when used together with quantization  
- Add support for **LLM** and **DiT/Video** modules on-the-fly quantization **quantization**  
  Only available when using native offloading, configure as usual in *settings -> quantization*  
- Add support for post-load quantization such as **NNCF**  
- Configurable resolution, frame-rate  
- Configurable video codec and codec options  
  Includes both hardware-accelerated codecs (e.g. `hevc_nvenc`) and software codecs (e.g. `libx264`)  
- Better preview quality  
- Expose advanced options: *recommended not to change*  
- Model download & load on-demand  
- Configurable torch cross-attention  

### Video

- Video is encoded using selected codec and codec options  
  Default codec is `libx264`, to see codecs available on your system, use refresh  
  *Note*: hardware-accelerated codecs (e.g. `hevc_nvenc`) will be at the top of the list  
- Video encoding can be very memory intensive depending on codec and number of frames  
  Use hardware-accelerated codecs whenever possible  
- Video encoding can be skipped by setting codec to `none`
- Video can optionally have additional interpolated frames added  
  For example, if you render 10sec 30fps video with 0 interpolated frames,  
  its 300 frames that need to be generated  
  But if you set 3 interpolated frames, video fps and duration do not change, but only 100 frames  
  need to be generated and additional 200 interpolated frames are added in-between generated frames  
- Set path in *settings -> image paths -> video*  
- If *settings -> image options -> keep incomplete images* is enabled, the video will be created even if interrupted  
- Does not create intermediate video or image files  
- Optional save raw video frames as `safetensors` file so they can be processed later  

### CLI

> `python framepack_cli.py`

Allows to:  
- Export frames from `safetensors` file as individual images  
  Those can be used for further processing or to manually create video using `ffmpeg` from-image-sequence  
- Encode frames from `safetensors` file into video using `cv2`  
- Encode frames from `safetensors` file into video using `torchvision/ffmpeg`  

### Other Internals

- Refactored main loop  
- Removed hardcoded device mappings  
- Modified HV prompt template  
- State management  
- Create actual model pipeline from individual components  
- Add inference stats  
- Redo logging  
- Redo video saving  

## TODO

- CFGzero
- LoRA support
- Custom system prompt
- Frame upscaling
- Full codebase integration
- API
