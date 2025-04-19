# SD.Next extension for HunyuanVideo FramePack

Implementation of **Lllyasviel** [FramePack](https://lllyasviel.github.io/frame_pack_gitpage/) for **Tencent** [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) I2V  
With some major differences and improvements:
- *quantization support, new offloading, more configuration options, cross-platform*  

> [!NOTE]
> At the moment implemented as [SD.Next](https://github.com/vladmandic/sdnext) extension, but will be fully > integrated into the main codebase in the future  
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

- Implement both SD.Next **Balanced offloading** (native) and **lllyasviel offloading**  
  Balanced offload will use more resources, but unless you have a low-end GPU, it should also be much faster  
  especially when used together with quantization  
- Add support for **LLM** and **DiT/Video** modules on-the-fly quantization **quantization**  
  Only available when using native offloading, configure as usual in *settings -> quantization*  
- Configurable resolution, frame-rate, video compression (crf)  
- Expose advanced options: *recommended not to change*  
- Model download & load on-demand  
- Configurable torch cross-attention  

### Internals  

- Removed hardcoded device mappings
- State management
- Create actual model pipeline from individual components  
- Add inference stats  
- Redo logging  
- Redo video saving  

### Video saving

- Video is encoded using `libx264` codec  
- Set path in *settings -> image paths -> video*  
- If *settings -> image options -> keep incomplete images* is enabled, the video will be created even if interrupted  
- Does not create intermediate video or image files  

## TODO

- integrated UI progress bar
- improve LoRA support
- frame interpolation
- full codebase integration
