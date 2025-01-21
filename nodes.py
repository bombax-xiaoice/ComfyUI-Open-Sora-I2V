import os
import os.path
import sys
import folder_paths
comfy_path = os.path.dirname(folder_paths.__file__)
sys.path.append(f'{comfy_path}/custom_nodes/ComfyUI-Open-Sora-I2V')
print(sys.path)

from PIL import Image
from comfy import model_management, latent_formats
import latent_preview
import argparse
import torch
import numpy as np
import tempfile
from itertools import chain
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.utils.config_utils import read_config
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.misc import to_torch_dtype
from opensora.utils.inference_utils import prepare_multi_resolution_info
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.models.text_encoder.t5 import text_preprocessing
from colossalai.cluster import DistCoordinator

checkpoint_path=f'{comfy_path}/models/checkpoints'
pretrained_checkpoint=os.listdir(checkpoint_path)

vae_path=f'{comfy_path}/models/vae'
pretrained_vae=os.listdir(vae_path)

clip_path=f'{comfy_path}/models/clip'
pretrained_clip=os.listdir(clip_path)

config_path=f'{comfy_path}/custom_nodes/ComfyUI-Open-Sora-I2V/configs'
subfolders = [sub for sub in os.listdir(config_path) if sub.startswith('opensora') or sub.startswith('pixart')]
config_lists=sum([[sub+'/inference/'+f for f in os.listdir(os.path.join(config_path, sub, 'inference'))] for sub in subfolders],[])

class OpenSoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "config": (config_lists, {"default": "opensora-v1-2/inference/sample.py"}),
                "duration": (["2s", "4s", "8s", "16s"], {"default": "2s"}),
                "resolution": (["144p", "240p", "360p", "480p", "720p"], {"default": "240p"}),
                "aspect_ratio": (["9:16", "16:9", "3:4", "4:3", "1:1"], {"default": "1:1"}),
                "custom_checkpoint": (["Default", *pretrained_checkpoint], {"default":"Default"}),
                "custom_vae": (["Default", *pretrained_vae], {"default":"Default"}),
                "custom_clip": (["Default", "Skip", *pretrained_clip], {"default":"Default"}),
            },
        }

    RETURN_TYPES = ("PIPE","INT","FLOAT","CLIP","VAE","INT","INT","INT","FLOAT",)
    RETURN_NAMES = ("pipe","steps","guidance","text_encoder","vae","num_frames","width","height","fps",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,config,duration,resolution,aspect_ratio,custom_checkpoint,custom_vae,custom_clip):
        config=f'{config_path}/{config}'
        cfg = read_config(config)
        custom_checkpoint_config = None #config.json under custom_checkpoint folder, or a pair of .json config file and model file (.pt|.pth|.safetensors) with the same filepath except extension 
        if custom_checkpoint and custom_checkpoint!="Default" and os.path.exists(os.path.join(checkpoint_path, custom_checkpoint)):
            if os.path.isdir(os.path.join(checkpoint_path, custom_checkpoint)) and os.path.exists(os.path.join(checkpoint_path, custom_checkpoint, "config.json")):
                cfg.model["from_pretrained"] = os.path.join(checkpoint_path, custom_checkpoint)
            elif os.path.splitext(custom_checkpoint)[1]=='.json' and os.path.exists(os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.safetensors')):
                custom_checkpoint_config = os.path.join(checkpoint_path, custom_checkpoint)
                cfg.model["from_pretrained"] = os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.safetensors')
            elif os.path.splitext(custom_checkpoint)[1]=='.json' and os.path.exists(os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.pt')):
                custom_checkpoint_config = os.path.join(checkpoint_path, custom_checkpoint)
                cfg.model["from_pretrained"] = os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.pt')
            elif os.path.splitext(custom_checkpoint)[1]=='.json' and os.path.exists(os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.pth')):
                custom_checkpoint_config = os.path.join(checkpoint_path, custom_checkpoint)
                cfg.model["from_pretrained"] = os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.pth')
            elif os.path.exists(os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.json')):
                custom_checkpoint_config = os.path.join(checkpoint_path, os.path.splitext(custom_checkpoint)[0]+'.json')
                cfg.model["from_pretrained"] = os.path.join(checkpoint_path, custom_checkpoint)
            else:
                cfg.model["from_pretrained"] = os.path.join(checkpoint_path, custom_checkpoint)
        if custom_vae and os.path.exists(os.path.join(vae_path, custom_vae)):
            cfg.vae["from_pretrained"] = os.path.join(vae_path, custom_vae)
        if custom_clip and custom_clip!="Default" and os.path.exists(os.path.join(clip_path, custom_clip)):
            cfg.text_encoder["from_pretrained"] = os.path.join(clip_path, custom_clip) if custom_clip!="Skip" else None
        
        if hasattr(cfg, 'multi_resolution') and cfg.multi_resolution!='PixArtMS':
            cfg.image_size=get_image_size(resolution, aspect_ratio)
        # init distributed
        enable_sequence_parallelism = False
        if os.environ.get('RANK') and os.environ.get('LOCAL_RANK') and os.environ.get('WORLD_SIZE'):
            colossalai.launch_from_torch({})
            coordinator = DistCoordinator()

            if coordinator.world_size > 1:
                set_sequence_parallel_group(dist.group.WORLD) 
                enable_sequence_parallelism = True

        torch.set_grad_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        dtype = torch.float32

        input_size = (get_num_frames(duration) if hasattr(cfg, 'multi_resolution') and cfg.multi_resolution!='PixArtMS' else cfg.num_frames, *cfg.image_size)
        if os.path.isdir(cfg.vae["from_pretrained"]) and os.path.exists(os.path.join(cfg.vae["from_pretrained"], "model.safetensors")): #vae by directory
            from opensora.models.vae.vae import VideoAutoencoderPipeline
            vae = VideoAutoencoderPipeline.from_pretrained(cfg.vae["from_pretrained"])
        elif os.path.splitext(cfg.vae["from_pretrained"])[1]=='.json': #vae by config file (and its pair model file)
            from opensora.models.vae.vae import VideoAutoencoderPipeline, VideoAutoencoderPipelineConfig
            vaeconfig = VideoAutoencoderPipelineConfig.from_pretrained(cfg.vae["from_pretrained"])
            vae = VideoAutoencoderPipeline(vaeconfig)
            from opensora.utils.ckpt_utils import load_checkpoint
            if os.path.exists(os.path.splitext(cfg.vae["from_pretrained"])[0]+'.safetensors'):
                load_checkpoint(vae, os.path.splitext(cfg.vae["from_pretrained"])[0]+'.safetensors')
            elif os.path.exists(os.path.splitext(cfg.vae["from_pretrained"])[0]+'.pt'):
                load_checkpoint(vae, os.path.splitext(cfg.vae["from_pretrained"])[0]+'.pt')
            elif os.path.exists(os.path.splitext(cfg.vae["from_pretrained"])[0]+'.pth'):
                load_checkpoint(vae, os.path.splitext(cfg.vae["from_pretrained"])[0]+'.pth')
            else:
                raise ValueError('No corresponding .pt .pth or .safetensors file for config:' + cfg.vae["from_pretrained"])
        #elif not os.path.isdir(cfg.vae["from_pretrained"]) and os.path.exists(os.path.splitext(cfg.vae["from_pretrained"])[0]+'.json'): #vae by model file (and its pair config file)
        #    from opensora.models.vae.vae import VideoAutoencoderPipeline, VideoAutoencoderPipelineConfig
        #    vaeconfig = VideoAutoencoderPipelineConfig.from_pretrained(os.path.splitext(cfg.vae["from_pretrained"])[0]+'.json')
        #    vae = VideoAutoencoderPipeline(vaeconfig)
        #    from opensora.utils.ckpt_utils import load_checkpoint
        #    load_checkpoint(vae, os.path.splitext(cfg.vae["from_pretrained"]))
        elif not os.path.isdir(cfg.vae["from_pretrained"]) and os.path.splitext(cfg.vae["from_pretrained"])[1] in ['.safetensor','.bin','.ckpt']:
            import comfy.utils
            import comfy.sd
            sd = comfy.utils.load_torch_file(cfg.vae["from_pretrained"])
            vae = comfy.sd.VAE(sd=sd)
            setattr(vae, 'get_latent_size', lambda i:(i[0], i[1]//vae.downscale_ratio, i[2]//vae.downscale_ratio))
            setattr(vae, 'eval', lambda:vae)
            setattr(vae, 'out_channels', vae.output_channels)
        else: #vae by huggingface or model file without any pair config
            vae = build_module(cfg.vae, MODELS).to(device=torch.device('cpu'), dtype=dtype)
        latent_size = vae.get_latent_size(input_size)
        vae = vae.eval()
        
        if os.path.isfile(cfg.text_encoder["from_pretrained"]): #text encoder by a single model file
            from comfy.sd import load_clip, CLIPType
            t5 = load_clip(ckpt_paths=[cfg.text_encoder["from_pretrained"]], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=CLIPType.SD3)
            if hasattr(t5, 'tokenizer') and hasattr(t5.tokenizer, 't5xxl') and hasattr(t5.tokenizer.t5xxl, 'min_length') and 'model_max_length' in cfg.text_encoder:
                t5.tokenizer.t5xxl.min_length = cfg.text_encoder['model_max_length']
            import typing
            text_encoder = typing.NewType('PseudoTextEncoder',typing.Generic)
            setattr(text_encoder, 't5', t5)
            setattr(text_encoder, 'encode', lambda prompt:{'y':text_encoder.t5.encode(prompt[0] if isinstance(prompt, list) else prompt)[:,None]})
            setattr(text_encoder, 'null', lambda n:text_encoder.y_embedder.y_embedding[None].repeat(n, 1, 1)[:, None])
        elif cfg.text_encoder["from_pretrained"]!=None: #text encoder huggingface or a local directory
            text_encoder = build_module(cfg.text_encoder, MODELS, device = torch.device('cpu'))
            text_encoder.t5.model = text_encoder.t5.model.eval()
        else: #skip loading text encoder
            text_encoder = None

        model_kwargs = {k: v for k, v in cfg.model.items() if k not in ("type", "from_pretrained", "force_huggingface")}
        if custom_checkpoint_config: #a pair of config and model file
            with open(custom_checkpoint_config, 'r') as r:
                content = ''.join(r.readlines())
                if 'STDiT3' in content or 'STDiT-v3' in content:
                    from opensora.models.stdit.stdit3 import STDiT3
                    from opensora.models.stdit.stdit3 import STDiT3Config
                    config = STDiT3Config.from_pretrained(custom_checkpoint_config, **model_kwargs)
                    model = STDiT3(config)
                elif 'STDiT2' in content or 'STDiT-v2' in content:
                    from opensora.models.stdit.stdit2 import STDiT2
                    from opensora.models.stdit.stdit2 import STDiT2Config
                    config = STDiT2Config.from_pretrained(custom_checkpoint_config, **model_kwargs)
                    model = STDiT2(config)
                else:
                    from opensora.models.stdit.stdit import STDiT
                    from transformers import PretrainedConfig
                    config = PretrainedConfig.from_pretrained(custom_checkpoint_config, **model_kwargs)
                    import inspect
                    model = STDiT(**{k:v for k,v in config.to_dict().items() if k!="self" and k in inspect.signature(STDiT.__init__).parameters})
                from opensora.utils.ckpt_utils import load_checkpoint
                load_checkpoint(model, cfg.model["from_pretrained"])
        elif os.path.exists(cfg.model["from_pretrained"]) and 'STDiT' in cfg.model["from_pretrained"]: #model file without any pairing config
            if 'STDiT3-3B' in cfg.model["from_pretrained"] or 'STDiT-v3-3B' in cfg.model["from_pretrained"]:
                from opensora.models.stdit.stdit3 import STDiT3 as STDiT
                from opensora.models.stdit.stdit3 import STDiT3Config
                config = STDiT3Config(depth=28, hidden_size=1872, patch_size=(1, 2, 2), num_heads=26, **model_kwargs)
            elif 'STDiT-v3' in cfg.model["from_pretrained"] or 'STDiT3' in cfg.model["from_pretrained"]:
                from opensora.models.stdit.stdit3 import STDiT3 as STDiT
                from opensora.models.stdit.stdit3 import STDiT3Config
                config = STDiT3Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **model_kwargs)
            elif 'STDiT-v2' in cfg.model["from_pretrained"] or 'STDiT2' in cfg.model["from_pretrained"]:
                from opensora.models.stdit.stdit2 import STDiT2 as STDiT
                from opensora.models.stdit.stdit2 import STDiT2Config
                config = STDiT2Config(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **model_kwargs)
            else:
                from opensora.models.stdit.stdit import STDiT
                config = None
            if os.path.isdir(cfg.model["from_pretrained"]):
                model = STDiT.from_pretrained(cfg.model["from_pretrained"], trust_remote_code=True, **model_kwargs)
            else:
                model = STDiT(config) if config else STDiT(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **model_kwargs)
                from opensora.utils.ckpt_utils import load_checkpoint
                load_checkpoint(model, cfg.model["from_pretrained"])
        else: #huggingface
            model = build_module(
                cfg.model,
                MODELS,
                input_size=latent_size,
                in_channels=vae.out_channels,
                caption_channels=text_encoder.output_dim if hasattr(text_encoder, 'output_dim') else text_encoder.t5.cond_stage_model.t5xxl.transformer.shared.weight.shape[-1] if hasattr(text_encoder,'t5') and hasattr(text_encoder.t5,'cond_stage_model') and hasattr(text_encoder.t5.cond_stage_model,'t5xxl') and hasattr(text_encoder.t5.cond_stage_model.t5xxl,'transformer') and hasattr(text_encoder.t5.cond_stage_model.t5xxl.transformer,'shared') else 4096,
                model_max_length=text_encoder.model_max_length if hasattr(text_encoder, 'model_max_length') else cfg.text_encoder['model_max_length'] if 'model_max_length' in cfg.text_encoder else 120,
                #dtype=dtype,
                enable_sequence_parallelism=enable_sequence_parallelism,
            ).to(device=torch.device('cpu'))
        model = model.eval()
        if text_encoder:
            setattr(text_encoder, 'y_embedder', model.y_embedder)
        
        if not "num_sampling_steps" in cfg.scheduler:
            cfg.scheduler["num_sampling_steps"] = 100 #default steps
        if not "cfg_scale" in cfg.scheduler:
            cfg.scheduler["cfg_scale"] = 7.0 #default guidance
        scheduler = build_module(cfg.scheduler, SCHEDULERS)
        
        latent_size = (1, vae.out_channels, *latent_size)
        return ((cfg, model, text_encoder, scheduler, input_size, latent_size), cfg.scheduler.num_sampling_steps, cfg.scheduler.cfg_scale, text_encoder, vae, input_size[0], input_size[2], input_size[1], cfg.fps)

class OpenSoraSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE",),
                "seed": ("INT", {"default": 0}),
                "dtype": (["fp32","fp16","bf16"], {"default": "fp16"}),
                "steps": ("INT", {"default": 100}),
                "guidance": ("FLOAT", {"default": 4.5}),

            },
            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "reference": ("LATENT",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,pipe,seed,dtype,steps,guidance,positive=None,negative=None,reference=None):
        (cfg, model, text_encoder, scheduler, input_size, latent_size) = pipe
        (num_frames, height, width) = input_size
        set_random_seed(seed=seed)
        if steps != None:
            scheduler.num_sampling_steps = steps
        if guidance != None:
            scheduler.cfg_scale = guidance

        device = model_management.get_torch_device()
        dtype = to_torch_dtype(dtype)
        olddtype = dtype

        model_args = prepare_multi_resolution_info(cfg.multi_resolution if hasattr(cfg, 'multi_resolution') else "OpenSora", 1, (height, width), num_frames, cfg.fps, device, dtype)
        
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        torch.cuda.empty_cache()
        if positive!=None: #do not need text_encoder if CONDITIONINGs are already provided
            model_args['positive_embeds'] = positive["y"].to(dtype=dtype) if isinstance(positive, dict) and "y" in positive else positive[0][0][:,None].to(dtype=dtype)
            if isinstance(positive, dict) and "mask" in positive:
                model_args['positive_attention_mask'] = positive["mask"] 
            elif len(positive[0])>1 and isinstance(positive[0][1], dict) and 'pooled_output' in positive[0][1]:
                model_args['positive_attention_mask'] = positive[0][1]['pooled_output'] 
            if negative!=None:
                model_args['negative_embeds'] = negative.to(dtype=dtype) if isinstance(negative, torch.Tensor) else negative[0][0][:,None].to(dtype=dtype)
        else:
            if hasattr(text_encoder,'t5') and hasattr(text_encoder.t5,'model'): #default loader or huggingface loader
                if text_encoder.t5.model.device != model_management.text_encoder_device():
                    try:
                        text_encoder.t5.model = text_encoder.t5.model.to(model_management.text_encoder_device())
                    except:
                        model_management.unload_all_models()
                        model_management.soft_empty_cache()
                        torch.cuda.empty_cache()
                        text_encoder.t5.model = text_encoder.t5.model.to('cpu')
            elif hasattr(text_encoder,'t5') and hasattr(text_encoder.t5, 'cond_stage_model'): #single file loader
                if text_encoder.t5.cond_stage_model.device != model_management.text_encoder_device():
                    try:
                        text_encoder.t5.cond_stage_model = text_encoder.t5.cond_stage_model.to(model_management.text_encoder_device())
                    except:
                        model_management.unload_all_models()
                        model_management.soft_empty_cache()
                        torch.cuda.empty_cache()
                        text_encoder.t5.cond_stage_model = text_encoder.t5.cond_stage_model.to('cpu')
            else: #unknown loader
                try:
                    text_encoder = text_encoder.to(model_management.text_encoder_device())
                except:
                    model_management.unload_all_models()
                    model_management.soft_empty_cache()
                    torch.cuda.empty_cache()
                    try:
                        text_encoder = text_encoder.to('cpu')
                    except:
                        pass

        if not (hasattr(model, 'device') and model.device==device or hasattr(model, 'parameters') and callable(model.parameters) and next(model.parameters()).device==device) or not (hasattr(model, 'dtype') and model.dtype==dtype):
            model = model.to(device = device, dtype = dtype)
        
        z = torch.randn(latent_size, device=device, dtype=dtype)
        masks = torch.ones(1, z.shape[-3], device=device, dtype=dtype)
        if reference != None: #reference image(s)
            olddtype = reference["samples"].dtype
            if reference["samples"].device != device and reference["samples"].dtype != dtype:
                reference["samples"] = reference["samples"].to(device = device, dtype = dtype)
            ref = reference["samples"].permute(1, 0, 2, 3)
            if ref.shape[-3]==1 or z.shape[-3]==1: #single ref image as the first frame
                z[0,:,0]=ref[:,0]
                masks[0,0] = 0
            elif ref.shape[-3] <= z.shape[-3]: #ref has less or equal frames than output
                indices = [int(round(i*(z.shape[-3]-1)/(ref.shape[-3]-1))) for i in range(ref.shape[-3])]
                z[0,:,indices] = ref
                masks[0,indices] = 0
            else: #ref has more frames than output
                indices = [int(round(i*(ref.shape[-3]-1)/(z.shape[-3]-1))) for i in range(z.shape[-3])]
                z[0] = ref[:,indices]
                masks[0,:] = 0

        try: #progress bar and preview
            import typing
            fakepipe = typing.NewType('FakePipe',typing.Generic)
            setattr(fakepipe, 'load_device', device)
            setattr(fakepipe, 'model', typing.NewType('FakeModel',typing.Generic))
            setattr(fakepipe.model, 'latent_format', latent_formats.SD15())
            callback = latent_preview.prepare_callback(fakepipe, scheduler.num_sampling_steps)
        except:
            callback = None
        
        samples = scheduler.sample(
            model,
            text_encoder,
            z = z,
            prompts=['',],
            device=device,
            additional_args=model_args,
            mask = masks,
            progress = lambda s,x0,x,t:callback(s,x0,x,t) if callback else True,
        )

        if positive==None: #do not need text_encoder if CONDITIONINGs are already provided
            if hasattr(text_encoder,'t5') and hasattr(text_encoder.t5,'model'): #default loader or huggingface loader
                if text_encoder.t5.model.device != model_management.text_encoder_offload_device():
                    text_encoder.t5.model = text_encoder.t5.model.to(model_management.text_encoder_offload_device())
            elif hasattr(text_encoder,'t5') and hasattr(text_encoder.t5, 'cond_stage_model'): #single file loader
                if text_encoder.t5.cond_stage_model.device != model_management.text_encoder_offload_device():
                    text_encoder.t5.cond_stage_model = text_encoder.t5.cond_stage_model.to(model_management.text_encoder_offload_device())
            else: #unknown loader
                try:
                    text_encoder = text_encoder.to(model_management.text_encoder_offload_device())
                except:
                    pass
        if not (hasattr(model, 'device') and model.device==model_management.vae_offload_device() or hasattr(model, 'parameters') and callable(model.parameters) and next(model.parameters()).device==model_management.vae_offload_device()) and not (hasattr(model, 'dtype') and model.dtype==dtype):
            model = model.to(model_management.vae_offload_device())
        if samples.device != model_management.vae_offload_device() or samples.dtype != olddtype:
            samples = samples.to(device = model_management.vae_offload_device(), dtype = olddtype)
        if reference!=None and reference["samples"].device != device and reference["samples"].dtype != dtype:
            reference["samples"] = reference["samples"].to(device = model_management.vae_offload_device())
        
        return ({"samples":samples},)


class OpenSoraTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_encoder": ("CLIP",),
                "positive_prompt": ("STRING",{"default":"", "multiline":True}),
            },
            "optional": {
                "negative_prompt": ("STRING",{"default":"", "multiline":True}),
                "camera_motion": (["none", "pan right", "pan left", "tilt up", "tilt down", "zoom in", "zoom out", "static"], {"default":"none"}),
                "enable_motion_strength": (["disabled", "enabled"], {"default":"disabled"}),
                "motion_strength": ("INT", {"default":5, "min":0, "max":100, "step":1}),
                "enable_aesthetic_score": (["disabled", "enabled"], {"default":"disabled"}),
                "aesthetic_score": ("FLOAT", {"default":6.5, "min":4.0, "max":7.0, "step":0.1}),
            }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("positive","negative",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,text_encoder,positive_prompt,negative_prompt=None, camera_motion="none", enable_motion_strength="disabled", motion_strength=5, enable_aesthetic_score="disabled", aesthetic_score=6.5):
        if enable_aesthetic_score=="enabled":
            positive_prompt += f" aesthetic score: {aesthetic_score:.1f}."
        if enable_motion_strength=="enabled":
            positive_prompt += f" motion score: {float(motion_strength):.1f}."
        if camera_motion != None and camera_motion != 'none':
            positive_prompt += f" camera motion: {camera_motion}."
        if hasattr(text_encoder,'t5') and hasattr(text_encoder.t5,'model'): #default loader or huggingface loader
            positive_prompt = [text_preprocessing(positive_prompt),]
            negative_prompt = [text_preprocessing(negative_prompt),] if negative_prompt else None
            if text_encoder.t5.model.device != model_management.text_encoder_device():
                try:
                    text_encoder.t5.model = text_encoder.t5.model.to(model_management.text_encoder_device())
                except:
                    model_management.unload_all_models()
                    model_management.soft_empty_cache()
                    torch.cuda.empty_cache()
                    text_encoder.t5.model = text_encoder.t5.model.to('cpu')
        elif hasattr(text_encoder,'t5') and hasattr(text_encoder.t5, 'cond_stage_model'): #single file loader
            if text_encoder.t5.cond_stage_model.device != model_management.text_encoder_device():
                try:
                    text_encoder.t5.cond_stage_model = text_encoder.t5.cond_stage_model.to(model_management.text_encoder_device())
                except:
                    model_management.unload_all_models()
                    model_management.soft_empty_cache()
                    torch.cuda.empty_cache()
                    text_encoder.t5.cond_stage_model = text_encoder.t5.cond_stage_model.to('cpu')
        else: #unknown loader
            try:
                text_encoder = text_encoder.to(model_management.text_encoder_device())
            except:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
                torch.cuda.empty_cache()
                try:
                    text_encoder = text_encoder.to('cpu')
                except:
                    pass

        positive_embeds = text_encoder.encode(positive_prompt)

        if hasattr(text_encoder,'t5') and hasattr(text_encoder.t5,'model'): #default loader or huggingface loader
            if text_encoder.t5.model.device != model_management.text_encoder_offload_device():
                text_encoder.t5.model = text_encoder.t5.model.to(model_management.text_encoder_offload_device())
        elif hasattr(text_encoder,'t5') and hasattr(text_encoder.t5, 'cond_stage_model'): #single file loader
            if text_encoder.t5.cond_stage_model.device != model_management.text_encoder_offload_device():
                text_encoder.t5.cond_stage_model = text_encoder.t5.cond_stage_model.to(model_management.text_encoder_offload_device())
        else: #unknown loader
            try:
                text_encoder = text_encoder.to(model_management.text_encoder_offload_device())
            except:
                pass
    
        if isinstance(positive_embeds, dict) and "y" in positive_embeds:
            if "mask" in positive_embeds:
                positive_embeds = [[positive_embeds['y'][0],{"cond":positive_embeds['y'][0],"pooled_output":positive_embeds['mask']}]]
            else:
                positive_embeds = [[positive_embeds['y'][0],{"cond":positive_embeds['y'][0]}]]
        if not negative_prompt and hasattr(text_encoder, 'null') and callable(text_encoder.null):
            negative_embeds = text_encoder.null(1)
            negative_embeds = [[negative_embeds[0], {"cond":negative_embeds[0]}]]
        else:
            negative_embeds = text_encoder.encode(negative_prompt)
            if isinstance(negative_embeds, dict) and "y" in negative_embeds:
                if "mask" in negative_embeds:
                    negative_embeds = [[negative_embeds['y'][0],{"cond":negative_embeds['y'][0],"pooled_output":negative_embeds['mask']}]]
                else:
                    negative_embeds = [[negative_embeds['y'][0],{"cond":negative_embeds['y'][0]}]]
        return (positive_embeds, negative_embeds,)

class OpenSoraEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "dtype": (["fp32","fp16","bf16"], {"default": "fp16"}),
            },
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,vae,image,dtype):
        imagedevice = image.device
        imagedtype = image.dtype
        dtype = to_torch_dtype(dtype)
        if vae.device != model_management.vae_device() or vae.dtype != dtype:
            if hasattr(vae, 'module') and hasattr(vae.module, 'encoder') and hasattr(vae.module, 'quant_conv'):
                vae.module.encoder = vae.module.encoder.to(device = model_management.vae_device(), dtype = dtype)
                vae.module.quant_conv = vae.module.quant_conv.to(device = model_management.vae_device(), dtype = dtype)
            elif hasattr(vae, 'spatial_vae') and hasattr(vae.spatial_vae, 'module') and hasattr(vae.spatial_vae.module, 'encoder') and hasattr(vae.spatial_vae.module, 'quant_conv') \
                and hasattr(vae, 'temporal_vae') and hasattr(vae.temporal_vae, 'encoder') and hasattr(vae.temporal_vae, 'quant_conv'):
                vae.spatial_vae.module.encoder = vae.spatial_vae.module.encoder.to(device = model_management.vae_device(), dtype = dtype)
                vae.spatial_vae.module.quant_conv = vae.spatial_vae.module.quant_conv.to(device = model_management.vae_device(), dtype = dtype)
                vae.temporal_vae.encoder = vae.temporal_vae.encoder.to(device = model_management.vae_device(), dtype = dtype)
                vae.temporal_vae.quant_conv = vae.temporal_vae.quant_conv.to(device = model_management.vae_device(), dtype = dtype)
                vae.scale = vae.scale.to(device = model_management.vae_device(), dtype = dtype)
                vae.shift = vae.shift.to(device = model_management.vae_device(), dtype = dtype)
            else:
                vae = vae.to(device = model_management.vae_device(), dtype = dtype)
        if image.device != model_management.vae_device() or image.dtype != dtype:
            image = image.to(device = model_management.vae_device(), dtype = dtype)
        samples = vae.encode((image*2.0-1.0).unsqueeze(0).permute(0,4,1,2,3))
        if vae.device != model_management.vae_offload_device():
            if hasattr(vae, 'module') and hasattr(vae.module, 'encoder') and hasattr(vae.module, 'quant_conv'):
                vae.module.encoder = vae.module.encoder.to(device = model_management.vae_offload_device())
                vae.module.quant_conv = vae.module.quant_conv.to(device = model_management.vae_offload_device())
            elif hasattr(vae, 'spatial_vae') and hasattr(vae.spatial_vae, 'module') and hasattr(vae.spatial_vae.module, 'encoder') and hasattr(vae.spatial_vae.module, 'quant_conv') \
                and hasattr(vae, 'temporal_vae') and hasattr(vae.temporal_vae, 'encoder') and hasattr(vae.temporal_vae, 'quant_conv'):
                vae.spatial_vae.module.encoder = vae.spatial_vae.module.encoder.to(device = model_management.vae_offload_device())
                vae.spatial_vae.module.quant_conv = vae.spatial_vae.module.quant_conv.to(device = model_management.vae_offload_device())
                vae.temporal_vae.encoder = vae.temporal_vae.encoder.to(device = model_management.vae_offload_device())
                vae.temporal_vae.quant_conv = vae.temporal_vae.quant_conv.to(device = model_management.vae_offload_device())
                vae.scale = vae.scale.to(device = model_management.vae_offload_device())
                vae.shift = vae.shift.to(device = model_management.vae_offload_device())
            else:
                vae = vae.to(device = model_management.vae_offload_device())
        if image.device != imagedevice or image.dtype != imagedtype:
            image = image.to(device = imagedevice, dtype = imagedtype)
        if samples.device != imagedevice or samples.dtype != imagedtype:
            samples = samples.to(device = imagedevice, dtype = imagedtype)

        return ({"samples":samples.permute(0, 2, 1, 3, 4).squeeze(0)},)

class OpenSoraDecoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": ("VAE",),
                "latents": ("LATENT",),
                "dtype": (["fp32","fp16","bf16"], {"default": "fp16"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "OpenSora"

    def run(self,vae,latents,dtype):
        olddevice = latents["samples"].device
        olddtype = latents["samples"].dtype
        dtype = to_torch_dtype(dtype)
        if vae.device != model_management.vae_device() or vae.dtype != dtype:
            if hasattr(vae, 'module') and hasattr(vae.module, 'decoder') and hasattr(vae.module, 'post_quant_conv'):
                vae.module.decoder = vae.module.decoder.to(device = model_management.vae_device(), dtype = dtype)
                vae.module.post_quant_conv = vae.module.post_quant_conv.to(device = model_management.vae_device(), dtype = dtype)
            elif hasattr(vae, 'spatial_vae') and hasattr(vae.spatial_vae, 'module') and hasattr(vae.spatial_vae.module, 'decoder') and hasattr(vae.spatial_vae.module, 'post_quant_conv') \
                and hasattr(vae, 'temporal_vae') and hasattr(vae.temporal_vae, 'decoder') and hasattr(vae.temporal_vae, 'post_quant_conv'):
                vae.spatial_vae.module.decoder = vae.spatial_vae.module.decoder.to(device = model_management.vae_device(), dtype = dtype)
                vae.spatial_vae.module.post_quant_conv = vae.spatial_vae.module.post_quant_conv.to(device = model_management.vae_device(), dtype = dtype)
                vae.temporal_vae.decoder = vae.temporal_vae.decoder.to(device = model_management.vae_device(), dtype = dtype)
                vae.temporal_vae.post_quant_conv = vae.temporal_vae.post_quant_conv.to(device = model_management.vae_device(), dtype = dtype)
                vae.scale = vae.scale.to(device = model_management.vae_device(), dtype = dtype)
                vae.shift = vae.shift.to(device = model_management.vae_device(), dtype = dtype)
            else:
                vae = vae.to(device = model_management.vae_device(), dtype = dtype)
        if latents["samples"].device != model_management.vae_device() or latents["samples"].dtype != dtype:
            latents["samples"] = latents["samples"].to(device = model_management.vae_device(), dtype = dtype)
        
        images = vae.decode(latents["samples"], num_frames=0) # 0 means always no time padding as num_frames are calculated by duration seconds
        
        if vae.device != model_management.vae_offload_device():
            if hasattr(vae, 'module') and hasattr(vae.module, 'decoder') and hasattr(vae.module, 'post_quant_conv'):
                vae.module.decoder = vae.module.decoder.to(device = model_management.vae_offload_device())
                vae.module.post_quant_conv = vae.module.post_quant_conv.to(device = model_management.vae_offload_device())
            elif hasattr(vae, 'spatial_vae') and hasattr(vae.spatial_vae, 'module') and hasattr(vae.spatial_vae.module, 'decoder') and hasattr(vae.spatial_vae.module, 'post_quant_conv') \
                and hasattr(vae, 'temporal_vae') and hasattr(vae.temporal_vae, 'decoder') and hasattr(vae.temporal_vae, 'post_quant_conv'):
                vae.spatial_vae.module.decoder = vae.spatial_vae.module.decoder.to(device = model_management.vae_offload_device())
                vae.spatial_vae.module.post_quant_conv = vae.spatial_vae.module.post_quant_conv.to(device = model_management.vae_offload_device())
                vae.temporal_vae.decoder = vae.temporal_vae.decoder.to(device = model_management.vae_offload_device())
                vae.temporal_vae.post_quant_conv = vae.temporal_vae.post_quant_conv.to(device = model_management.vae_offload_device())
                vae.scale = vae.scale.to(device = model_management.vae_offload_device())
                vae.shift = vae.shift.to(device = model_management.vae_offload_device())
            else:
                vae = vae.to(device = model_management.vae_offload_device())
        if latents["samples"].device != olddevice or latents["samples"].dtype != olddtype:
            latents["samples"] = latents["samples"].to(device = olddevice, dtype = olddtype)
        if images.device != olddevice or images.dtype != olddtype:
            images = images.to(device = olddevice, dtype = olddtype)
        
        images = (images / 2.0 + 0.5).clamp(0,1).permute(0, 2, 3, 4, 1).squeeze(0).contiguous()
        return (images,)


NODE_CLASS_MAPPINGS = {
    "OpenSoraLoader":OpenSoraLoader,
    "OpenSoraTextEncoder":OpenSoraTextEncoder,
    "OpenSoraSampler":OpenSoraSampler,
    "OpenSoraDecoder":OpenSoraDecoder,
    "OpenSoraEncoder":OpenSoraEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenSoraLoader":"(Down)Load Open Sora Model",
    "OpenSoraTextEncoder":"Open Sora Text Encoder",
    "OpenSoraSampler":"Open Sora Sampler",
    "OpenSoraDecoder":"Open Sora Decoder",
    "OpenSoraEncoder":"Open Sora Encoder",
}
