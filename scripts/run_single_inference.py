import argparse
import os
import torch
import json
import numpy as np
from yaml import load, Loader
from copy import deepcopy
from einops import rearrange
import matplotlib.pyplot as plt

# Add project root to path
import sys
sys.path.append(os.getcwd())

from utils import import_custom_class, save_video
from utils.model_utils import (
    load_condition_models, 
    load_latent_models, 
    load_vae_models, 
    load_diffusion_model, 
    count_model_parameters,
    unwrap_model
)
from utils.data_utils import read_img
from data.utils.statistics import StatisticInfo

def main():
    parser = argparse.ArgumentParser(description="Run single inference with Genie-Envisioner")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file (yaml)')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint file (safetensors)')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--prompt', type=str, default="Do something", help='Text instruction')
    parser.add_argument('--output_dir', type=str, default="inference_outputs", help='Directory to save outputs')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to run inference on')
    parser.add_argument('--state', type=str, default=None, help='Path to numpy file containing current robot state (optional)')
    
    args = parser.parse_args()
    
    # 1. Load Config
    print(f"Loading config from {args.config}")
    with open(args.config, "r") as f:
        config_dict = load(f, Loader=Loader)
    
    # Create a namespace from config dict
    config = argparse.Namespace(**config_dict)
    
    # Override config with inference settings
    config.load_weights = True
    config.load_diffusion_model_weights = True
    config.diffusion_model['model_path'] = args.ckpt
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device(args.device)
    dtype = torch.bfloat16 # Default to bf16 as per training config usually
    
    # 2. Load Models
    print("Initializing models...")
    
    # Tokenizer & Text Encoder
    tokenizer_class = import_custom_class(
        config.tokenizer_class, getattr(config, "tokenizer_class_path", "transformers")
    )
    textenc_class = import_custom_class(
        config.textenc_class, getattr(config, "textenc_class_path", "transformers")
    )
    
    cond_models = load_condition_models(
        tokenizer_class, textenc_class,
        config.pretrained_model_name_or_path if not hasattr(config, "tokenizer_pretrained_model_name_or_path") else config.tokenizer_pretrained_model_name_or_path,
        load_weights=config.load_weights
    )
    tokenizer = cond_models["tokenizer"]
    text_encoder = cond_models["text_encoder"].to(device, dtype=dtype).eval()
    
    # VAE
    vae_class = import_custom_class(
        config.vae_class, getattr(config, "vae_class_path", "transformers")
    )
    if getattr(config, 'vae_path', False):
        vae = load_vae_models(vae_class, config.vae_path).to(device, dtype=dtype).eval()
    else:
        vae = load_latent_models(vae_class, config.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        
    if hasattr(vae, "enable_slicing") and config.enable_slicing:
        vae.enable_slicing()
    if hasattr(vae, "enable_tiling") and config.enable_tiling:
        vae.enable_tiling()
        
    # Diffusion Model
    diffusion_model_class = import_custom_class(
        config.diffusion_model_class, getattr(config, "diffusion_model_class_path", "transformers")
    )
    print(f"Loading diffusion model from {args.ckpt}")
    diffusion_model = load_diffusion_model(
        model_cls=diffusion_model_class,
        model_dir=config.diffusion_model['model_path'],
        load_weights=True,
        **config.diffusion_model['config']
    ).to(device, dtype=dtype)
    
    # Scheduler
    diffusion_scheduler_class = import_custom_class(
        config.diffusion_scheduler_class, getattr(config, "diffusion_scheduler_class_path", "diffusers")
    )
    if hasattr(config, "diffusion_scheduler_args"):
        scheduler = diffusion_scheduler_class(**config.diffusion_scheduler_args)
    else:
        scheduler = diffusion_scheduler_class()
        
    # Pipeline
    pipeline_class = import_custom_class(
        config.pipeline_class, getattr(config, "pipeline_class_path", "diffusers")
    )
    
    pipe = pipeline_class(
        scheduler, vae, text_encoder, tokenizer, diffusion_model
    )
    
    # 3. Prepare Input
    print(f"Preparing input image from {args.image}")
    # LTX video usually uses 512x704 or similar.
    # Let's check config.data['train']['resolution'] or 'sample_size'
    h, w = 384, 512 # Default fallback
    if 'train' in config.data:
        if 'resolution' in config.data['train']:
            res = config.data['train']['resolution']
            if isinstance(res, list):
                h, w = res
            else:
                h, w = res, res
        elif 'sample_size' in config.data['train']:
            # sample_size is usually [h, w]
            res = config.data['train']['sample_size']
            if isinstance(res, list):
                h, w = res
            else:
                h, w = res, res
            
    raw_image = read_img(args.image, target_shape=(w, h)) # read_img takes (width, height)
    # raw_image shape: (3, H, W), range [-1, 1]
    
    # Prepare image tensor: (BV, C, T, H, W)
    # We assume single view, single frame input for now.
    # If model expects multiple views, we duplicate.
    n_view = len(config.data['train'].get('valid_cam', ['default']))
    
    # (C, H, W) -> (1, C, 1, H, W)
    image_tensor = raw_image.unsqueeze(0).unsqueeze(2)
    
    # Duplicate for n_view
    if n_view > 1:
        image_tensor = image_tensor.repeat(n_view, 1, 1, 1, 1)
        
    # Prepare State if needed
    history_action_state = None
    add_state = getattr(config, "add_state", False)
    
    # Load Statistics
    stats = StatisticInfo
    if 'val' in config.data and config.data['val'].get('stat_file', None) is not None:
        stat_file = config.data['val']['stat_file']
        if os.path.exists(stat_file):
            print(f"Loading statistics from {stat_file}")
            with open(stat_file, "r") as f:
                stats = json.load(f)
        else:
            print(f"Warning: stat_file {stat_file} not found, using default StatisticInfo")

    domain_name = config.data['train']['domains'][0] if 'domains' in config.data['train'] else "agibotworld"
    action_space = config.data['train'].get('action_space', 'joint')
    action_type = config.data['train'].get('action_type', 'absolute')
    
    state_key = f"{domain_name}_state_{action_space}"
    if state_key not in stats:
        # Fallback to generic key if domain specific not found
        print(f"Warning: {state_key} not found in stats, trying to infer...")
        # Try to find any key ending with _state_{action_space}
        for k in stats.keys():
            if k.endswith(f"_state_{action_space}"):
                state_key = k
                break
    
    if state_key in stats:
        sta_mean = np.array(stats[state_key]["mean"])
        sta_std = np.array(stats[state_key]["std"])
    else:
        print("Error: Could not find state statistics. State normalization will be incorrect.")
        sta_mean = 0
        sta_std = 1

    if add_state:
        print("Model requires state input.")
        if args.state and os.path.exists(args.state):
            print(f"Loading state from {args.state}")
            current_state = np.load(args.state)
        else:
            print("Warning: No state file provided or file not found. Using Mean State (zeros after normalization).")
            # Use mean state (which becomes 0 after normalization)
            # We need the dimension. Usually action_dim + gripper? Or just action_dim.
            # Let's infer from stats mean length
            if isinstance(sta_mean, np.ndarray):
                current_state = sta_mean # So that (state - mean) / std = 0
            else:
                # Fallback
                action_dim = config.diffusion_model["config"].get("action_in_channels", 14)
                current_state = np.zeros(action_dim)

        # Normalize state
        normed_state = (current_state - sta_mean) / sta_std
        # To tensor: (1, 1, C)
        history_action_state = torch.from_numpy(normed_state).float().to(device, dtype=dtype)
        if history_action_state.ndim == 1:
            history_action_state = history_action_state.unsqueeze(0).unsqueeze(0)
        elif history_action_state.ndim == 2:
            history_action_state = history_action_state.unsqueeze(0)
            
    # 4. Run Inference
    print(f"Running inference with prompt: '{args.prompt}'")
    
    # Determine parameters
    n_prev = config.data['train'].get('n_previous', 1)
    action_chunk = config.data['train'].get('action_chunk', 10)
    num_inference_steps = getattr(config, "num_inference_step", 50) # Note: yaml uses 'step' singular often
    pixel_wise_timestep = getattr(config, "pixel_wise_timestep", True)
    
    # We force return_video=True and return_action=True to see everything
    preds = pipe.infer(
        image=image_tensor,
        prompt=args.prompt,
        height=h,
        width=w,
        num_inference_steps=num_inference_steps,
        decode_timestep=0.03,
        decode_noise_scale=0.025,
        guidance_scale=1.0, ### close CFG for action-prediction
        n_view=n_view,
        return_action=True,
        return_video=True,
        n_prev=n_prev,
        action_chunk=action_chunk,
        action_dim=config.diffusion_model["config"].get("action_in_channels", 14),
        n_chunk=1, # Generate 1 chunk of video/action
        history_action_state=history_action_state,
        pixel_wise_timestep=pixel_wise_timestep
    )[0]
    
    # 5. Save Outputs
    print(f"Saving outputs to {args.output_dir}")
    
    # Save Video
    if 'video' in preds:
        video = preds['video'].data.cpu()
        # video shape: (b*v, c, t, h, w)
        # Rearrange to (b, c, t, h, v*w) for visualization if multi-view
        video = rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)
        save_path = os.path.join(args.output_dir, "output_video.mp4")
        save_video(video[0], save_path, fps=8)
        print(f"Saved video to {save_path}")
        
    # Save Actions
    if 'action' in preds:
        # actions shape: (b, t, c)
        raw_actions = preds['action'].data.cpu().numpy()
        
        # Denormalize actions
        if action_type == "delta":
            act_key = f"{domain_name}_delta_{action_space}"
        else:
            act_key = f"{domain_name}_{action_space}"
            
        if act_key not in stats:
             # Fallback
            print(f"Warning: {act_key} not found in stats. Trying to infer...")
            for k in stats.keys():
                if k.endswith(f"_{action_space}") and ("delta" in k) == ("delta" in act_key):
                    act_key = k
                    break
        
        if act_key in stats:
            act_mean = np.array(stats[act_key]["mean"])
            act_std = np.array(stats[act_key]["std"])
            
            # Expand dims for broadcasting: (1, 1, C)
            act_mean = act_mean[None, None, :]
            act_std = act_std[None, None, :]
            
            denorm_actions = raw_actions * act_std + act_mean
            
            # If delta, we need to integrate (cumsum) + initial state
            if action_type == "delta":
                if args.state and os.path.exists(args.state):
                    # current_state is already loaded above
                    pass 
                else:
                    # If no state provided, assume starting from 0 or mean state (unnormalized)
                    # But wait, if we used mean state for input, that means we are at the "average pose".
                    # Let's just use the current_state we derived earlier (which might be mean state)
                    pass
                
                # Integrate
                # We need to know arm dims vs gripper dims. 
                # Usually gripper is last dim or last of each arm.
                # Assuming simple cumsum for now for visualization
                # For precise robot control, you need to know the structure (e.g. 14 dims = 7 left + 7 right)
                # Here we just do cumsum + init_state
                
                # current_state shape (C,) -> (1, 1, C)
                init_state = current_state[None, None, :]
                denorm_actions = np.cumsum(denorm_actions, axis=1) + init_state
                
        else:
            print("Warning: Could not find action statistics. Actions are raw normalized values.")
            denorm_actions = raw_actions

        save_path = os.path.join(args.output_dir, "output_actions.npy")
        np.save(save_path, denorm_actions)
        print(f"Saved actions to {save_path}")
        
        # Simple plot
        plt.figure(figsize=(10, 5))
        for dim in range(denorm_actions.shape[-1]):
            plt.plot(denorm_actions[0, :, dim], label=f'Dim {dim}')
        plt.title("Predicted Actions")
        # plt.legend() # Too many lines maybe
        plt.savefig(os.path.join(args.output_dir, "output_actions_plot.png"))
        print(f"Saved action plot to {os.path.join(args.output_dir, 'output_actions_plot.png')}")

if __name__ == "__main__":
    main()
