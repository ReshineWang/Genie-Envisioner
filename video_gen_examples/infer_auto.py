import os
import sys
import subprocess
import json
# Add current directory to path to import from infer.py
sys.path.insert(0, os.path.dirname(__file__))
# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
import cv2
import numpy as np
import torch
import pandas as pd
import math
import decord
from yaml import load, Loader
from einops import rearrange
from utils import save_video


from infer import prepare_model, load_config

def get_video_info_ffmpeg(video_path):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-count_packets',
        '-show_entries', 'stream=nb_read_packets',
        '-of', 'csv=p=0',
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        return int(output)
    except:
        return 0

def get_video_codec(video_path):
    cmd = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode('utf-8').strip()
        return output
    except:
        return None

def read_frame_ffmpeg(video_path):
    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-vframes', '1',
        '-f', 'image2',
        '-v', 'error',
        'pipe:1'
    ]
    try:
        output = subprocess.check_output(cmd)
        # Convert bytes to numpy array
        nparr = np.frombuffer(output, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        print(f"FFmpeg fallback failed: {e}")
        return None

def load_first_frame(video_path, size=(256, 192)):
    # Check codec first to avoid noisy failures with decord/cv2 on AV1
    codec = get_video_codec(video_path)
    
    frame = None
    total_frames = 0
    
    if codec == 'av1':
        # Use ffmpeg directly for AV1
        frame = read_frame_ffmpeg(video_path)
        total_frames = get_video_info_ffmpeg(video_path)
        if frame is None:
             # If ffmpeg fails, we might as well try the others just in case, 
             # but usually this is the most robust method for AV1
             pass 
    
    if frame is None:
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total_frames = len(vr)
            frame = vr[0].asnumpy()
        except Exception as e:
            # Fallback to cv2 if decord fails (though decord is preferred)
            # Only print if we haven't tried ffmpeg yet or if we really expect it to work
            if codec != 'av1':
                print(f"Decord failed for {video_path}, trying cv2: {e}")
            try:
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    raise ValueError("cv2 read failed")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except Exception as e2:
                if codec != 'av1':
                    print(f"cv2 failed for {video_path}, trying ffmpeg: {e2}")
                frame = read_frame_ffmpeg(video_path)
                total_frames = get_video_info_ffmpeg(video_path)
                if frame is None:
                    raise ValueError(f"Could not read frame from {video_path} using any method")

    frame = cv2.resize(frame, size)
    frame = frame.astype(np.float32) / 255.0 * 2.0 - 1.0
    # H, W, C -> C, H, W
    frame = torch.from_numpy(np.transpose(frame, (2, 0, 1)))
    # C, H, W -> C, T, H, W (T=1)
    frame = frame.unsqueeze(1)
    return frame, total_frames

def infer_auto(
    config_file,
    video_root,
    metadata_file,
    output_root,
    device="cuda",
    interval=50
):
    args = load_config(config_file)
    
    # Override some args for inference if needed
    # args.data['train']['n_previous'] is used in infer.py, let's check what it is
    n_prev = args.data['train'].get('n_previous', 4)
    
    tokenizer, text_encoder, vae, diffusion_model, scheduler, pipe = prepare_model(args, device=device)
    
    # Get image size from config
    # sample_size is [H, W] or [W, H]? 
    # In infer.py: size=(args.data["train"]["sample_size"][1], args.data["train"]["sample_size"][0])
    # Config says sample_size: [192, 256] (H, W usually, but let's check infer.py usage)
    # infer.py: size=(256, 192) passed to cv2.resize (width, height)
    # config: sample_size: [192, 256]
    # So config is [H, W]. cv2.resize takes (W, H).
    h, w = args.data["train"]["sample_size"]
    size = (w, h)
    
    # Check valid cams to determine number of views
    valid_cams = args.data["train"].get("valid_cam", ["default"])
    num_views = len(valid_cams)
    
    # Load metadata
    if metadata_file.endswith('.csv'):
        df = pd.read_csv(metadata_file)
        data_items = []
        for i in range(len(df)):
            data_items.append(df.iloc[i])
        is_lerobot = False
    elif metadata_file.endswith('.jsonl'):
        with open(metadata_file, 'r') as f:
            data_items = [json.loads(line) for line in f]
        is_lerobot = True
    else:
        raise ValueError("metadata_file must be .csv or .jsonl")
    
    # Create output directory
    os.makedirs(output_root, exist_ok=True)
    
    SPATIAL_DOWN_RATIO = vae.spatial_compression_ratio
    TEMPORAL_DOWN_RATIO = vae.temporal_compression_ratio
    
    action_chunk = args.data['train'].get('action_chunk', 50)
    


    for i in range(0, len(data_items), interval):
        item = data_items[i]
        
        if is_lerobot:
            episode_index = item['episode_index']
            prompt = item['tasks'][0]
            video_filename = f"episode_{episode_index:06d}.mp4" # For logging
            
            # Determine chunk
            chunk_idx = episode_index // 1000
            chunk_str = f"chunk-{chunk_idx:03d}"
            
            # We need to load frames for all valid_cams
            frames = []
            total_frames = 0
            
            try:
                for cam_name in valid_cams:
                    # Construct path: video_root/videos/chunk-XXX/cam_name/episode_XXXXXX.mp4
                    video_path = os.path.join(video_root, "videos", chunk_str, cam_name, f"episode_{episode_index:06d}.mp4")
                    
                    if not os.path.exists(video_path):
                        raise FileNotFoundError(f"Video not found: {video_path}")
                        
                    frame, tf = load_first_frame(video_path, size=size)
                    frames.append(frame)
                    total_frames = tf # Assume all views have same length
                
                # Stack frames: V, C, 1, H, W
                obs = torch.stack(frames, dim=0)
                
                # Repeat for n_prev
                # obs: V, C, 1, H, W -> V, C, n_prev, H, W
                if n_prev > 1:
                    obs = obs.repeat(1, 1, n_prev, 1, 1)
                    
            except Exception as e:
                print(f"Error loading frames for episode {episode_index}: {e}")
                continue
                
        else:
            # CSV mode
            video_filename = item['video']
            prompt = item['prompt']
            
            video_path = os.path.join(video_root, video_filename)
            if not os.path.exists(video_path):
                print(f"Video not found: {video_path}, skipping...")
                continue
                
            try:
                # Load first frame
                frame, total_frames = load_first_frame(video_path, size=size) # C, 1, H, W
                
                # Repeat for n_prev
                if n_prev > 1:
                    frame = frame.repeat(1, n_prev, 1, 1)
                    
                # Repeat for num_views
                obs = frame.unsqueeze(0).repeat(num_views, 1, 1, 1, 1)
            except Exception as e:
                print(f"Error loading frame for {video_filename}: {e}")
                continue

        print(f"Processing {i}: {video_filename}")
        print(f"Prompt: {prompt}")
        
        try:
            n_chunk = math.ceil(total_frames / action_chunk)
            print(f"Video frames: {total_frames}, action_chunk: {action_chunk}, n_chunk: {n_chunk}")
            
            save_path = os.path.join(output_root, f"sample_{i}")
            os.makedirs(save_path, exist_ok=True)
            
            # Save prompt
            with open(os.path.join(save_path, "prompt.txt"), "w") as f:
                f.write(prompt)
            
            preds = pipe.infer(
                image=obs.to(device),
                prompt=[prompt],
                negative_prompt="",
                num_inference_steps=50,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                height=h,
                width=w,
                n_view=num_views,
                guidance_scale=1.0,
                return_action=False, # Assuming we don't need action for this visualization
                n_prev=n_prev,
                chunk=(args.data['train']['chunk']-1)//TEMPORAL_DOWN_RATIO+1,
                return_video=True,
                noise_seed=42,
                pixel_wise_timestep=args.pixel_wise_timestep,
                n_chunk=n_chunk,
                action_chunk=action_chunk,
                history_action_state=None,
            )[0]
            
            if 'video' in preds:
                video = preds['video'].data.cpu()
                # video shape: (b v) c t h w
                # rearrange to b c t h (v w) for saving side-by-side
                # b=1
                video_out = rearrange(video, '(b v) c t h w -> b c t h (v w)', v=num_views)[0]
                save_video(
                    video_out,
                    os.path.join(save_path, "video.mp4"),
                    fps=30 # Default fps
                )
                print(f"Saved to {save_path}/video.mp4")
                
        except Exception as e:
            print(f"Error processing {video_filename}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default='/data/dex/Genie-Envisioner/configs/ltx_model/video_model_lerobot.yaml')
    parser.add_argument('--video_root', type=str, default='/data/dex/RoboTwin/lerobot_data/huggingface/lerobot/RoboTwin')
    parser.add_argument('--metadata_file', type=str, default='/data/dex/RoboTwin/lerobot_data/huggingface/lerobot/RoboTwin/meta/episodes.jsonl')
    parser.add_argument('--output_root', type=str, default='/data/dex/Genie-Envisioner/output_RoboTwin_35000')
    parser.add_argument('--interval', type=int, default=50)
    
    args = parser.parse_args()
    
    infer_auto(
        args.config_file,
        args.video_root,
        args.metadata_file,
        args.output_root,
        interval=args.interval
    )
