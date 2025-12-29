#!/usr/bin/bash

IP_ADDRESS_OF_SERVER="127.0.0.1"
PORT=8001
DOMAIN_NAME="RoboTwin"


python3 web_infer_scripts/main_server.py \
    -c configs/ltx_model/policy_model_lerobot.yaml \
    -w /data/dex/Genie-Envisioner/GE_ACT_ROBOTWIN_FINETUNE/2025_11_20_17_35_39/step_8000/diffusion_pytorch_model.safetensors \
    --add_state \
    --denoise_step 10 \
    --host ${IP_ADDRESS_OF_SERVER} \
    --port ${PORT} \
    --domain_name ${DOMAIN_NAME}
