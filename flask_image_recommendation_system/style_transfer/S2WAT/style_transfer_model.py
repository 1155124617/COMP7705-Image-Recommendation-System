import torch
import torch.nn as nn
from pathlib import Path

from style_transfer.S2WAT.model.configuration import TransModule_Config
from style_transfer.S2WAT.model.s2wat import S2WAT
from style_transfer.S2WAT.net import TransModule, Decoder_MVGG
from style_transfer.S2WAT.tools import save_transferred_imgs, Sample_Test_Net
from const.pathname import *

# Basic options
output_dir = Path(OUTPUT_IMAGE_DIR)
output_dir.mkdir(exist_ok=True, parents=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_model(model_loading_path):
    # Models Config
    transModule_config = TransModule_Config(
        nlayer=3,
        d_model=768,
        nhead=8,
        mlp_ratio=4,
        qkv_bias=False,
        attn_drop=0.,
        drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_first=True
    )

    # Hardware Setting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Models
    encoder = S2WAT(
        img_size=224,
        patch_size=2,
        in_chans=3,
        embed_dim=192,
        depths=[2, 2, 2],
        nhead=[3, 6, 12],
        strip_width=[2, 4, 7],
        drop_path_rate=0.,
        patch_norm=True
    )
    decoder = Decoder_MVGG(d_model=768, seq_input=True)
    transModule = TransModule(transModule_config)

    network = Sample_Test_Net(encoder, decoder, transModule)

    # Load the checkpoint
    print('loading checkpoint...')
    checkpoint = torch.load(model_loading_path, map_location=device)

    loss_count_interval = checkpoint['loss_count_interval']
    print('loading finished')

    network.encoder.load_state_dict(checkpoint['encoder'])
    network.decoder.load_state_dict(checkpoint['decoder'])
    network.transModule.load_state_dict(checkpoint['transModule'])
    network.to(device)

    return network


model_1 = get_model(MODEL_1_LOADING_PATH)
model_2 = get_model(MODEL_2_LOADING_PATH)


# ===============================================Execute Style Transfer===============================================
def do_style_transfer(model_type, input_content_image_dir, input_style_image_dir, output_image_dir):
    if model_type == 'car':
        model = model_2
    else:
        # default model is model 1
        model = model_1

    save_transferred_imgs(model, input_content_image_dir, input_style_image_dir, output_image_dir, device=device)
