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


# ===============================================Execute Style Transfer===============================================

def do_style_transfer(input_style_image_dir = INPUT_STYLE_IMAGE_DIR):
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    checkpoint = torch.load(MODEL_LOADING_PATH, map_location=device)

    loss_count_interval = checkpoint['loss_count_interval']
    print('loading finished')

    network.encoder.load_state_dict(checkpoint['encoder'])
    network.decoder.load_state_dict(checkpoint['decoder'])
    network.transModule.load_state_dict(checkpoint['transModule'])
    network.to(device)
    save_transferred_imgs(network, INPUT_CONTENT_IMAGE_DIR, input_style_image_dir, OUTPUT_IMAGE_DIR, device=device)
