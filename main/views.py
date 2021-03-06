from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from rest_framework.response import Response
from rest_framework.decorators import api_view, renderer_classes
# Create your views here.
import click
import re
import os
from typing import List, Optional
import torch
from main.stylegan2_ada_pytorch import dnnlib
from main.stylegan2_ada_pytorch import legacy
import boto3
from botocore.client import Config 
import PIL.Image
import numpy as np
from StyleGAN2_django.aws_config import (
    AWS_ACCESS_KEY,  
    AWS_SECRET_KEY,
    AWS_BUCKET_NAME,
    AWS_REGION,
    AWS_STORAGE_PREFIX
)
def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def generate_images(
    network_pkl,
    seeds,
    truncation_psi,
    noise_mode,
    outdir  
):
    """Generate images using pretrained network pickle.

    Examples:

    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl

    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl

    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    # device = torch.device('cuda')
    device = torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)

  

    # if seeds is None:
    #     ctx.fail('--seeds option is required when not using --projected-w')

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Generate images.
    file_list = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        # img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode, force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        filename = f'{outdir}/seed{seed:04d}_{str(truncation_psi)}.png'
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(filename)
        file_list.append(filename)
    return file_list
def home(request):
    return render(request, 'Mainpage/home.html', {})

@api_view(['POST'])
def create_image(request):
    try:
        if request.method == "POST":
            s3 = boto3.resource(    
                "s3",
                aws_access_key_id=AWS_ACCESS_KEY(),
                aws_secret_access_key=AWS_SECRET_KEY(),
                config=Config(signature_version="s3v4"),
            )
            processing_result = ''
            seeds_list = request.data['seeds']
            trunc = request.data["trunc"]
            network_pkl = './main/stylegan2_ada_pytorch/network-snapshot-000128_last.pkl'
            seeds = num_range(seeds_list)
            truncation_psi = float(trunc)
            file_list = generate_images(network_pkl, seeds,   truncation_psi, noise_mode ='const',outdir ='out')
            uploaded_filelist = []
            for file in file_list:
                with open(file, "rb") as data:
                    bucket_object_name = file.split("/")[-1]
                    s3.Bucket(AWS_BUCKET_NAME()).put_object(Key=bucket_object_name,Body=data,ACL="public-read")
                    uploaded_filelist.append(AWS_STORAGE_PREFIX() + bucket_object_name)
            return Response({'result': "success", 'data':uploaded_filelist})

    except Exception as error:
        return Response({'result': str(error)})