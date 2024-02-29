import json
import random
import os, io

import gradio as gr
import numpy as np
import torch
import tempfile
from PIL import Image
from torchvision import transforms
from rembg import remove
import zipfile

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits




title = 'A simple tool for efficient background remove'

with gr.Blocks(title=title).queue() as root:
    # github_banner_path = 'https://galaxyfs-in-dev.dev.ihuman.com/nas/ai-tools/bgremove_title.png'
    # gr.HTML(f'<p align="center"><a href="https://github.com/uiyo/BGremoval"><img src={github_banner_path} width="1200"/></a></p>')
    gr.Markdown("> 批量图片背景移除工具")
    with gr.Row():
        gallery = gr.Gallery(label="segemented images", show_label=False, visible=True, elem_id='gallery', columns = [3], height='auto', interactive=False)
        upload = gr.Files(label="上传图片", file_types=["image"])
        def show_processed_images(files):
            if files is None:
                return gr.update(value=None)
            # 确保 files 是一个列表
            if not isinstance(files, list):
                files = [files]
            image_list = []

            for im in files:
                # 转换成PIL格式
                orig_im = Image.open(im.name)
                proc_im = remove(orig_im, alpha_matting=True, post_process_mask=True)
                image_list.append(proc_im)

            return gr.update(value=image_list)
        upload.change(fn=show_processed_images, inputs=upload, outputs=gallery, show_progress=True)
    with gr.Row():
        with gr.Column():
            download = gr.Button("打包所有", visible=True, elem_id='download_all')
        with gr.Column():    
            download_file = gr.File(label='下载', visible=False)
        def download_all(gallery, upload_files):
            image_list = gallery

            temp_dir = "pics"
            os.makedirs(temp_dir, exist_ok=True)
            if gallery is None: 
                return None
            zip_filename = "images.zip"
            with zipfile.ZipFile(os.path.join(temp_dir, zip_filename), "w") as zipf:
                for i,im in enumerate(image_list):
                    with open(im[0],'rb') as f:
                        zipf.writestr(upload_files[i].split('/')[-1], f.read())
            return gr.update(value=zip_filename, visible=True)
        download.click(download_all, inputs=[gallery,upload], outputs=download_file)
        
root.launch( 
    server_name='0.0.0.0',
    server_port=6666,
)