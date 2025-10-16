import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, BitsAndBytesConfig, CLIPImageProcessor

from model.CityAnchor import LISAForCausalLM as LISAForCausalLM_Grounding
from model.llava import conversation as conversation_lib
from model.llava.mm_utils import tokenizer_image_token
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX)
from lib.config import CONF
from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
import open3d as o3d
import gradio as gr
import plotly.graph_objects as go
import json
import rasterio

from model.LISA_ROI import LISAForCausalLM as LISAForCausalLM_ROI
from model.llava_ROI import conversation as conversation_lib
from model.llava_ROI.mm_utils import tokenizer_image_token as tokenizer_image_token_ROI


def parse_args(args):

    parser = argparse.ArgumentParser(description="CityAnchor")
    # For pipeline 
    parser.add_argument("--no_roi", action="store_true", default=False)

    # For stage 1
    parser.add_argument("--version_stage_1", default="./LISA-13B-llama2-v1")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str) # for ROI
    parser.add_argument(
        "--precision_stage_1",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference in stage 1",
    )
    parser.add_argument("--image_size_stage_1", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length_stage_1", default=512, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit_stage_1", action="store_true", default=False)
    parser.add_argument("--load_in_4bit_stage_1", action="store_true", default=False)
    parser.add_argument(
        "--conv_type_stage_1",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )

    # For stage 2
    parser.add_argument("--version", default="./LISA-13B-llama2-v1")
    # parser.add_argument("--version", default="./llava-llama-2-13b-chat-lightning-preview")
    parser.add_argument("--dome_save_path", default="./demo_output", type=str) # for grounding results
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--local-rank", default=0, type=int, help="node rank")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    x = (x - pixel_mean) / pixel_std
    # Pad
    h, w = x.shape[-2:]
    padh = img_size - h
    padw = img_size - w
    x = F.pad(x, (0, padw, 0, padh))
    return x

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def downsample_point_cloud(points, voxel_size = 0.25):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(points[:,3:])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = np.asarray(downsampled_pcd.colors)
    return downsampled_points,downsampled_colors

def init_model_ROI(args):
    # args = parse_args(args) # global
    # os.makedirs(args.vis_save_path, exist_ok=True)
    # os.makedirs(args.dome_save_path, exist_ok=True)

    # Create model for ROI (stage 1)
    global tokenizer_ROI
    tokenizer_ROI = AutoTokenizer.from_pretrained(
        args.version_stage_1,
        cache_dir=None,
        model_max_length=args.model_max_length_stage_1,
        padding_side="right",
        use_fast=False,
    )
    tokenizer_ROI.pad_token = tokenizer_ROI.unk_token
    args.seg_token_idx_ROI = tokenizer_ROI("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype_ROI = torch.float32
    if args.precision_stage_1 == "bf16":
        torch_dtype_ROI = torch.bfloat16
    elif args.precision_stage_1 == "fp16":
        torch_dtype_ROI = torch.half
    
    kwargs_ROI = {"torch_dtype": torch_dtype_ROI}
    if args.load_in_4bit_stage_1:
        kwargs_ROI.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit_stage_1:
        kwargs_ROI.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model_ROI = LISAForCausalLM_ROI.from_pretrained(
        args.version_stage_1, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx_ROI, **kwargs_ROI
    )

    model_ROI.config.eos_token_id = tokenizer_ROI.eos_token_id
    model_ROI.config.bos_token_id = tokenizer_ROI.bos_token_id
    model_ROI.config.pad_token_id = tokenizer_ROI.pad_token_id

    model_ROI.get_model().initialize_vision_modules(model_ROI.get_model().config)
    vision_tower_ROI = model_ROI.get_model().get_vision_tower()
    vision_tower_ROI.to(dtype=torch_dtype_ROI)

    if args.precision_stage_1 == "bf16":
        model_ROI = model_ROI.bfloat16().cuda()
    elif (
        args.precision_stage_1 == "fp16" and (not args.load_in_4bit_stage_1) and (not args.load_in_8bit_stage_1)
    ):
        vision_tower_ROI = model_ROI.get_model().get_vision_tower()
        model_ROI.model.vision_tower = None
        import deepspeed

        model_engine_ROI = deepspeed.init_inference(
            model=model_ROI,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model_ROI = model_engine_ROI.module
        model_ROI.model.vision_tower = vision_tower_ROI.half().cuda()
    elif args.precision_stage_1 == "fp32":
        model_ROI = model_ROI.float().cuda()

    # Create model for grounding
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM_Grounding.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    return model, model_ROI

def init_model(args):
    # args = parse_args(args) # global
    # os.makedirs(args.vis_save_path, exist_ok=True)
    # os.makedirs(args.dome_save_path, exist_ok=True)

    # Create model for grounding
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}
    if args.load_in_4bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    llm_int8_skip_modules=["visual_model"],
                ),
            }
        )
    elif args.load_in_8bit:
        kwargs.update(
            {
                "torch_dtype": torch.half,
                "quantization_config": BitsAndBytesConfig(
                    llm_int8_skip_modules=["visual_model"],
                    load_in_8bit=True,
                ),
            }
        )

    model = LISAForCausalLM_Grounding.from_pretrained(
        args.version, low_cpu_mem_usage=True, vision_tower=args.vision_tower, seg_token_idx=args.seg_token_idx, **kwargs
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype)

    if args.precision == "bf16":
        model = model.bfloat16().cuda()
    elif (
        args.precision == "fp16" and (not args.load_in_4bit) and (not args.load_in_8bit)
    ):
        vision_tower = model.get_model().get_vision_tower()
        model.model.vision_tower = None
        import deepspeed

        model_engine = deepspeed.init_inference(
            model=model,
            dtype=torch.half,
            replace_with_kernel_inject=True,
            replace_method="auto",
        )
        model = model_engine.module
        model.model.vision_tower = vision_tower.half().cuda()
    elif args.precision == "fp32":
        model = model.float().cuda()

    return model

def create_bbox_line(bbox):
    x, y, z = bbox[:3]
    w, l, h = bbox[3:6]
    
    corners = np.array([
        [x-w/2, y-l/2, z-h/2],
        [x+w/2, y-l/2, z-h/2],
        [x+w/2, y+l/2, z-h/2],
        [x-w/2, y+l/2, z-h/2],
        [x-w/2, y-l/2, z+h/2],
        [x+w/2, y-l/2, z+h/2],
        [x+w/2, y+l/2, z+h/2],
        [x-w/2, y+l/2, z+h/2]
    ])
    
    lines = []
    for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        lines.append(go.Scatter3d(x=corners[[i,j],0], y=corners[[i,j],1], z=corners[[i,j],2],
              mode='lines', line=dict(color='red', width=2),showlegend=False))
    return lines

def pc_vis(scene_id,Tar_object_id = None):

    pg_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".pth")
    coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, \
              landmark_names, landmark_ids, globalShift = torch.load(pg_file, weights_only=False)

    colors = colors.astype(np.float32) / 255  # model input is (0-1)
    mesh_vertices = np.concatenate([coords,colors], axis=1) # total point cloud for this scene
    points,colors = downsample_point_cloud(mesh_vertices,2)
    color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

    if Tar_object_id == None:
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=color_strings,  # Use the list of RGB strings for the marker colors
                    )
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectratio=dict(x=1, y=1, z=0.05)
                ),
                paper_bgcolor='rgb(255,255,255)'  # Set the background color to dark gray 50, 50, 50
            ),
          )
        
    else:
        Tar_points = mesh_vertices[instance_ids == Tar_object_id,:]
        Tar_object_bbox = instance_bboxes[Tar_object_id][:6]
        lines = create_bbox_line(Tar_object_bbox)
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=points[:, 0], y=points[:, 1], z=points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=color_strings,   
                    ),
                    showlegend=False
                )
            ] + lines,
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectratio=dict(x=1, y=1, z=0.02)
                ),
                paper_bgcolor='rgb(255,255,255)'  # Set the background color to dark gray 50, 50, 50
            )
          )
    return fig

def pc_grounding(scene_id,input_text):
    
    # Model eval status
    from lib.config import CONF
    from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
    DC = SensatUrbanDatasetConfig()            
    
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()  

    # Grounding!
    pg_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".pth")
    feat_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".json")
    landmark_feat_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+"_landmark.json")
    landmark_nearest_file = os.path.join(CONF.PATH.LANDMARK_DATA, scene_id+".landmark_feat.json")

    coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, \
            landmark_names, landmark_ids, globalShift = torch.load(pg_file, weights_only=False)

    with open(feat_file, 'r') as json_file:
        feats = json.load(json_file)

    with open(landmark_feat_file, 'r') as json_file:
        landmark_feat = json.load(json_file)
        
    with open(landmark_nearest_file, 'r') as json_file:
        landmark_nearest_feat = json.load(json_file) 
    
    mesh_vertices = np.concatenate([coords, colors], axis=1) # total point cloud for this scene
    instance_bboxes = np.stack([instance_bboxes[instance_id] for instance_id in sorted(instance_bboxes.keys()) if instance_id != -100])
    instance_labels = instance_ids
    semantic_labels = label_ids
    cand_instance_ids = np.array([cand_id for cand_id in np.unique(instance_labels) if cand_id != -100])
    cand_instance_ids = cand_instance_ids[:]

    print("cand_instance_ids:",cand_instance_ids)
    print("cand_instance_ids.shape:",cand_instance_ids.shape)
        
    lang_feat_list = []
    object_list = []

    for i_2 in cand_instance_ids:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []

        subimage=np.zeros((3,9,9))

        object_id = i_2
        prompt = input_text + " Please output segmentation mask."
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
      
        raster_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".tif")

        with rasterio.open(raster_file) as src:
            image = src.read()

        instance_bboxes_pixel = np.copy(instance_bboxes)
        instance_bboxes_pixel[:, 0] = instance_bboxes_pixel[:,0] / 0.1
        instance_bboxes_pixel[:, 1] = instance_bboxes_pixel[:,1] / 0.1
        instance_bboxes_pixel[:, 3] = instance_bboxes_pixel[:,3] / 0.1
        instance_bboxes_pixel[:, 4] = instance_bboxes_pixel[:,4] / 0.1
        arr = instance_bboxes[:, -1]
        row = find_rows_with_value(arr, object_id)

        if len(row) == 0:
            subimage = np.zeros((256, 256, 3))
        else:         
            X_pixel=int(instance_bboxes_pixel[row,0])
            Y_pixel=int(instance_bboxes_pixel[row,1])
            X_delta=int(instance_bboxes_pixel[row,3])
            Y_delta=int(instance_bboxes_pixel[row,4])
            Image_LH = max(X_delta,Y_delta)
            local_image = extract_subimage(image,  image.shape[2] - Y_pixel, X_pixel, Image_LH, Image_LH)

        local_image = np.transpose(local_image, (1, 2, 0)) # (1676, 2190, 3)
        image_np = local_image
        original_size_list = [image_np.shape[:2]]
        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # pc embedding
        pc_feats = []
        try:
            feat_instance = feats[str(int(object_id))]
        except KeyError:
            feat_instance = np.zeros((1, 1024)) # Unable to equip pre-trained pc_feats for object
        pc_feats.append(feat_instance) # len:1 
        pc_feats = np.vstack(pc_feats) # [1,1024]
        pc_feats = pc_feats[0]

        # landmark embedding
        landmark_feats = []
        try:
            landmark_feat_instance = landmark_feat[str(int(object_id))]
        except KeyError:
            landmark_feat_instance  = np.zeros((1, 128)) # Unable to equip pre-trained pc_feats for object
        landmark_feats.append(landmark_feat_instance)
        landmark_feats = np.vstack(landmark_feats)
        landmark_feats = landmark_feats[0]

        # nearest_pc embedding
        nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, 5) # nearest instance finding
        while len(nearest_instance_ids) < 5: # K value 
            nearest_instance_ids.append(object_id)
        pc_feats_nearest = []
        for i in nearest_instance_ids:
            feat_instance = np.zeros((1, 1024))
            try:
                feat_instance = feats[str(int(i))]
            except KeyError:
                feat_instance = np.zeros((1, 1024))
            pc_feats_nearest.append(feat_instance)
        pc_feats_nearest = np.vstack(pc_feats_nearest)

        # K-nearest landmark - BiGRU feat
        landmark_feats_nearest = []
        for i in nearest_instance_ids:
            landmark_feat_instance = np.zeros((1, 128))
            try:
                landmark_feat_instance = landmark_nearest_feat[str(int(i))]
            except KeyError:
                landmark_feat_instance = np.zeros((1, 128))
            landmark_feats_nearest.append(landmark_feat_instance)
        landmark_feats_nearest = np.vstack(landmark_feats_nearest)

        lang_ans_object = model.evaluate(
            pc_feats,
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )

        lang_feat_list.append(lang_ans_object)
        object_list.append(i_2)

    list_combined = list(zip(object_list, lang_feat_list))
    list_sorted_combined = sorted(list_combined, key=lambda x: x[1], reverse=True)

    list_sorted_object, list_sorted_lang = zip(*list_sorted_combined)
    list_sorted_object = list(list_sorted_object)
    list_sorted_lang = list(list_sorted_lang)

    print("Sorted object:", list_sorted_object)
    print("Sorted lang:", list_sorted_lang)

    colors = colors.astype(np.float32) / 255  # model input is (0-1)
    mesh_vertices = np.concatenate([coords, colors], axis=1) # total point cloud for this scene
    print("mesh_vertices.shape:",mesh_vertices.shape)
    
    scene_plot = pc_vis(scene_id,list_sorted_object[0]) 

    Top5_plots = []

    for i in range(5):
      Tar_object_id = list_sorted_object[i]
      Tar_object_lang = list_sorted_lang[i]

      points = mesh_vertices[instance_ids==Tar_object_id,:]
      color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in points[:,3:]]
      
      fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=color_strings,  # Use the list of RGB strings for the marker colors
                )
            )
        ],
        layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectratio=dict(x=1, y=1, z=0.2)
                ),
                paper_bgcolor='rgb(255,255,255)',  # Set the background color to dark gray 50, 50, 50
                annotations=[
                    dict(
                        x=0.95,
                        y=0.95,
                        xref='paper',
                        yref='paper',
                        text='object id:{} | score:{:.3f}'.format(Tar_object_id,Tar_object_lang.float().cpu().numpy()[0]),
                        font=dict(
                            size=16,
                            color="Black"
                        ),
                        showarrow=False,
                        xanchor='right',
                        align="right"
                    )
                  ],
                
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=True,  
                height=225 
            ),
        
      )
      Top5_plots.append(fig)

    return scene_plot,*Top5_plots

def pc_grounding_with_ROI(scene_id, input_text):
    
    # Model eval status
    from lib.config import CONF
    from data.sensaturban.model_util_sensaturban import SensatUrbanDatasetConfig
    DC = SensatUrbanDatasetConfig()            
    
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank)
    clip_image_processor = CLIPImageProcessor.from_pretrained(model.config.vision_tower)
    transform = ResizeLongestSide(args.image_size)
    model.eval()  

    vision_tower_ROI = model_ROI.get_model().get_vision_tower()
    vision_tower_ROI.to(device=args.local_rank)
    clip_image_processor_ROI = CLIPImageProcessor.from_pretrained(model_ROI.config.vision_tower)
    transform_ROI = ResizeLongestSide(args.image_size)
    model_ROI.eval()  

    # ROI
    conv_stage_1 = conversation_lib.conv_templates[args.conv_type_stage_1].copy()
    conv_stage_1.messages = []
    prompt_stage_1 = DEFAULT_IMAGE_TOKEN + "\n" + input_text

    if args.use_mm_start_end:
        replace_token = (
            DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
        )
        prompt_stage_1 = prompt_stage_1.replace(DEFAULT_IMAGE_TOKEN, replace_token)

    conv_stage_1.append_message(conv_stage_1.roles[0], prompt_stage_1)
    conv_stage_1.append_message(conv_stage_1.roles[1], "")
    prompt_stage_1 = conv_stage_1.get_prompt()

    image_path = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".tif")
    image_np_stage_1 = cv2.imread(image_path)
    image_np_stage_1 = cv2.cvtColor(image_np_stage_1, cv2.COLOR_BGR2RGB)
    original_size_list_stage_1 = [image_np_stage_1.shape[:2]]

    image_clip_stage_1 = (
        clip_image_processor_ROI.preprocess(image_np_stage_1, return_tensors="pt")[
            "pixel_values"
        ][0]
        .unsqueeze(0)
        .cuda()
    )
    if args.precision_stage_1 == "bf16":
        image_clip_stage_1 = image_clip_stage_1.bfloat16()
    elif args.precision_stage_1 == "fp16":
        image_clip_stage_1 = image_clip_stage_1.half()
    else:
        image_clip_stage_1 = image_clip_stage_1.float()
    
    image_stage_1 = transform.apply_image(image_np_stage_1)
    resize_list_stage_1 = [image_stage_1.shape[:2]]

    image_stage_1 = (
        preprocess(torch.from_numpy(image_stage_1).permute(2, 0, 1).contiguous())
        .unsqueeze(0)
        .cuda()
    )
    if args.precision_stage_1 == "bf16":
        image_stage_1 = image_stage_1.bfloat16()
    elif args.precision_stage_1 == "fp16":
        image_stage_1 = image_stage_1.half()
    else:
        image_stage_1 = image_stage_1.float()

    input_ids_ROI = tokenizer_image_token(prompt_stage_1, tokenizer_ROI, return_tensors="pt")
    input_ids_ROI = input_ids_ROI.unsqueeze(0).cuda()

    output_ids_ROI, pred_masks_ROI = model_ROI.evaluate(
        image_clip_stage_1,
        image_stage_1,
        input_ids_ROI,
        resize_list_stage_1,
        original_size_list_stage_1,
        max_new_tokens=512,
        tokenizer=tokenizer_ROI,
    )
    output_ids_ROI = output_ids_ROI[0][output_ids_ROI[0] != IMAGE_TOKEN_INDEX]
    text_output_ROI = tokenizer_ROI.decode(output_ids_ROI, skip_special_tokens=False)
    text_output_ROI = text_output_ROI.replace("\n", "").replace("  ", " ")
    pred_mask_ROI = pred_masks_ROI[0].detach().cpu().numpy()[0]

    pred_mask_ROI_standardized = (pred_mask_ROI - np.mean(pred_mask_ROI)) / np.std(pred_mask_ROI)
    min_standardized = np.min(pred_mask_ROI_standardized)
    max_standardized = np.max(pred_mask_ROI_standardized)
    pred_mask_ROI_scaled = 2 * (pred_mask_ROI_standardized - min_standardized) / (max_standardized - min_standardized) - 1

    # Grounding!
    pg_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".pth")
    feat_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".json")
    landmark_feat_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+"_landmark.json")
    landmark_nearest_file = os.path.join(CONF.PATH.LANDMARK_DATA, scene_id+".landmark_feat.json")

    coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, \
            landmark_names, landmark_ids, globalShift = torch.load(pg_file, weights_only=False)

    with open(feat_file, 'r') as json_file:
        feats = json.load(json_file)

    with open(landmark_feat_file, 'r') as json_file:
        landmark_feat = json.load(json_file)
        
    with open(landmark_nearest_file, 'r') as json_file:
        landmark_nearest_feat = json.load(json_file)
        
    mesh_vertices = np.concatenate([coords, colors], axis=1) # total point cloud for this scene
    
    instance_bboxes = np.stack([instance_bboxes[instance_id] for instance_id in sorted(instance_bboxes.keys()) if instance_id != -100])
    instance_labels = instance_ids
    semantic_labels = label_ids
    cand_instance_ids = np.array([cand_id for cand_id in np.unique(instance_labels) if cand_id != -100])
    cand_instance_ids = cand_instance_ids[:]

    print("cand_instance_ids:",cand_instance_ids)
    print("cand_instance_ids.shape:",cand_instance_ids.shape)
        
    lang_feat_list = []
    object_list = []

    for i_2 in cand_instance_ids:
        conv = conversation_lib.conv_templates[args.conv_type].copy()
        conv.messages = []
        subimage=np.zeros((3,9,9))
        object_id = i_2

        prompt = input_text + " Please output segmentation mask."
        # prompt = input_text
        
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt
        if args.use_mm_start_end:
            replace_token = (
                DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            )
            prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()
        raster_file = os.path.join(CONF.PATH.SCAN_DATA, scene_id+".tif")

        with rasterio.open(raster_file) as src:
            image = src.read()

        instance_bboxes_pixel = np.copy(instance_bboxes)
        instance_bboxes_pixel[:, 0] = instance_bboxes_pixel[:,0] / 0.1
        instance_bboxes_pixel[:, 1] = instance_bboxes_pixel[:,1] / 0.1
        instance_bboxes_pixel[:, 3] = instance_bboxes_pixel[:,3] / 0.1
        instance_bboxes_pixel[:, 4] = instance_bboxes_pixel[:,4] / 0.1
        arr = instance_bboxes[:,-1]
        row = find_rows_with_value(arr, object_id)

        if len(row) == 0:
            subimage = np.zeros((256, 256, 3)) # pos -> 0
        else:         
            X_pixel=int(instance_bboxes_pixel[row, 0])
            Y_pixel=int(instance_bboxes_pixel[row, 1])
            X_delta=int(instance_bboxes_pixel[row, 3])
            Y_delta=int(instance_bboxes_pixel[row, 4])
            
            Image_LH = max(X_delta, Y_delta)
            local_image = extract_subimage(image, image.shape[2] - Y_pixel, X_pixel, Image_LH, Image_LH)
        
        # print(pred_mask_ROI_scaled[image.shape[2] - Y_pixel, X_pixel])
        if pred_mask_ROI_scaled[image.shape[2] - Y_pixel, X_pixel] < 0.3:
            continue

        local_image = np.transpose(local_image, (1, 2, 0))

        image_np = local_image
        original_size_list = [image_np.shape[:2]]
        image_clip = (
            clip_image_processor.preprocess(image_np, return_tensors="pt")[
                "pixel_values"
            ][0]
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image_clip = image_clip.bfloat16()
        elif args.precision == "fp16":
            image_clip = image_clip.half()
        else:
            image_clip = image_clip.float()

        image = transform.apply_image(image_np)
        resize_list = [image.shape[:2]]

        image = (
            preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
            .unsqueeze(0)
            .cuda()
        )
        if args.precision == "bf16":
            image = image.bfloat16()
        elif args.precision == "fp16":
            image = image.half()
        else:
            image = image.float()

        input_ids = tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        input_ids = input_ids.unsqueeze(0).cuda()

        # pc embedding
        pc_feats = []
        try:
            feat_instance = feats[str(int(object_id))]
        except KeyError:
            feat_instance = np.zeros((1, 1024)) # Unable to equip pre-trained pc_feats for object
        pc_feats.append(feat_instance) # len:1 
        pc_feats = np.vstack(pc_feats) # [1,1024]
        pc_feats = pc_feats[0]

        # landmark embedding
        landmark_feats = []
        try:
            landmark_feat_instance = landmark_feat[str(int(object_id))]
        except KeyError:
            landmark_feat_instance  = np.zeros((1, 128)) # Unable to equip pre-trained pc_feats for object
        landmark_feats.append(landmark_feat_instance)
        landmark_feats = np.vstack(landmark_feats)
        landmark_feats = landmark_feats[0]

        # nearest_pc embedding
        nearest_instance_ids = find_nearest_instances(instance_bboxes, object_id, 5) # nearest instance finding
        while len(nearest_instance_ids) < 5: # K value 
            nearest_instance_ids.append(object_id)
        pc_feats_nearest = []
        for i in nearest_instance_ids:
            feat_instance = np.zeros((1, 1024))
            try:
                feat_instance = feats[str(int(i))]
            except KeyError:
                feat_instance = np.zeros((1, 1024))
            pc_feats_nearest.append(feat_instance)
        pc_feats_nearest = np.vstack(pc_feats_nearest)

        # K-nearest landmark - BiGRU feat
        landmark_feats_nearest = []
        for i in nearest_instance_ids:
            landmark_feat_instance = np.zeros((1, 128))
            try:
                landmark_feat_instance = landmark_nearest_feat[str(int(i))]
            except KeyError:
                landmark_feat_instance = np.zeros((1, 128))
            landmark_feats_nearest.append(landmark_feat_instance)
        landmark_feats_nearest = np.vstack(landmark_feats_nearest)

        lang_ans_object = model.evaluate(
            pc_feats,
            landmark_feats,
            pc_feats_nearest,
            landmark_feats_nearest,
            image_clip,
            image,
            input_ids,
            resize_list,
            original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )

        lang_feat_list.append(lang_ans_object)
        object_list.append(i_2)

    print(lang_feat_list)
    print(object_list)

    list_combined = list(zip(object_list, lang_feat_list))
    list_sorted_combined = sorted(list_combined, key=lambda x: x[1], reverse=True)

    list_sorted_object, list_sorted_lang = zip(*list_sorted_combined)
    list_sorted_object = list(list_sorted_object)
    list_sorted_lang = list(list_sorted_lang)

    print("Sorted object:", list_sorted_object)
    print("Sorted lang:", list_sorted_lang)

    colors = colors.astype(np.float32) / 255  # model input is (0-1)
    mesh_vertices = np.concatenate([coords, colors], axis=1) # total point cloud for this scene
    print("mesh_vertices.shape:",mesh_vertices.shape)
    
    scene_plot = pc_vis(scene_id,list_sorted_object[0]) 
    Top5_plots = []

    for i in range(5):

      if i >= len(list_sorted_object):
        fig = go.Figure()
        Top5_plots.append(fig)
        continue

      Tar_object_id = list_sorted_object[i]
      Tar_object_lang = list_sorted_lang[i]

      points = mesh_vertices[instance_ids==Tar_object_id,:]
      color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in points[:,3:]]
      
      fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=1.5,
                    color=color_strings,  # Use the list of RGB strings for the marker colors
                )
            )
        ],
        layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectratio=dict(x=1, y=1, z=0.2)
                ),
                paper_bgcolor='rgb(255,255,255)',  # Set the background color to dark gray 50, 50, 50
                annotations=[
                    dict(
                        x=0.95,
                        y=0.95,
                        xref='paper',
                        yref='paper',
                        text='object id:{} | score:{:.3f}'.format(Tar_object_id,Tar_object_lang.float().cpu().numpy()[0]),
                        font=dict(
                            size=16,
                            color="Black"
                        ),
                        showarrow=False,
                        xanchor='right',
                        align="right"
                    )
                  ],
                
                margin=dict(l=0, r=0, t=0, b=0),
                autosize=True,  
                height=225 
            ),
        
      )
      Top5_plots.append(fig)

    return scene_plot, *Top5_plots
  
def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes, train_scenes_to_use=None, val_scenes_to_use=None):
    # get initial scene list
    if train_scenes_to_use is not None:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train if data["scene_id"] in train_scenes_to_use])))
    else:
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        
    if val_scenes_to_use is not None:
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val if data["scene_id"] in val_scenes_to_use])))
    else:        
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        
    if num_scenes == -1:
        num_scenes = len(train_scene_list)
    else:
        assert len(train_scene_list) >= num_scenes
        
    # slice train_scene_list
    train_scene_list = train_scene_list[:num_scenes]

    # filter data in chosen scenes
    new_scanrefer_train = []
    for data in scanrefer_train:
        if data["scene_id"] in train_scene_list:
            new_scanrefer_train.append(data)

    new_scanrefer_val = []
    for data in scanrefer_val:
        if data["scene_id"] in val_scene_list:
            new_scanrefer_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list


## def for construction of our cityrefer dataset (Query, 3D instance, 2D image)
def one_hot(length, position):
    zeros = [0 for _ in range(length)]
    zeros[position] = 1
    zeros = np.array(zeros)
    return zeros


def shuffle_items_with_indices(lst):
    indices = list(range(len(lst)))  
    random.shuffle(indices) 
    shuffled_items = [lst[idx] for index, idx in enumerate(indices)]
    return shuffled_items, indices


def extract_subimage(image, center_x, center_y, width, height):

    start_x = max(center_x - width // 2, 0)
    end_x = min(center_x + width // 2, image.shape[2])
    start_y = max(center_y - height // 2, 0)
    end_y = min(center_y + height // 2, image.shape[1])
    subimage = image[:,  start_x:end_x,start_y:end_y]
    
    return subimage

def find_rows_with_value(arr, value):
    rows_with_value = np.where(arr == value)[0]
    return rows_with_value

def find_nearest_instances(arr, objectID,K):
    object_row = arr[arr[:, -1] == objectID]
    if object_row.shape[0] == 0:
        return []
    object_xyz = object_row[:, :3]
    distances = np.sqrt(np.sum((arr[:, :3] - object_xyz) ** 2, axis=1))
    distances[arr[:, -1] == objectID] = np.inf
    nearest_indices = np.argsort(distances)[:K]
    return arr[nearest_indices, -1]


  
if __name__ == "__main__":

  global args 
  args = parse_args(sys.argv[1:])
  print('loading model...')

  global model_ROI # for stage 1 (optional)
  global model # for stage 2

  if args.no_roi: 
      model = init_model(args)
  else: 
      model, model_ROI = init_model_ROI(args)

  with gr.Blocks(
    css="#pc_scene { height: 500px; width: 100%; } .grounding_plot { height: 250px; width: 200px; }"
  ) as demo:

    gr.Markdown(
            """
            # CityAnchor: City-scale 3D Visual Grounding with Multi-modality LLMs. 🚀
            ## If you think this demo interesting, please consider starring 🌟 our github repo. :)
            [[Project Page]] [[Paper]] [[Code]] 
            """
            )

    gr.Markdown(
            """
            ## Usage:
            1. Select one of the city-scale point cloud to visualize. 
            2. Click **Visualize**.
            3. Input a text description of the object you want to query (e.g. a red car.).
            4. Click **Grounding** (it might takes a few minutes).
            5. The grounding results will be displayed in the following frame. 
            """)
    
    scene_id_list = ["birmingham_block_4","birmingham_block_5","cambridge_block_21","birmingham_block_12","cambridge_block_10"]

    with gr.Row():
        scene_id = gr.Dropdown(choices = scene_id_list , 
                label = "Select a file to visualize",
                value = 'birmingham_block_4')
        input_text = gr.Textbox(label = 'Input a description of the object')
    with gr.Row():
        pc_vis_button = gr.Button("Visualize")
        start_btn = gr.Button('Grounding') 
    pc_scene = gr.Plot(label = f"Point Cloud City Scene",elem_id = "pc_scene")

    gr.Markdown(
            """
            ## The top 5 objects with the highest scores : 
            """)
    with gr.Row(): 
        Top1 = gr.Plot(label=f"Grounding Result Top 1",elem_classes=["grounding_plot"]) 
        Top2 = gr.Plot(label=f"Grounding Result Top 2",elem_classes=["grounding_plot"]) 
    with gr.Row():
        Top3 = gr.Plot(label=f"Grounding Result Top 3",elem_classes=["grounding_plot"]) 
        Top4 = gr.Plot(label=f"Grounding Result Top 4",elem_classes=["grounding_plot"])
        Top5 = gr.Plot(label=f"Grounding Result Top 5",elem_classes=["grounding_plot"])

    pc_grounding_vis = [Top1,Top2,Top3,Top4,Top5]
    pc_vis_button.click(fn = pc_vis ,inputs = scene_id  ,outputs = pc_scene)

    if args.no_roi: 
        start_btn.click(fn = pc_grounding,
            inputs = [scene_id,input_text],
            outputs = [pc_scene]+ pc_grounding_vis)
    else: 
        start_btn.click(fn = pc_grounding_with_ROI,
            inputs = [scene_id,input_text],
            outputs = [pc_scene]+ pc_grounding_vis)

    demo.queue()
    demo.launch(share = True)
