import argparse
import torch
import numpy as np

def parser_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--FP16', action='store', type=bool, default=False)
    parser.add_argument('--RTN2', action='store', type=bool, default=False)
    parser.add_argument('--RTN3', action='store', type=bool, default=False)
    parser.add_argument('--RTN4', action='store', type=bool, default=False)
    parser.add_argument('--fuse_weights', action='store', type=bool, default=False)
    parser.add_argument('--RotateKV2', action='store', type=bool, default=False)
    parser.add_argument('--RotateKV3', action='store', type=bool, default=False)
    parser.add_argument('--RotateKV4', action='store', type=bool, default=False)
    parser.add_argument('--head_group_num', type=int, default=4)
    parser.add_argument('--generate_for_calibration', action='store', type=bool, default=False)
    parser.add_argument('--Grouped_Head_Key_Rotation', action='store', type=bool, default=False)   
    parser.add_argument('--Outlier_Aware_Key_Rotation', action='store', type=bool, default=False)   
    parser.add_argument('--Attention_Sink_Aware', action='store', type=bool, default=False)   
    
    parser.add_argument('--save_attention_scores', action='store', type=bool, default=False)
    parser.add_argument('--save_k_pre_rope', action='store', type=bool, default=False)
    parser.add_argument('--save_k_post_rope', action='store', type=bool, default=False)
    parser.add_argument('--save_massive_activations', action='store', type=bool, default=False)

    # General Arguments
    parser.add_argument('--model', type=str, default='llama2_7b', choices=["llama2_7b", "llama2_13b", "llama3_8b", "llama2_7b_80K", "mistral_7b"])
    parser.add_argument('--PPL_seq_length', type=int, default=4096, choices=[2048,4096])
    parser.add_argument('--attn_implementation', type=str, default="eager", choices=["flash_attention_2", "eager", "sdpa"])
    parser.add_argument('--use_safetensors', action='store', type=bool, default=False)
    
    # KV-Cache Quantization Arguments
    parser.add_argument('--k_bits', type=int, default=2)
    parser.add_argument('--k_groupsize', type=int, default=128)
    parser.add_argument('--k_clip_ratio', type=float, default=1) 
    parser.add_argument('--v_bits', type=int, default=2)
    parser.add_argument('--v_groupsize', type=int, default=128)
    parser.add_argument('--v_clip_ratio', type=float, default=1)

    # GSM8K
    parser.add_argument("--seed", type=int, default=42, help="Random seed.",)


    args = parser.parse_args()
    if args.save_attention_scores or args.save_k_pre_rope or args.save_k_post_rope or args.save_massive_activations:
        if not os.path.exists("./save_tensors"):
            os.makedirs("./save_tensors")
        args.FP16 = True
        args.model = "llama2-7B"
        args.attn_implementation = "eager"
        args.PPL_seq_length= 4096
    if args.generate_for_calibration:
        if not os.path.exists("./save_tensors/calibration_tensors"):
            os.makedirs("./save_tensors/calibration_tensors")
        args.attn_implementation = "eager"
        args.Grouped_Head_Key_Rotation = True
        args.head_group_num = 4
        args.PPL_seq_length= 4096
    if args.RotateKV2 or args.RotateKV3 or args.RotateKV4:
        args.attn_implementation = "flash_attention_2"
        args.head_group_num = 4
        args.fuse_weights = True        
        args.Grouped_Head_Key_Rotation = True
        args.Outlier_Aware_Key_Rotation = True
        args.Attention_Sink_Aware = True    
        if args.RotateKV2:
            args.k_bits = 2
            args.v_bits = 2 
            args.k_clip_ratio = 0.8
            args.v_clip_ratio = 0.8
        elif args.RotateKV3:
            args.k_bits = 3
            args.v_bits = 3
            args.k_clip_ratio = 1
            args.v_clip_ratio = 1
        else:
            args.k_bits = 4
            args.v_bits = 4
            args.k_clip_ratio = 1
            args.v_clip_ratio = 1
    if args.FP16:
        args.attn_implementation = "eager"
    if args.RTN2:
        args.attn_implementation = "eager"
        args.k_bits = 2
        args.v_bits = 2       
    if args.RTN3:
        args.attn_implementation = "eager"
        args.k_bits = 3
        args.v_bits = 3
    if args.RTN4:
        args.attn_implementation = "eager"
        args.k_bits = 4
        args.v_bits = 4

        




    return args

