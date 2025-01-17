import torch
import tqdm
from fast_hadamard_transform import hadamard_transform
import math
import utils

args = utils.parser_gen()

def apply_exact_had_to_linear(module, had_dim=-1, output=False, R=None):
    in_features, out_features = module.in_features, module.out_features

    W_ = module.weight.data
    dtype = W_.dtype
    dev = W_.device
    init_shape = W_.shape
    W_ = W_.float().cuda()
    hadK = R.to(torch.float64)
    if output:
        W_ = W_.t()
        transposed_shape = W_.shape
        temp = W_.reshape(-1, transposed_shape[-1] // had_dim, had_dim)
        temp = temp.to(torch.float64) @ hadK
        W_ = temp.reshape(transposed_shape).t()    
    else:
        init_shape = W_.shape
        temp = W_.reshape(-1, init_shape[-1] // had_dim, had_dim)
        temp = temp.to(torch.float64) @ hadK
        W_ = temp.reshape(init_shape)
    module.weight.data = W_.to(device=dev, dtype=dtype)
    
def rotate_ov_proj(layer, had_dim, R=None):
    v_proj = layer.self_attn.v_proj
    o_proj = layer.self_attn.o_proj

    apply_exact_had_to_linear(v_proj, had_dim=had_dim, output=True, R=R)
    apply_exact_had_to_linear(o_proj, had_dim=had_dim, output=False, R=R)

def rotate_reorder_k_proj(layer, had_dim, R=None, indices=None):
    k_proj = layer.self_attn.k_proj
    apply_exact_had_to_linear(k_proj, had_dim=had_dim, output=True, R=R)
    
    W_ = k_proj.weight.data
    dtype = W_.dtype
    dev = W_.device
    W_ = W_.float().cuda()
    W_ = W_.t()
    indices_repeat = indices.repeat([W_.shape[0],1])
    temp = torch.gather(W_, -1, indices_repeat)
    W_ = temp.t()    
    k_proj.weight.data = W_.to(device=dev, dtype=dtype)
    
@torch.inference_mode()
def fuse_weights(model):
    layers = [layer for layer in model.model.layers]
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads
    for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Fusing Weights")):
        WalshH = hadamard_transform(torch.eye(head_dim).cuda().float(), scale=1/math.sqrt(head_dim))
        WalshH_G = hadamard_transform(torch.eye(head_dim*args.head_group_num).cuda().float(), scale=1/math.sqrt(head_dim*args.head_group_num))
        indices_k = torch.load(f"./reordering_indices/reordering_indices_{args.model}.pt")[idx,:].cuda()
        rotate_ov_proj(layers[idx], head_dim, R=WalshH)
        rotate_reorder_k_proj(layers[idx], head_dim*args.head_group_num, R=WalshH_G, indices=indices_k)
    torch.cuda.empty_cache()

