import torch

def find_params_per_token_groupwise(x, clip_ratio, bit, quant_gs):
    init_shape = x.shape
    reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // quant_gs, quant_gs)
    xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * clip_ratio
    xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * clip_ratio
    maxq = torch.tensor(bit**2 - 1)
    minq = 0
    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)
    scale = scale.repeat(1, 1, 1, quant_gs).reshape(init_shape)
    zero = zero.repeat(1, 1, 1, quant_gs).reshape(init_shape)
    return scale, zero


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def fake_quant(x, clip_ratio, bit, quant_gs):
    scale, zero = find_params_per_token_groupwise(x, clip_ratio, bit, quant_gs)
    qx, scale, zero = asym_quant(x, scale, zero, maxq=torch.tensor(2**bit - 1))
    return asym_dequant(qx, scale, zero)