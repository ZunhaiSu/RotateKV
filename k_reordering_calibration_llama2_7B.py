import torch
import os

for layer_num in range(32):
    channel_sum_all = torch.empty((0, 4096)).cuda()
    for iter_num in range(83):
        key_states = torch.load(f'/root/RotateKV/save_tensors/calibration_tensor/sample_num{sample_num}_layer_idx{layer_num}.pt').cuda()
        (bsz, num_of_heads, seq_length, head_dim) = key_states.shape
        key_states = key_states.transpose(1, 2).reshape(bsz*seq_length, num_of_heads* head_dim)
        channel_sum = torch.sum(key_states, dim=0)
        channel_sum_all = torch.cat((channel_sum_all,channel_sum.unsqueeze(0)), dim = 0)
        print(channel_sum_all.shape)
    channel_sum_all_sum= torch.sum(channel_sum_all, dim=0)
    _, indices = torch.sort(channel_sum_all_sum)
    print(f"layer_num_{layer_num}")    
    print(indices.shape)#(32, 128)
    
    torch.save(indices, f"/root/RotateKV/reordering_indices_llama2_7b/layer_num_{layer_num}.pt")