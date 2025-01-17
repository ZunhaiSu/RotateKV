import torch
import os
import utils

def main():
    args = utils.parser_gen()
    model_config = {
        "llama2_7b": {"total_layers": 32, "total_samples": 83, "hidden_size": 4096},
        "llama2_7b_80K": {"total_layers": 32, "total_samples": 83, "hidden_size": 4096},
        "llama2_13b": {"total_layers": 40, "total_samples": 83, "hidden_size": 5120},
        "llama3_8b": {"total_layers": 32, "total_samples": 70, "hidden_size": 1024},
        "mistral_7b": {"total_layers": 32, "total_samples": 81, "hidden_size": 1024},
    }
    config = model_config.get(args.model, {"total_layers": 32, "total_samples": 81, "hidden_size": 1024})
    total_layers = config["total_layers"]
    total_samples = config["total_samples"]
    hidden_size = config["hidden_size"]
    all_indices = []
    for layer_num in range(total_layers):
        channel_sum_all = torch.empty((0, hidden_size)).cuda()
        for iter_num in range(total_samples):
            key_states = torch.load(f'./save_tensors/calibration_tensor_{args.model}/sample_num{sample_num}_layer_idx{layer_num}.pt').cuda()
            (bsz, num_of_heads, seq_length, head_dim) = key_states.shape
            key_states = key_states.transpose(1, 2).reshape(bsz*seq_length, num_of_heads* head_dim)
            channel_sum = torch.sum(key_states, dim=0)
            channel_sum_all = torch.cat((channel_sum_all,channel_sum.unsqueeze(0)), dim = 0)
            print(channel_sum_all.shape)
        channel_sum_all_sum= torch.sum(channel_sum_all, dim=0)
        _, indices = torch.sort(channel_sum_all_sum)
        all_indices.append(indices)
        print(f"layer_num_{layer_num}")    
        # print(indices.shape)#(32, 128)
        
    all_indices_tensor = torch.stack(all_indices, dim=0)
    print(f"All indices shape: {all_indices_tensor.shape}")
    torch.save(all_indices_tensor, f"./reordering_indices/reordering_indices_{args.model}.pt")
if __name__ == '__main__':
    main()