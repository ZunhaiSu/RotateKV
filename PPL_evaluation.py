import utils
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
from RotateKV_fake_quant.modeling_llama_RotateKV import LlamaForCausalLM_RotateKV
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")
import tqdm
import torch.nn as nn
from fuse_weights import fuse_weights

def ppl_evaluator(model, testenc, sequence_length, generate_for_calibration):
    tokenized_dataset = testenc.input_ids.to("cuda")
    num_of_iterations = tokenized_dataset.shape[1] // sequence_length
    nlls = []
    with torch.no_grad():
        for i in tqdm.tqdm(range(num_of_iterations), desc="Evaluating..."):
                batch = tokenized_dataset[:, (i * sequence_length) : ((i + 1) * sequence_length)].to(model.device)
                outputs = model(batch)
                lm_logits = outputs.logits
                shift_logits = lm_logits[:, :-1, :].contiguous().float()
                shift_labels = tokenized_dataset[:, (i * sequence_length) : ((i + 1) * sequence_length)][:, 1:]
                loss_fc = nn.CrossEntropyLoss()
                loss = loss_fc(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                neg_log_likelihood = loss.float() * sequence_length
                nlls.append(neg_log_likelihood)
        
        ppl = torch.exp(torch.stack(nlls).sum() / (num_of_iterations * sequence_length))
    if generate_for_calibration:
        print("generate for calibration successfully!")
    else:
        print(f"WikiText-2 perplexity: {ppl}")
    return ppl.item()


def main():
    args = utils.parser_gen()
    model_paths = {
        "llama2_7b": "/root/autodl-tmp/Llama-2-7b-hf",
        "llama2_7b_80K": "your_path_for_llama2_7b_80K",
        "llama2_13b": "your_path_for_llama2_13b",
        "llama3_8b": "your_path_for_llama3_8b",
        "mistral_7b": "your_path_for_mistral_7b"
    }
    model_path = model_paths.get(args.model, "default_path")
    model = LlamaForCausalLM_RotateKV.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", attn_implementation=args.attn_implementation, use_safetensors=args.use_safetensors)
    if args.fuse_weights:
        fuse_weights(model)
    model.eval()      
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split='test') 
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    dataset_ppl = ppl_evaluator(model, testenc, args.PPL_seq_length, args.generate_for_calibration)

if __name__ == '__main__':
    main()
