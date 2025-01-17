# RotateKV: Accurate and Robust 2-Bit KV Cache Quantization for LLMs via Outlier-Aware Adaptive Rotations
This repository contains the code used to reproduce the simulation results of RotateKV.


## Abstract
In this work, we explore the potential of rotation techniques for 2-bit KV quantization and propose RotateKV, which achieves accurate and robust performance with high compression ratio by incorporating several novel improvements:  
(i) Outlier-Aware Rotation, which utilizes channel-reordering to adapt the rotations to varying channel-wise outlier distributions without sacrificing the computational efficiency of the Fast Walsh-Hadamard transform (FWHT);  
(ii) Pre-RoPE Grouped-Head Rotation, which mitigates the impact of rotary position embedding (RoPE) on rotation and further smooths outliers across heads;  
(iii) Attention-Sink-Aware Quantization, which leverages the massive activations to efficiently protect attention sinks.  
RotateKV achieves less than 0.3 perplexity (PPL) degradation with 2-bit quantization on WikiText-2 using LLaMA-2-13B, maintains strong CoT reasoning and long-context capabilities, with less than 1.7\% degradation on GSM8K, outperforming existing methods even at lower average bit-widths.  
<img src="figure/main.png" alt="main" width="800"/>

## Installation
```bash
conda create -n RotateKV python==3.10 -y
conda activate RotateKV

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

git clone https://github.com/ZunhaiSu/RotateKV.git
cd RotateKV
pip install -r requirements.txt

# Install the fast-hadamard-transform
git clone https://github.com/Dao-AILab/fast-hadamard-transform.git
cd fast-hadamard-transform
pip install -e.
```
## Evaluation
Simulation results of RotateKV.
### Calibration for the reordering indices
```bash
python PPL_evaluation.py --generate_for_calibration True
python k_reordering_calibration.py --model llama2_7b
```
### PPL Evaluation
```bash
# For FP16 baseline.
python PPL_evaluation.py --FP16 True
# For RotateKV results
# INT2
python PPL_evaluation.py --RotateKV2 True
# INT3
python PPL_evaluation.py --RotateKV3 True
# INT4
python PPL_evaluation.py --RotateKV4 True
```
### GSM8K Evaluation
```bash
# For FP16 baseline.
python gsm8k_evaluation.py --FP16 True
# For RotateKV results
# INT2
python gsm8k_evaluation.py --RotateKV2 True
# INT3
python gsm8k_evaluation.py --RotateKV3 True
# INT4
python gsm8k_evaluation.py --RotateKV4 True
```
### LongBench Evaluation  
Please load the model with RotateKV and refer to the [LongBench](https://github.com/THUDM/LongBench.git) for conducting the LongBench evaluations.

### MileBench Evaluation  
Please load the model with RotateKV and refer to the [MileBench](https://github.com/MileBench/MileBench.git) for conducting the MileBench evaluations.

### Needle-in-a-Haystack Evaluation  
Please load the model with RotateKV and refer to the [Long-Context-Data-Engineering](https://github.com/FranxYao/Long-Context-Data-Engineering.git) for conducting the NIAH evaluations.

## Visualization
### 2D Visualizations of Keys
```bash
# generate the key_states
# pre_RoPE
python PPL_evaluation.py --save_k_pre_rope True
# post_RoPE
python PPL_evaluation.py --save_k_post_rope True
```
Use `RotateKV/visualize/2D Visualizations of Key Tensors.ipynb` to generate the following 2D visualizations of Keys.  
<img src="figure/2D_key.png" width="400"/>
### 3D Visualizations of Keys
```bash
# generate the key_states
# pre_RoPE
python PPL_evaluation.py --save_k_pre_rope True
# post_RoPE
python PPL_evaluation.py --save_k_post_rope True

```
Use `RotateKV/visualize/3D Visualizations of Key Tensors.ipynb` to generate the following 3D visualizations of Keys.  
<img src="figure/Llama-2-7B Layer 10 Head 1 Key States.png" width="200"/><img src="figure/Llama-2-7B Layer 10 Head 2 Key States.png" width="200"/><img src="figure/Llama-2-7B Layer 10 Head 30 Key States.png" width="200"/><img src="figure/Llama-2-7B Layer 10 Head 31 Key States.png" width="200"/>
### Massive Activations
```bash
# generate the massive activations
python PPL_evaluation.py --save_massive_activations True
```
Use `RotateKV/visualize/Massive Activations.ipynb` to generate the following visualizations of massive activations.  
<img src="figure/massive_activation.png" width="200"/>
### Attention Sinks
```bash
# generate the attetnion scores
python PPL_evaluation.py --save_attention_scores True

```
Use `RotateKV/visualize/Attention Sinks.ipynb` to generate the following visualizations of attention sinks.  
<img src="figure/attn_weights_layer_11_head0.png" width="200"/><img src="figure/attn_weights_layer_11_head1.png" width="200"/><img src="figure/attn_weights_layer_11_head2.png" width="200"/><img src="figure/attn_weights_layer_11_head3.png" width="200"/>
## Citation
