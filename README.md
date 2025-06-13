# LLMInit: Selective Initialization from Large Language Models for Recommendation

This is the PyTorch Implementation for LLMInit.

## Environment Setup

1. Create a new conda environment:
```bash
conda create -n llminit python=3.8
conda activate llminit
```

2. Install PyTorch (adjust CUDA version as needed):
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install RecBole and other dependencies:
```bash
pip install recbole
pip install transformers
pip install scikit-learn
```

## Data Processing

The data processing is automatically handled by RecBole framework. When you run the code with our provided configuration:

1. The dataset (e.g., Amazon-Beauty) will be automatically downloaded
2. Data preprocessing will be performed automatically
3. The processed data will be cached for future use

No manual data processing is required.

## Usage Examples

(1) run the **LLMInit-Rand** with the **LightGCN** on Amazon-Beauty
```bash
python run_recbole.py --opt rand -d amazon-beauty -m ContGCN
```

(2) run the **LLMInit-Uni** with the **SGL** on Amazon-Beauty
```bash
python run_recbole.py --opt uni -d amazon-beauty -m ContSGL
```

(3) run the **LLMInit-Var** with the **SGCL** on Amazon-Beauty
```bash
python run_recbole.py --opt var -d amazon-beauty -m ContSGCL
```

## Acknowledgement
The structure of this repo is built based on [RecBole](https://github.com/RUCAIBox/RecBole). Thanks for their great work.
