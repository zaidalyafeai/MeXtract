# MeXtract

MeXtract is a family of light-weight models fintuned using Qwen2.5. This repository provides the scripts to finetune and evaluate these models.

## Installation

Install uv from https://github.com/astral-sh/uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and install dependencies:

```bash
uv sync
```

Create a `.env` file in the project root with your API keys:

```bash
OPENROUTER_API_KEY=sk-***
# Add other API keys as needed
```
## Finetuning 

### Supervised Fine-Tuning (SFT)

```bash
uv run src/finetune_hf.py \
    --model_name_or_path base_model_path \
    --output_model_name sft_model_path \
    --max_model_len 8192 \
    --max_output_len 2048
```

### Direct Preference Optimization (DPO)

```bash
uv run src/finetune_preference.py \
    --dataset_name preference_data_path \
    --model_name_or_path sft_model_path  \
    --learning_rate 5.0e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --max_steps 300 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --eval_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 100 \
    --output_dir dpo_adapters_path \
    --no_remove_unused_columns \
    --use_peft \
    --lora_r 8 \
    --lora_alpha 16
```

## Evaluation

```bash
    uv run src/evaluate.py \
        --split test \
        --backend vllm \
        --model dpo_model_path \
        --schema_name model \
        --max_model_len 8192 \
        --max_output_len 2048 \
        --overwrite
```

## Visualization 

```bash
uv run src/plots.py --split test --schema_name all --group_by_x category
```

## Download Models 

All the 3 models are available through Gogle cloud in this link 

https://storage.googleapis.com/mextract-models


## Inference Example 

```python
from schema import TextSchema
from type_classes import *
from search import extract

class ExampleSchema(TextSchema):
    Name: Field(Str, 1, 5)
    Hobbies: Field(List[Str], 1, 1, ['Hiking', 'Swimming', 'Reading'])
    Age : Field(Int, 1, 100)
    Married: Field(Bool, 1, 1)

text = """
My name is Zaid. I am 25 years old. I like swimming and reading. I am married. 
"""
metadata = extract(
    text, "IVUL-KAUST/MeXtract-3B", schema=ExampleSchema, backend = "transformers"   
)
print(metadata)

## {'Name': 'Zaid', 'Hobbies': ['Swimming'], 'Age': 25, 'Married': True}
```
## Citation
```
@misc{mextract,
      title={MeXtract: Light-Weight Metadata Extraction from Scientific Papers}, 
      author={Zaid Alyafeai and Maged S. Al-Shaibani and Bernard Ghanem},
      year={2025},
      eprint={2510.06889},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.06889}, 
}
```
