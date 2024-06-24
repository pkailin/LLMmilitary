# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run the KTO training script with the commands below. In general, the optimal configuration for KTO will be similar to that of DPO.

# Full training:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step

# QLoRA:
python examples/scripts/kto.py \
    --model_name_or_path=trl-lib/qwen1.5-1.8b-sft \
    --per_device_train_batch_size 8 \
    --num_train_epochs 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type=cosine \
    --gradient_accumulation_steps 1 \
    --logging_steps 10 \
    --eval_steps 500 \
    --output_dir=kto-aligned-model-lora \
    --warmup_ratio 0.1 \
    --report_to wandb \
    --bf16 \
    --logging_first_step \
    --use_peft \
    --load_in_4bit \
    --lora_target_modules=all-linear \
    --lora_r=16 \
    --lora_alpha=16
"""

from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from trl import KTOConfig, KTOTrainer, ModelConfig, get_peft_config, setup_chat_format
from datasets import Dataset
import torch
import pandas as pd 

torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = HfArgumentParser((KTOConfig, ModelConfig))
    kto_args, model_args = parser.parse_args_into_dataclasses()

    # Load a pretrained model
    model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", torch_dtype=torch.bfloat16).to("cuda")
    print("Model Initialisation Complete!")

    model_ref = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-7b-it", torch_dtype=torch.bfloat16).to("cuda")
    print("Reference Model Initialisation Complete!")

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
    print("Tokenizer Initialisation Complete!")
    
    # Dataset Creation (Replace with Actual Data)
    import pandas as pd
    df = pd.read_csv('missionData_llm.csv')

    test_data_dict = {
    'prompt': df['prompt'].tolist()[0:4] + df['prompt'].tolist()[94:],
    'completion': df['completion'].tolist()[0:4] + df['completion'].tolist()[94:],
    'label': df['label'].tolist()[0:4] + df['label'].tolist()[94:]
}

    train_data_dict = {
        'prompt': df['prompt'].tolist()[4:94], 
        'completion': df['completion'].tolist()[4:94], 
        'label': df['label'].tolist()[4:94]
    }

    from datasets import Dataset
    train_dataset = Dataset.from_dict(train_data_dict)
    test_dataset = Dataset.from_dict(test_data_dict)

    dataset_dict = {
        'train': train_dataset,
        'test': test_dataset,
    }

    print("Dataset Creation Complete!")

    # Initialize the KTO trainer
    kto_trainer = KTOTrainer(
        model,
        model_ref,
        args=kto_args,
        train_dataset=dataset_dict["train"],
        eval_dataset=dataset_dict["test"],
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    kto_trainer.train()
    print("Model Trained!")
    kto_trainer.save_model(kto_args.output_dir)
    print("Model Saved!")
    #kto_trainer.push_to_hub()
