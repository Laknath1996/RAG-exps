import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
import os
import re
import json


def divide_to_chapter(text):
    chapter_pattern = r'(?i)^chapter\s+[a-z]+'
    titles = re.findall(chapter_pattern, text, re.MULTILINE)
    splits = re.split(chapter_pattern, text, flags=re.MULTILINE)
    chapters = [f"{title}\n{body.strip()}" for title, body in zip(titles, splits[1:])]
    return chapters

def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def preprocess_text(text):
    text = re.sub(r"^(CHAPTER \w+\n)(.*\n\n)", "", text, flags=re.MULTILINE)
    text = text.strip()
    return text

class TextDataset(Dataset):
    def __init__(self, tokenizer, text, block_size):

        tokenized_text = tokenizer(text, return_tensors='pt')['input_ids']
        print(f"Tokenized text shape: {tokenized_text.shape}")
        self.examples = []
        for i in range(0, tokenized_text.size(1) - block_size + 1, block_size):
            self.examples.append(tokenized_text[0, i:i + block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA.")

    # dataset args
    parser.add_argument('--chapters_path', type=str, default='hp/hp1_2_chapters.json', help='Path to the book text file.')
    parser.add_argument("--current", type=int, default=3, help="Currently completed chapter.")
    parser.add_argument("--block_size", type=int, default=128, help="Size of text blocks for training.")

    # LORA args
    parser.add_argument("--lora_alpha", type=int, default=32, help="Alpha parameter for LoRA.")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout rate for LoRA.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank for LoRA.")

    # training args
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training.") 
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Base model name or path.")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Number of warmup steps for the scheduler.")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use for training (e.g., 'cuda:0', 'cpu').")

    # checkpoint args
    parser.add_argument("--save_at_each_epoch", action='store_true', help="Flag to save model checkpoint at each epoch.")

    # seed
    parser.add_argument("--seed", type=int, default=1996, help="Random seed for reproducibility.")

    args = parser.parse_args()

    model_name = args.model_name
    current = args.current
    lora_rank = args.lora_rank
    lora_alpha = args.lora_alpha
    lora_dropout = args.lora_dropout
    chapters_path = args.chapters_path
    block_size = args.block_size
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    warmup_steps = args.warmup_steps
    gradient_accumulation_steps = args.gradient_accumulation_steps
    seed = args.seed
    device = args.device
    save_at_each_epoch = args.save_at_each_epoch

    print(f"Finetuning on chapters up to {current}...")
    book_name = os.path.basename(chapters_path).split('.')[0].replace('_chapters', '')
    with open(chapters_path) as f:
        chapters = json.load(f)
    output_dir = f"lora_adapters/{book_name}/{book_name}_ch{current}"

    # Set seed for reproducibility
    torch.manual_seed(seed)
    import numpy as np
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare the text string for training
    train_chapters = chapters[:current+1]
    train_text = "\n\n".join(train_chapters)

    train_dataset = TextDataset(tokenizer, train_text, block_size)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # Model and LoRA Setup
    model = AutoModelForCausalLM.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # Training Loop
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in progress_bar:
            inputs = batch.to(device)
            labels = batch.to(device)

            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps # Accumulate gradients

            loss.backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})

        print(f"Epoch {epoch+1} Loss: {epoch_loss / len(train_dataloader)}")

        # save model checkpoint at each epoch
        if save_at_each_epoch:
            model.save_pretrained(os.path.join(output_dir, f"checkpoint-{epoch+1}"))


    # Save the adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"LoRA adapters saved to {output_dir}")