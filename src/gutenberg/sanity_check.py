from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig, AutoModelForCausalLM
import tiktoken
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

from gutenberg.utils import Library

batch_size = 8
num_epochs = 5

data_file = 'gutenberg/books.bin'
auxdata_file = 'gutenberg/books_metadata.json'

data = np.memmap(data_file, dtype=np.uint16, mode='r')
auxdata = []
with open(auxdata_file, 'r') as file:
    for line in file:
        auxdata.append(json.loads(line))

@torch.no_grad()
def evaluate(model, eval_loader):
    model.eval()
    total_log_likelihood = 0.0
    total_tokens = 0

    for batch in eval_loader:
        X, Y = batch
        inputs = {
            'input_ids': X.to(1),
            'labels': Y.to(1)
        }

        outputs = model(**inputs)
        loss = outputs.loss

        log_likelihood = -loss.item() * X.numel()
        total_log_likelihood += log_likelihood
        total_tokens += X.numel()

    avg_neg_log_likelihood = -total_log_likelihood / total_tokens

    try:
        perplexity = torch.exp(torch.tensor(avg_neg_log_likelihood))
    except OverflowError:
        perplexity = float('inf')
    return perplexity.item()

# TODO: check the tokenizer and config stuff
# config = AutoConfig.from_pretrained("gpt2")
# model = GPT2LMHeadModel(config)
model = AutoModelForCausalLM.from_pretrained("gpt2")
model.to(1)

library = Library(data, auxdata)

# select the book with the most tokens
id = np.argmax(library.lengths_)
library.current = id
book = next(library)

data = book['train_data']    
dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

model.to(1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

training_loss = []
perplexity = []

for epoch in range(num_epochs):

    print(f"epoch : {epoch+1}")

    progress_bar = tqdm(enumerate(dataloader))

    for k, batch in progress_bar:
        model.train()

        x, y = batch
        inputs = {
            'input_ids': x.to(1),
            'labels': y.to(1)
        }

        outputs = model(**inputs)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tr_loss = loss.item()
        training_loss.append(tr_loss)

        # evaluate
        if k < len(dataloader)-2:
            remaining_data = torch.utils.data.Subset(data, range((k+1)*batch_size, len(data)))
        else:
            break

        eval_loader = DataLoader(remaining_data, batch_size=batch_size, shuffle=False)
        ppl = evaluate(model, eval_loader)
        perplexity.append(ppl)

        progress_bar.set_description(f"step [{k + 1}/{len(dataloader)}] - tr_loss: {tr_loss:.4f}, ppl: {ppl:.4f}")

    # save results
    results = dict(tr_loss=training_loss, perplexity=perplexity)
    with open('sc_pt_results.json', 'w') as file:
        json.dump(results, file, indent=4)