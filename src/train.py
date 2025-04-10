import os
from transformers import GPT2LMHeadModel, AutoConfig
import torch
from torch.utils.data import DataLoader
import numpy as np
import json
from tqdm import tqdm

from utils import Library

os.path.join(os.path.dirname(__file__), 'gutenberg')

data_file = os.path.join(os.path.dirname(__file__), 'gutenberg/books.bin')
auxdata_file = os.path.join(os.path.dirname(__file__), 'gutenberg/books_metadata.json')

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

config = AutoConfig.from_pretrained("gpt2")
model = GPT2LMHeadModel(config)
model.to(1)

library = Library(data, auxdata)

library.reset()

model.to(1)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

training_loss = []
perplexity = []

progress_bar = tqdm(enumerate(library), total=len(library), unit='book')
ppl = float('inf')

for book_id, book in progress_bar:

    train_data = book['train_data']    
    train_loader = DataLoader(train_data, batch_size=8, shuffle=False)
    
    model.train()

    for k, batch in enumerate(train_loader):

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
        progress_bar.set_description(f"step [{k + 1}/{len(train_loader)}] - tr_loss: {tr_loss:.4f}, ppl: {ppl:.4f}")

    eval_data = book['eval_data']
    eval_loader = DataLoader(eval_data, batch_size=16, shuffle=False)
    ppl = evaluate(model, eval_loader)
    perplexity.append(ppl)
    progress_bar.set_description(f"step [{k + 1}/{len(train_loader)}] - tr_loss: {tr_loss:.4f}, ppl: {ppl:.4f}")

    # save results
    results = dict(tr_loss=training_loss, perplexity=perplexity)
    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)