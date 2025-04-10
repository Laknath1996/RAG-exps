import os
from tqdm import tqdm
import numpy as np
import tiktoken
import ast

from datasets import load_dataset

if __name__ == '__main__':

    # subject_choice = "English fiction"

    dataset = load_dataset("sedthh/gutenberg_english")['train']
    enc = tiktoken.get_encoding("gpt2")

    # metadata = [ast.literal_eval(row['METADATA']) for row in dataset]
    # subjects = [row['subjects'] for row in metadata]

    # indices = []
    # for idx, subject in enumerate(subjects):
    #     if "English fiction" in subject:
    #         print(subject)
    #         count += 1
    #         indices.append(idx)
    #     else:
    #         continue

    # sub_dataset = dataset.select(indices)

    def process(example):
        ids = enc.encode_ordinary(example['TEXT']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        out = {'ids': ids, 'len': len(ids)}
        return out

    tokenized = dataset.map(
        process,
        remove_columns=['TEXT'],
        desc="Tokenizing",
        num_proc=8,
    )

    arr_len = np.sum(tokenized['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), 'books.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

    total_batches = 10

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
        # Batch together samples for faster write
        batch = tokenized.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
        arr_batch = np.concatenate(batch['ids'])
        # Write into mmap
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)
    arr.flush()

    metadata = tokenized.remove_columns(['ids', 'SOURCE'])
    metadata_filename = os.path.join(os.path.dirname(__file__), 'books_metadata.json')
    metadata.to_json(metadata_filename)

