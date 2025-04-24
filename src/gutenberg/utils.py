import numpy as np
import json
import matplotlib.pyplot as plt
import ast
import tiktoken
import torch
from torch.utils.data import TensorDataset
from copy import deepcopy

enc = tiktoken.get_encoding("gpt2")
EOS_TOKEN = enc.eot_token

class Library:

    def __init__(self, data, auxdata, selected_subject="English fiction", block_size=1024):

        lengths = np.array([row['len'] for row in auxdata])
        margins = [0] + np.cumsum(lengths).tolist()
        metadata = np.array([ast.literal_eval(row['METADATA']) for row in auxdata])
        subjects = [row['subjects'] for row in metadata]

        # Get the indices of the books with the given subject
        indices = []
        for idx, subject in enumerate(subjects):
            if selected_subject in subject:
                indices.append(idx)
            else:
                continue
        
        # get the tokens and the margins of the selected books
        data_ = []
        for idx in indices:
            start = margins[idx]
            end = margins[idx + 1] if idx + 1 < len(margins) else len(data)
            data_.append(data[start:end])

        data_ = np.concatenate(data_)
        lengths_ = lengths[indices]
        margins_ = [0] + np.cumsum(lengths_).tolist()
        num_books = len(margins_) - 1

        self.num_books = num_books
        self.block_size = block_size
        self.metadata = metadata[indices]
        self.margins_ = margins_
        self.data_ = data_
        self.lengths_ = lengths_
        
        self.start = 0
        self.end = num_books - 1
        self.current = 0

    def preprocess(self, data):
        num_tokens = len(data)
        block_size = self.block_size
        num_blocks = (num_tokens + block_size - 2) // block_size

        # form source and target sequences
        X = []
        for j in range(num_blocks):
            start = j * block_size
            end = min((j+1) * block_size, num_tokens-1)
            X.append(
                torch.from_numpy((data[start:end]).astype(np.int64))
            )

        Y = deepcopy(X)

        # pad the last sequence
        X[-1] = torch.nn.functional.pad(X[-1], pad=(0, block_size - len(X[-1])), value=EOS_TOKEN)
        Y[-1] = torch.nn.functional.pad(Y[-1], pad=(0, block_size - len(Y[-1])), value=-100)

        return TensorDataset(torch.stack(X), torch.stack(Y))
    
    def __len__(self):
        return self.num_books

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration

        # Get the tokens for the current book
        start = self.margins_[self.current]
        end = self.margins_[self.current + 1]
        book_tokens = self.data_[start:end]

        # Get remaining tokens for evaluation
        if self.current + 1 < self.num_books:
            eval_start = self.margins_[self.current + 1]
            eval_end = self.margins_[self.current + 2] if self.current + 2 < len(self.margins_) else len(self.data_)
            eval_tokens = self.data_[eval_start:eval_end]
        else:
            eval_tokens = np.array([], dtype=self.data_.dtype)  # Empty array if no more books

        # Preprocess the data
        train_data = self.preprocess(book_tokens)
        eval_data = self.preprocess(eval_tokens)

        metadata = self.metadata[self.current]
        metadata['length'] = self.lengths_[self.current]

        # Increment the current book
        self.current += 1

        return dict(train_data=train_data, eval_data=eval_data, metadata=metadata)

    def reset(self):
        self.current = self.start
    
    
if __name__ == "__main__":

    batch_size = 8
    data_file = 'gutenberg/books.bin'
    auxdata_file = 'gutenberg/books_metadata.json'

    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    auxdata = []
    with open(auxdata_file, 'r') as file:
        for line in file:
            auxdata.append(json.loads(line))

    library = Library(data, auxdata)

    book = next(library)

    X, Y = book['data']    

    print(book['metadata'])
    print(book['tokens'][:10])

    # Form indices of batches
    num_blocks = len(X)
    block_indices = np.arange(num_blocks)
    batch_indices = [block_indices[i:min(i + batch_size, num_blocks)] for i in range(0, num_blocks, batch_size)]

    for ids in batch_indices:
        X_batch = torch.stack([X[i] for i in ids])
        Y_batch = torch.stack([Y[i] for i in ids])
        print(X_batch.shape)
        print(Y_batch.shape)
        break

