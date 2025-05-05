import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import namedtuple
from torch.nn import RNN
import json

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_sequence(c=0.005, f_0=0, sf=10):
    """
    Generate a linear chirp. x is the time-domain signal and f is the instantaneous frequency
    """
    t = np.arange(0, 50, 1/sf)
    f = c * t + f_0
    phi = 2 * np.pi * (c / 2 * t**2 + f_0 * t)
    x = np.sin(phi)
    return x, f, t


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNCell, self).__init__()
        self.hidden_size = hidden_size

        self.rnn = RNN(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.rnn(x)
        output = self.h2o(h)
        return output
    
def train_rnn(rnn, data, t, t_pretrain, hp, ignore_pretrained_data=False):
    # train the RNN on the first t steps of the sequence
    X, Y = data
    input_seq, output_seq = X[:t], Y[:t]
    len_seq = len(input_seq)

    x = torch.from_numpy(input_seq).view(1, -1, 1).float()
    y = torch.from_numpy(output_seq).float()

    rnn.train()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=hp.lr)
    criterion = nn.MSELoss(reduction='none')
    pbar = tqdm(range(hp.num_epochs))

    for epoch in pbar:    
        # iterate over the sequence
        out = rnn(x)
        
        # compute loss
        unmasked_losses = criterion(out.squeeze(), y)
        if ignore_pretrained_data:
            mask = torch.zeros_like(unmasked_losses, dtype=torch.bool)
            mask[t_pretrain:] = True
            masked_losses = unmasked_losses * mask.float()
            loss = masked_losses.mean()
        else:
            loss = unmasked_losses.mean()

        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Epoch {epoch+1}/{hp.num_epochs}, Loss: {loss.item()/len_seq:.4f}")
    return rnn

def evaluate(rnn, data, t):
    # get data (only evaluate on the future, beyond t)
    X, Y = data
    input_seq, output_seq = X, Y

    x = torch.from_numpy(input_seq).view(1, -1, 1).float()
    y = torch.from_numpy(output_seq).float()

    rnn.eval()
    with torch.no_grad():
        out = rnn(x)
        unmasked_losses = (out.squeeze() - y).abs().detach().numpy()
    return np.mean(unmasked_losses[t:])

def pretrain(data, t, hp, sample_rate, seed=1996):
    """
    pretrain the RNN on the first t steps of the sequence
    """
    t = int(t * sample_rate)

    # initialize
    set_seed(seed)
    rnn = RNNCell(1, hp.hidden_size, 1)

    if t > 0:
        rnn = train_rnn(rnn, data, t, t, hp, ignore_pretrained_data=False)
    error = evaluate(rnn, data, t)
    return rnn.state_dict(), error

def run_at_time(data, t, t_pretrain, pretrained_state_dict, hp, sample_rate, ignore_pretrained_data=False, finetune=True, seed=1996):
    """
    We train an RNN to predict the instantaneous frequency in the future, given the time-domain value and the previous state.

    The RNN pretrained on the first `t_pretrain` steps of the sequence.

    When train == True: 
        The RNN is trained on the first t (including the pretraining data) steps of the sequence and evaluated on the future steps

    When train == False
        The RNN is not trained at all (no gradient steps)    
    """
    t = int(t * sample_rate)
    t_pretrain = int(t_pretrain * sample_rate)

    # initialize
    set_seed(seed)
    rnn = RNNCell(1, hp.hidden_size, 1)
    rnn.load_state_dict(pretrained_state_dict)

    # train
    if finetune:
        rnn = train_rnn(rnn, data, t, t_pretrain, hp, ignore_pretrained_data=ignore_pretrained_data)

    # evaluate
    error = evaluate(rnn, data, t)
    
    return error

def run_on_seed(
        data, 
        t_pretrain, 
        t_list, 
        hp_pretrain, 
        hp_finetune, 
        sample_rate, 
        ignore_pretrained_data=False,
        seed=1996
    ):
    
    err_ft_list = []
    err_noft_list = []

    pretrained_state_dict, initial_error = pretrain(data, t_pretrain, hp_pretrain, sample_rate, seed=seed)

    for t in t_list:
        print(f"t = {t}")
        err = run_at_time(data, t, t_pretrain, pretrained_state_dict, hp_finetune, sample_rate, ignore_pretrained_data=ignore_pretrained_data, finetune=True, seed=seed)
        err_ft_list.append(err)

        err = run_at_time(data, t, t_pretrain, pretrained_state_dict, hp_finetune, sample_rate, ignore_pretrained_data=ignore_pretrained_data, finetune=False, seed=seed)
        err_noft_list.append(err)

    return err_ft_list, err_noft_list, initial_error


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run the RNN on a toy dataset')
    parser.add_argument('--hidden_size', type=int, default=8, help='Hidden size of the RNN')
    parser.add_argument('--ignore_pretrained_data', action='store_true', help='Ignore the pretrained data')
    parser.add_argument('--num_pt_epochs', type=int, default=100, help='Number of epochs for pretraining')
    parser.add_argument('--num_ft_epochs', type=int, default=100, help='Number of epochs for finetuning')
    parser.add_argument('--variable_seeds', action='store_true', help='use variable seeds')

    args = parser.parse_args()

    variable_seeds = args.variable_seeds
    sample_rate = 10
    hidden_size = args.hidden_size
    ignore_pretrained_data = args.ignore_pretrained_data
    num_reps = 3

    lr_pretrain = 0.001
    num_epochs_pretrain = args.num_pt_epochs

    lr_finetune = 0.001
    num_epochs_finetune = args.num_ft_epochs

    t_pretrain = 10
    t_list = np.arange(10+1, 25, 1)

    Hyperparameters = namedtuple('Hyperparameters', ['hidden_size', 'lr', 'num_epochs'])
    hp_pretrain = Hyperparameters(hidden_size=hidden_size, lr=lr_pretrain, num_epochs=num_epochs_pretrain)
    hp_finetune = Hyperparameters(hidden_size=hidden_size, lr=lr_finetune, num_epochs=num_epochs_finetune)

    X, Y, _ = generate_sequence(sf=sample_rate)

    err_ft = []
    err_noft = []
    err_init = []

    for rep in range(num_reps):
        if variable_seeds:
            seed = 1000 * rep + 1996
        else:
            seed = 1996

        errs = run_on_seed(
            (X, Y), 
            t_pretrain, 
            t_list, 
            hp_pretrain, 
            hp_finetune, 
            sample_rate, 
            ignore_pretrained_data=ignore_pretrained_data,
            seed=seed
        )
        err_ft.append(errs[0])
        err_noft.append(errs[1])
        err_init.append(errs[2])

    err_ft_mean = np.mean(err_ft, axis=0)
    err_noft_mean = np.mean(err_noft, axis=0)
    err_init_mean = np.mean(err_init, axis=0)

    err_ft_std = np.std(err_ft, axis=0)
    err_noft_std = np.std(err_noft, axis=0)
    err_init_std = np.std(err_init, axis=0)

    # Calculate 95% confidence interval for error bars
    z_score = 1.96
    err_ft_margin = z_score * err_ft_std / np.sqrt(len(err_ft))
    err_noft_margin = z_score * err_noft_std / np.sqrt(len(err_noft))

    results = {
    'err_ft_mean': err_ft_mean.tolist(),
    'err_noft_mean': err_noft_mean.tolist(),
    'err_init_mean': err_init_mean.tolist(),
    'err_ft_std': err_ft_std.tolist(),
    'err_noft_std': err_noft_std.tolist(),
    'err_init_std': err_init_std.tolist(), 
    't_list': t_list.tolist()
    }


    exp_name = f'h{hidden_size}'
    if ignore_pretrained_data:
        exp_name += '_ignore_ptd'
    if variable_seeds:
        exp_name += '_var_seeds'

    filename = f'results/{exp_name}.json'
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

    