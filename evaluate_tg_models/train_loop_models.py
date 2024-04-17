import subprocess
import itertools
from tqdm import tqdm, trange

## first round
# hidden_dims = [128, 512]
# dropout_rate = [0, 0.05, 0.1]
# phase_noise = [0, 0.25, 0.5]

## second round 
# hidden_dims = [16, 32, 64, 128, 256]
# dropout_rate = [0, 0.05, 0.1, 0.2]
# phase_noise = [0, 0.1, 0.2, 0.3, 0.4]

## third round
# hidden_dims = [16, 32, 64]
# dropout_rate = [0, 0.05, 0.1]
# phase_noise = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
# above n_epochs = 100 implicitly

## fourth round
# hidden_dims = [32, 64, 128]
# dropout_rate = [0, 0.05]
# phase_noise = [0, 0.1, 0.2, 0.3, 0.4]
# n_epochs = [100, 900]

## fifth round
hidden_dims = [32, 48, 64]
dropout_rate = [0, 0.05, 0.10]
phase_noise = [0, 0.1, 0.2, 0.3, 0.4]
n_epochs = [100, 900]

products = list(itertools.product(hidden_dims, dropout_rate, phase_noise, n_epochs))

for (hidden_dim, dropout, noise, n_ep) in tqdm(products, ncols=80):
    command = [
        'python',
        'train_tg_model.py',
        '--hidden_dim',
        str(hidden_dim),
        '--dropout_rate',
        str(dropout),
        '--phase_noise',
        str(noise),
        '--n_pred_btt',
        '0',
        '--n_epochs',
        str(n_ep)
    ]

    subprocess.run(command, check=True)
