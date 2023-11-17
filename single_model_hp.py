import itertools
import torch
import pandas as pd
from devinterp.slt import (
    estimate_learning_coeff_with_summary,
    plot_learning_coeff_trace,
)
from tqdm import tqdm
from devinterp.optim import SGLD, SGNHT

from utils import get_dataset, get_model, criterion_llm, pd_model_grouper, estimate_llc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_NAME = "gpt2"
DATASET_NAME = "imdb"
MAX_TOKEN_LENGTH = 200
BATCH_SIZE = 5
SAMPLING_METHOD = "SGNHT"
config = {
    "sampling_method": SAMPLING_METHOD,
    "num_chains": 7,
    "num_draws": 200,
    "num_burnin_steps": 20,
    "num_steps_bw_draws": 1,
    "clip_chains_quantile": 0.7,
}

hp_dict = {
    "SGLD": {
        "lr": [1e-7, 1e-5, 1e-3, 1e-1],
        "weight_decay": [0, 0.01, 0.1],
        "elasticity": [0, 0.1, 1],
        "bounding_box_size": [None, 0.5, 1],
    },
    "SGNHT": {
        "lr": [1e-7, 1e-5, 1e-3, 1e-1],
        "diffusion_factor": [0.001, 0.01, 0.1],
        "bounding_box_size": [None, 0.5, 1],
    },
}

tokenizer, model = get_model(MODEL_NAME)
dataset = get_dataset(
    DATASET_NAME,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_length=MAX_TOKEN_LENGTH,
)

for sampling, hps in hp_dict.items():
    optimizer_hps = [
        dict(zip(hps.keys(), values)) for values in itertools.product(*hps.values())
    ]
    print("Sampling method:", sampling)
    config["sampling_method"] = sampling
    results = []
    for optimizer_hp in tqdm(optimizer_hps):
        print("Config:", optimizer_hp)
        trace, learning_coeff_stats = estimate_llc(
            model,
            dataset,
            config,
            optimizer_kwargs=optimizer_hp,
            device=DEVICE,
        )
        optimizer_hp["mean"] = learning_coeff_stats["mean"]
        optimizer_hp["std"] = learning_coeff_stats["std"]
        results.append(optimizer_hp)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"hp_results_{sampling}.csv")
