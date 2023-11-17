import itertools
import torch
import pandas as pd
from transformers import BitsAndBytesConfig
from tqdm import tqdm
from devinterp.optim import SGLD, SGNHT

from utils import get_dataset, get_model, estimate_llc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_NAME = "meta-llama/Llama-2-13b-hf"

QUANTIZED = True
DATASET_NAME = "imdb"
MAX_TOKEN_LENGTH = 200
BATCH_SIZE = 1

config = {
    "sampling_method": None,
    "num_chains": 5,
    "num_draws": 100,
    "num_burnin_steps": 20,
    "num_steps_bw_draws": 1,
    "clip_chains_quantile": 0.9,
}

if QUANTIZED:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    kwargs = {"quantization_config": bnb_config}
else:
    kwargs = {}

hp_dict = {
    "SGLD": {
        "lr": [1e-6],
        "weight_decay": [0],
        "elasticity": [1.0, 2.0, 3.0],
        "bounding_box_size": [0.001, 0.0001],
    },
    "SGNHT": {
        "lr": [1e-7, 1e-5, 1e-3, 1e-1],
        "diffusion_factor": [0.001, 0.01, 0.1],
        "bounding_box_size": [None, 0.5, 1],
    },
}

tokenizer, model = get_model(MODEL_NAME, device=DEVICE, model_loading_kwargs=kwargs)
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
    break
