import yaml
import torch
import pandas as pd
from tqdm import tqdm
from utils import get_dataset, get_model, pd_model_grouper, estimate_llc
from transformers import BitsAndBytesConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_FAMILY = "llama2"
# MODEL_LIST = ["gpt2-xl", "gpt2-large", "gpt2-medium", "gpt2"]
MODEL_LIST = ["meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-13b-hf"]
DATASET_NAME = "imdb"
MAX_TOKEN_LENGTH = 200
BATCH_SIZE = 1
SAMPLING_METHOD = "SGLD"
NUM_ITERATIONS = 3
QUANTIZED = True

config = {
    "sampling_method": SAMPLING_METHOD,
    "num_chains": 5,
    "num_draws": 180,
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

tokenizer, _ = get_model(MODEL_LIST[0], device=DEVICE, model_loading_kwargs=kwargs)
results = []
for i in tqdm(range(NUM_ITERATIONS)):
    dataset = get_dataset(
        DATASET_NAME,
        tokenizer,
        batch_size=BATCH_SIZE,
        max_token_length=MAX_TOKEN_LENGTH,
        shuffle=True,
    )
    optimizer_kwargs = dict(
        lr=1e-6, elasticity=2.0, bounding_box_size=0.0001, num_samples=len(dataset)
    )
    for model_name in MODEL_LIST:
        print("Model:", model_name)
        MODEL_NAME = model_name
        _, model = get_model(model_name, device=DEVICE, model_loading_kwargs=kwargs)

        trace, learning_coeff_stats = estimate_llc(
            model,
            dataset,
            config,
            optimizer_kwargs=optimizer_kwargs,
            device=DEVICE,
        )
        learning_coeff_stats["model"] = model_name
        learning_coeff_stats["shuffle_id"] = i
        results.append(learning_coeff_stats)

results_df = pd.DataFrame(results)
timestamp = pd.Timestamp.now().strftime("%Y%m%d-%H%M%S")
results_df.to_csv(f"results_{MODEL_FAMILY}_{SAMPLING_METHOD}_{timestamp}.csv")

agg_table = results_df.groupby("model", group_keys=False).apply(pd_model_grouper)
print(agg_table)
