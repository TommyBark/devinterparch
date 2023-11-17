import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForImageClassification,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD, SGNHT
import yaml

from typing import Optional
import random
import numpy as np


def is_model_quantized(model) -> bool:
    for _, param in model.named_parameters():
        if param.dtype == torch.uint8:
            return True
    return False


def get_model(
    model_name: str, device: str = "cuda", model_loading_kwargs=Optional[dict]
):
    if "llama" in model_name:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, **model_loading_kwargs)
        model = LlamaForCausalLM.from_pretrained(model_name, **model_loading_kwargs)
        if not is_model_quantized(model):
            model = model.to(device)
        # model.config.pad_token_id = model.config.eos_token_id
    elif model_name == "resnet-tiny-mnist":
        model = AutoModelForImageClassification.from_pretrained(
            "fxmarty/resnet-tiny-mnist"
        ).to(device)
        tokenizer = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, **model_loading_kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **model_loading_kwargs
        ).to(device)
        model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model


def get_dataset(
    dataset_name: str,
    tokenizer,
    batch_size: int = 256,
    max_token_length: int = 200,
    shuffle: bool = True,
):
    dataset = load_dataset(dataset_name, split="train")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dataset = CustomDataset(
        dataset,
        tokenizer=tokenizer,
        padding="max_length",
        max_length=max_token_length,
        min_text_length=max_token_length*3.5,  # 3.5 to approximate average token char length
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataset


def criterion(inputs, outputs):
    return F.cross_entropy(
        inputs.logits, outputs
    )  # transformers doesn't output a vector


def criterion_llm(inputs, outputs):
    return CrossEntropyLoss()(inputs.logits[:, -1, :], outputs)


def pd_model_grouper(group):
    row = group.iloc[0][["mean", "std"]]
    mean = group["mean"]
    mean.loc[mean <= 0] = np.nan
    row["mean"] = mean.mean()
    row["std"] = group["std"].mean()
    return row


class CustomDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        tokenizer,
        batch_size,
        max_length=1024,
        padding="max_length",
        min_text_length: Optional[int] = None,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.min_text_length = min_text_length
        self.batch_size = batch_size

        if self.min_text_length is not None:
            self.dataset = [
                t for t in hf_dataset["text"] if len(t) > self.min_text_length
            ]
        else:
            self.dataset = hf_dataset["text"]

        if random_seed is not None:
            random.seed(random_seed)

        if shuffle:
            random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))

        batch_texts = self.dataset[start_idx:end_idx]
        if self.padding == "max_length":
            encoding = self.tokenizer(
                batch_texts,
                padding="max_length",
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            )
        else:
            encoding = self.tokenizer(
                batch_texts, padding="longest", truncation=True, return_tensors="pt"
            )
        return encoding["input_ids"][:, :-1], encoding["input_ids"][:, -1]


class CustomCrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, inputs, outputs):
        return super().forward(inputs.logits[:, -1, :], outputs)


def estimate_llc(
    model,
    dataset,
    config: dict,
    optimizer_kwargs: dict | None = None,
    device: torch.device = torch.device("cuda"),
):
    num_samples = len(dataset)
    sampling = config["sampling_method"]
    if sampling == "SGLD":
        sampling_method = SGLD
        if optimizer_kwargs is None:
            optimizer_kwargs = dict(lr=1e-5, elasticity=1.0, num_samples=num_samples)
    elif sampling == "SGNHT":
        sampling_method = SGNHT
        if optimizer_kwargs is None:
            optimizer_kwargs = dict(
                lr=1e-5, diffusion_factor=0.01, num_samples=num_samples
            )
    else:
        raise ValueError(f"Sampling method {sampling} not recognized.")

    optimizer_kwargs["num_samples"] = optimizer_kwargs.get("num_samples", num_samples)

    learning_coeff_stats = estimate_learning_coeff_with_summary(
        model,
        loader=dataset,
        criterion=CustomCrossEntropyLoss,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_chains=config["num_chains"],
        num_draws=config["num_draws"],
        num_burnin_steps=config["num_burnin_steps"],
        num_steps_bw_draws=config["num_steps_bw_draws"],
        clip_chains_quantile=config["clip_chains_quantile"],
        device=device,
    )

    trace = learning_coeff_stats.pop("trace")

    print("Lambda hat estimates:")
    print(yaml.dump(learning_coeff_stats))

    return trace, learning_coeff_stats
