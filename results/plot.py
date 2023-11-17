import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def pd_model_grouper(group):
    row = group.iloc[0][["mean", "std"]]
    mean = group["mean"]
    mean.loc[mean <= 0] = np.nan
    row["mean"] = mean.mean()
    row["std"] = group["std"].mean()
    return row


df = pd.read_csv("results_gpt2_SGLD_20231117-163104.csv", index_col=0)

agg_table = df.groupby("model", group_keys=False).apply(pd_model_grouper)
agg_table["model_size"] = [117, 774, 345, 1558]
plt.figure(figsize=(8, 5))
sns.lineplot(agg_table, y="mean", x="model_size")

x_values_to_highlight = {
    "gpt2": 117,
    "gpt2-large": 774,
    "gpt2-medium": 345,
    "gpt2-xl": 1558,
}  # List of x-values
for name, x in x_values_to_highlight.items():
    plt.axvline(x=x, color="green", linestyle="--", alpha=0.6)  # Striped vertical line
    plt.text(
        x, plt.gca().get_ylim()[0], f"{name}", color="blue", ha="center", va="bottom"
    )  # Add label

tick_interval = 400  # Adjust this based on your data range
max_x = 1600
new_ticks = range(0, max_x + 1, tick_interval)
new_labels = [f"{tick}M" for tick in new_ticks]
plt.xticks(ticks=new_ticks, labels=new_labels)
plt.ylabel(r"$\hat{\lambda}$", rotation=0)
plt.xlabel("Model Size")
plt.title("Estimates of Local Learning Coefficient for GPT-2 family")
plt.show()

plt.savefig("results_gpt2_llc.png", bbox_inches="tight", dpi=300)
