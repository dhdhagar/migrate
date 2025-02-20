import os
import json
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--strategy", type=str, default="Oracle_Single")
    parser.add_argument("--date", type=str, default="")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()
    return args


def load_results(strat, date, steps):
    targets = [
        "airbase",
        "birthstone",
        "cement",
        "computer",
        "filament",
        "machetes",
        "meatloaf",
        "mob",
        "polyethylene",
        "skillet",
    ]
    results = {strat: []}
    for target in targets:
        files = glob.glob(f"logs/{strat}/{date}/{target}/*.log")
        files = sorted(files)
        if len(files) == 0:
            raise Exception(f"Results at logs/{strat}/{date}/{target}/*.log does not exist")
        samples = []
        for file_name in files:
            sample = []
            with open(file_name, "r") as file:
                data = json.load(file)
                for i in range(steps):
                    try:
                        batch = data["Guesses"][i]
                        batch = batch[list(batch.keys())[0]]
                        sims = [x[1] for x in batch]
                        sample.append([min(sims), max(sims)])
                    except Exception as _:
                        sample.append(sample[-1])
            samples.append(sample)
        results[strat].append(samples)

    base_scores = np.load("random_sample.npy")  # (targets, samples, iterations, min/max sims)
    best_base_scores = np.maximum.accumulate(base_scores[:, :, :, 1], axis=2)
    best_base_scores = np.mean(best_base_scores, axis=1)
    best_base_scores = best_base_scores[:, :, np.newaxis]
    base_scores = np.mean(base_scores, axis=1)
    results["base_scores"] = np.concatenate((base_scores, best_base_scores), axis=2)  # type: ignore
    return results


def plot_strat_scores(results, strat, steps, results_dir):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    scores = np.array(results[strat])  # (targets, samples, steps, min/max)
    best_scores = np.fmax.accumulate(scores[:, :, :, 1], axis=2)  # (targets, samples, steps, best)
    best_scores = np.mean(best_scores, axis=0)  # (samples, steps, best)
    scores = np.mean(scores, axis=0)  # (samples, steps, min/max)

    x = np.arange(0, steps, 1)
    scores_mean = np.mean(scores, axis=0)
    best_scores_mean = np.mean(best_scores, axis=0)
    base_scores = np.mean(results["base_scores"], axis=0)
    if scores.shape[0] > 1:
        scores_std = np.std(scores, axis=0)
        best_scores_std = np.std(best_scores, axis=0)
        plt.fill_between(
            x,
            scores_mean[:, 0] - scores_std[:, 0],
            scores_mean[:, 0] + scores_std[:, 0],
            alpha=0.2,
            color=default_colors[0],
        )
        plt.fill_between(
            x,
            scores_mean[:, 1] - scores_std[:, 1],
            scores_mean[:, 1] + scores_std[:, 1],
            alpha=0.2,
            color=default_colors[1],
        )
        plt.fill_between(
            x,
            best_scores_mean - best_scores_std,
            best_scores_mean + best_scores_std,
            alpha=0.2,
            color=default_colors[2],
        )
    plt.plot(x, scores_mean[:, 0], color=default_colors[0])
    plt.plot(x, scores_mean[:, 1], color=default_colors[1])
    plt.plot(x, best_scores_mean, color=default_colors[2])
    plt.plot(x, base_scores[:, 0], color=default_colors[0], linestyle="--")
    plt.plot(x, base_scores[:, 1], color=default_colors[1], linestyle="--")
    plt.plot(x, base_scores[:, 2], color=default_colors[2], linestyle="--")

    legend_handles = [
        Line2D([0], [0], color=default_colors[0], lw=2, linestyle="-"),
        Line2D([0], [0], color=default_colors[1], lw=2, linestyle="-"),
        Line2D([0], [0], color=default_colors[2], lw=2, linestyle="-"),
        Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    plt.figlegend(
        handles=legend_handles,
        labels=["Batch Mean", "Batch Max", "Best Found", "Sampling 1000 words from base model"],
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0, -0.18, 0.5, 0.5),
    )
    plt.xlabel("Training Iterations")
    plt.ylabel("Cosine Similarity with Target")
    plt.title(f'{strat.replace("_", "-")} Black-Box Scores')
    plt.ylim(0.12, 1.05)
    plt.grid(alpha=0.5)
    plt.savefig(f"{results_dir}/{strat}_scores.png", bbox_inches="tight")
    plt.show()


def plot_strat_accuracy(results, strat, steps, results_dir):
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    scores = np.array(results[strat])  # (targets, samples, steps, min/max)
    best_scores = np.fmax.accumulate(scores[:, :, :, 1], axis=2)  # (targets, samples, steps, best)
    accuracy = np.where(best_scores < 0.99, 0.0, 1.0)
    accuracy = np.mean(accuracy, axis=0)
    accuracy_mean = np.mean(accuracy, axis=0)
    base_scores = np.array(results["base_scores"])[:, :, 1]  # (targets, samples, steps, min/max)
    base_accuracy = np.where(np.fmax.accumulate(base_scores, axis=1) < 0.99, 0.0, 1.0)
    mean_base_accuracy = np.mean(base_accuracy, axis=0)

    x = np.arange(0, steps, 1)
    if accuracy.shape[0] > 1:
        accuracy_std = np.std(accuracy, axis=0)
        plt.fill_between(
            x, accuracy_mean - accuracy_std, accuracy_mean + accuracy_std, alpha=0.2, color=default_colors[0]
        )
    plt.plot(x, accuracy_mean, color=default_colors[0], label=strat.replace("_", "-"))
    plt.plot(x, mean_base_accuracy, color="black", linestyle="dashed", label="Sampling 1000 words from base model")
    plt.title(f'{strat.replace("_", "-")}: 0-1 Accuracy')
    plt.xlabel("Training Iterations")
    plt.ylabel("% Found")
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.5)
    plt.savefig(f"{results_dir}/{strat}_accuracy.png")
    plt.show()


def main():
    args = parse_arguments()
    params = vars(args)

    results_dir = f'results/{params["strategy"]}/{params["date"]}'
    os.makedirs(results_dir, exist_ok=True)

    results = load_results(params["strategy"], params["date"], params["steps"])
    plot_strat_scores(results, params["strategy"], params["steps"], results_dir)
    plot_strat_accuracy(results, params["strategy"], params["steps"], results_dir)


if __name__ == "__main__":
    main()
