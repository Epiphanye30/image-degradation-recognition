import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_ROOT / "checkpoints" / "full_redo"
FIG_DIR = PROJECT_ROOT / "figures"

PLAN_CONFIG = {
    "Plan A: Standard Split": [
        ("ResNet18", "standard_resnet18"),
        ("R18-trunc3", "standard_resnet18_trunc3"),
        ("R18-trunc2", "standard_resnet18_trunc2"),
        ("R18-trunc1", "standard_resnet18_trunc1"),
        ("LMS-CNN bc32", "standard_customized_bc32"),
        ("LMS-CNN bc16", "standard_customized_bc16"),
        ("LMS-CNN bc8", "standard_customized_bc8"),
        ("LMS-CNN bc4", "standard_customized_bc4"),
    ],
}

MODEL_SIZE_PROXY = {
    "ResNet18": 11180103,
    "R18-trunc1": 157959,
    "R18-trunc2": 683975,
    "R18-trunc3": 2784583,
    "LMS-CNN bc32": 33784,
    "LMS-CNN bc16": 11320,
    "LMS-CNN bc8": 4720,
    "LMS-CNN bc4": 2596,
}

PART_B_SOURCES = {
    "ood_customized_bc8": "LMS-CNN-bc8",
    "ood_customized_bc16": "LMS-CNN-bc16",
    "ood_resnet18_trunc2": "ResNet18-trunc2",
    "ood_resnet18": "ResNet18",
}

PART_B_PARAMETER_PROXY = {
    "ood_customized_bc8": 4720,
    "ood_customized_bc16": 11320,
    "ood_resnet18_trunc2": 683975,
    "ood_resnet18": 11180103,
}

PART_C_SUBPLOTS = {
    "Transfer Strategy": {
        "ResNet18": ["standard_resnet18", "transfer_resnet18_mdc_to_foundir"],
        "ResNet18-trunc2": ["standard_resnet18_trunc2", "transfer_resnet18_trunc2_mdc_to_foundir"],
    },
    "Mix Strategy": {
        "ResNet18": ["mix_resnet18"],
        "ResNet18-trunc2": ["mix_resnet18_trunc2"],
    },
}

PART_C_COLORS = {
    ("ResNet18", "MDC"): "#1A5276",
    ("ResNet18", "FoundIR"): "#3498DB",
    ("ResNet18-trunc2", "MDC"): "#922B21",
    ("ResNet18-trunc2", "FoundIR"): "#E67E22",
}

PRIMARY = "#1f4e79"
SECONDARY = "#d95f02"
ACCENT = "#2a9d8f"
HIGHLIGHT = "#c84c09"
GRID_ALPHA = 0.18


plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "#fbfbf8",
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate experiment figures from checkpoint CSV metrics.")
    parser.add_argument(
        "--figure",
        default="all",
        choices=[
            "all",
            "fig1",
            "fig2",
            "fig3",
            "fig4",
            "fig5",
            "part-b",
            "part-c",
        ],
        help="Figure target to generate.",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=ROOT_DIR,
        help="Directory containing experiment subfolders with metrics.csv and test_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FIG_DIR,
        help="Directory used for saved figures.",
    )
    return parser.parse_args()


def safe_get_metric(row: dict, key: str):
    if key not in row:
        return None
    value = row[key]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except Exception:
        return None


def load_csv(root_dir: Path, exp_dir: str, filename: str) -> pd.DataFrame:
    path = root_dir / exp_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path, skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if df.empty:
        raise ValueError(f"Empty CSV: {path}")
    return df


def load_test_row(root_dir: Path, exp_dir: str) -> dict:
    df = load_csv(root_dir, exp_dir, "test_metrics.csv")
    return df.iloc[-1].to_dict()


def load_metrics_df(root_dir: Path, exp_dir: str) -> pd.DataFrame:
    return load_csv(root_dir, exp_dir, "metrics.csv")


def annotate_bars(ax, xs, ys, fontsize=8):
    for x, y in zip(xs, ys):
        if y is None or pd.isna(y):
            continue
        ax.text(x, y + 0.01, f"{y:.3f}", ha="center", va="bottom", fontsize=fontsize)


def apply_common_axis_style(ax, y_label=None, x_label=None, y_lim=(0.0, 1.0)):
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    ax.grid(axis="y", alpha=GRID_ALPHA, linestyle="--", linewidth=0.8)
    ax.set_axisbelow(True)


def millions_formatter(x, _pos):
    if x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def get_validation_curve(df: pd.DataFrame):
    if "selection_score" in df.columns:
        return df["selection_score"]
    if "mean_val_micro_f1" in df.columns:
        return df["mean_val_micro_f1"]
    if "val0_micro_f1" in df.columns:
        return df["val0_micro_f1"]
    return None


def collect_plan_metrics(root_dir: Path, plan_items, metric_keys):
    labels = []
    results = {key: [] for key in metric_keys}

    for display_name, exp_dir in plan_items:
        row = load_test_row(root_dir, exp_dir)
        labels.append(display_name)
        for key in metric_keys:
            results[key].append(safe_get_metric(row, key))

    return labels, results


def plot_fig1_plan_a(root_dir: Path, output_dir: Path):
    output_path = output_dir / "fig1_planA_microf1_exactmatch.png"
    labels, metrics = collect_plan_metrics(
        root_dir,
        PLAN_CONFIG["Plan A: Standard Split"],
        ["test0_micro_f1", "test0_exact_match"],
    )

    micro_f1_scores = metrics["test0_micro_f1"]
    exact_match_scores = metrics["test0_exact_match"]
    x = list(range(len(labels)))
    width = 0.36
    x_micro = [i - width / 2 for i in x]
    x_exact = [i + width / 2 for i in x]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.bar(x_micro, micro_f1_scores, width=width, label="Test Micro-F1", color=PRIMARY, edgecolor="white", linewidth=0.8)
    ax.bar(x_exact, exact_match_scores, width=width, label="Test Exact Match", color=SECONDARY, edgecolor="white", linewidth=0.8)

    ax.set_title("Test Performance among Different Models")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    apply_common_axis_style(ax, y_label="Score", y_lim=(0.0, 1.0))
    ax.legend()
    annotate_bars(ax, x_micro, micro_f1_scores)
    annotate_bars(ax, x_exact, exact_match_scores)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig2_capacity_vs_performance(root_dir: Path, output_dir: Path):
    output_path = output_dir / "fig2_planA_capacity_vs_performance.png"
    labels = []
    micro_scores = []
    exact_scores = []
    sizes = []

    for display_name, exp_dir in PLAN_CONFIG["Plan A: Standard Split"]:
        row = load_test_row(root_dir, exp_dir)
        labels.append(display_name)
        micro_scores.append(safe_get_metric(row, "test0_micro_f1"))
        exact_scores.append(safe_get_metric(row, "test0_exact_match"))
        sizes.append(MODEL_SIZE_PROXY.get(display_name))

    fig, ax = plt.subplots(figsize=(10, 7))

    for label, size, micro, exact in zip(labels, sizes, micro_scores, exact_scores):
        if size is None:
            warnings.warn(f"Skip {label} due to missing size")
            continue

        if micro is not None:
            ax.scatter(size, micro, s=120, color=ACCENT, edgecolor="white", linewidth=0.9, marker="o", zorder=3)
            ax.text(size, micro + 0.012, label, fontsize=9, ha="center")

        if exact is not None:
            ax.scatter(size, exact, s=120, color=SECONDARY, edgecolor="white", linewidth=0.9, marker="s", zorder=3)

    ax.set_title("Capacity vs Performance")
    apply_common_axis_style(ax, y_label="Score", x_label="Trainable Parameters (log scale)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
    ax.grid(axis="x", alpha=0.12, linestyle=":")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", label="Micro-F1", markerfacecolor=ACCENT, markersize=10),
            Line2D([0], [0], marker="s", color="w", label="Exact Match", markerfacecolor=SECONDARY, markersize=10),
        ],
        loc="lower right",
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_train_val_curves(ax, root_dir: Path, items, title):
    color_map = {"pretrained": PRIMARY, "scratch": SECONDARY}

    for display_name, exp_dir in items:
        try:
            df = load_metrics_df(root_dir, exp_dir)
        except Exception as exc:
            warnings.warn(f"Skip curve for {exp_dir}: {exc}")
            continue

        x = df["epoch"] if "epoch" in df.columns else list(range(len(df)))
        lower_name = display_name.lower()
        color = color_map["pretrained"] if "pretrain" in lower_name or "pretrained" in lower_name else color_map["scratch"]

        if "train_micro_f1" in df.columns:
            ax.plot(x, df["train_micro_f1"], linestyle="-", linewidth=2.0, color=color, label=f"{display_name} train")

        val_curve = get_validation_curve(df)
        if val_curve is not None:
            ax.plot(x, val_curve, linestyle="--", linewidth=2.0, color=color, alpha=0.9, label=f"{display_name} val")

    ax.set_title(title)
    apply_common_axis_style(ax, x_label="Epoch", y_label="Micro-F1")
    ax.legend(fontsize=8)


def plot_fig3_pretrain_curves(root_dir: Path, output_dir: Path):
    output_path = output_dir / "fig3_pretrain_vs_scratch_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    plot_train_val_curves(
        axes[0],
        root_dir,
        [
            ("R18 pretrained", "pretrain_cmp_resnet18_pretrained"),
            ("R18 scratch", "pretrain_cmp_resnet18_no_pretrained"),
        ],
        "ResNet18: Pretrained vs Scratch",
    )
    plot_train_val_curves(
        axes[1],
        root_dir,
        [
            ("R18-t2 pretrained", "pretrain_cmp_resnet18_trunc2_pretrained"),
            ("R18-t2 scratch", "pretrain_cmp_resnet18_trunc2_no_pretrained"),
        ],
        "R18-trunc2: Pretrained vs Scratch",
    )

    fig.suptitle("Training Curves: Pretrained vs Scratch", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig4_basechannel_curves(root_dir: Path, output_dir: Path):
    output_path = output_dir / "fig4_customized_basechannel_curves.png"
    items = [
        ("LMS-CNN bc32", "standard_customized_bc32"),
        ("LMS-CNN bc16", "standard_customized_bc16"),
        ("LMS-CNN bc8", "standard_customized_bc8"),
        ("LMS-CNN bc4", "standard_customized_bc4"),
    ]

    fig, ax = plt.subplots(figsize=(11, 7))
    colors = plt.get_cmap("tab10").colors

    for idx, (display_name, exp_dir) in enumerate(items):
        try:
            df = load_metrics_df(root_dir, exp_dir)
        except Exception as exc:
            warnings.warn(f"Skip curve for {exp_dir}: {exc}")
            continue

        x = df["epoch"] if "epoch" in df.columns else list(range(len(df)))
        color = colors[idx]

        if "train_micro_f1" in df.columns:
            ax.plot(x, df["train_micro_f1"], linestyle="-", linewidth=2.2, color=color, label=f"{display_name} train")

        val_curve = get_validation_curve(df)
        if val_curve is not None:
            ax.plot(x, val_curve, linestyle="--", linewidth=2.2, color=color, label=f"{display_name} val")

    ax.set_title("Customized Models: Training Curves Across Base Channels")
    apply_common_axis_style(ax, x_label="Epoch", y_label="Micro-F1")
    ax.legend(fontsize=9, ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_fig5_pretrain_final(root_dir: Path, output_dir: Path):
    output_path = output_dir / "fig5_pretrain_vs_scratch_final_performance.png"
    groups = {
        "ResNet18": [
            ("Pretrained", "pretrain_cmp_resnet18_pretrained"),
            ("Scratch", "pretrain_cmp_resnet18_no_pretrained"),
        ],
        "R18-trunc2": [
            ("Pretrained", "pretrain_cmp_resnet18_trunc2_pretrained"),
            ("Scratch", "pretrain_cmp_resnet18_trunc2_no_pretrained"),
        ],
    }

    backbones = list(groups.keys())
    micro_pre = []
    micro_scr = []
    exact_pre = []
    exact_scr = []

    for backbone in backbones:
        items = groups[backbone]
        row_pre = load_test_row(root_dir, items[0][1])
        row_scr = load_test_row(root_dir, items[1][1])
        micro_pre.append(safe_get_metric(row_pre, "test0_micro_f1"))
        micro_scr.append(safe_get_metric(row_scr, "test0_micro_f1"))
        exact_pre.append(safe_get_metric(row_pre, "test0_exact_match"))
        exact_scr.append(safe_get_metric(row_scr, "test0_exact_match"))

    x = list(range(len(backbones)))
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    ax = axes[0]
    ax.bar([i - width / 2 for i in x], micro_pre, width=width, color=PRIMARY, label="Pretrained")
    ax.bar([i + width / 2 for i in x], micro_scr, width=width, color=SECONDARY, label="Scratch")
    ax.set_title("Micro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    apply_common_axis_style(ax, y_label="Score")
    annotate_bars(ax, [i - width / 2 for i in x], micro_pre)
    annotate_bars(ax, [i + width / 2 for i in x], micro_scr)
    ax.legend()

    ax = axes[1]
    ax.bar([i - width / 2 for i in x], exact_pre, width=width, color=PRIMARY)
    ax.bar([i + width / 2 for i in x], exact_scr, width=width, color=SECONDARY)
    ax.set_title("Exact Match")
    ax.set_xticks(x)
    ax.set_xticklabels(backbones)
    apply_common_axis_style(ax)
    annotate_bars(ax, [i - width / 2 for i in x], exact_pre)
    annotate_bars(ax, [i + width / 2 for i in x], exact_scr)

    fig.suptitle("Pretrained vs Scratch: Final Performance", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def transform_part_b_x(params):
    log_p = np.log10(params)
    if 5.0 < log_p < 6.5:
        return log_p - 1.2
    if log_p >= 6.5:
        return log_p - 2.0
    return log_p


def transform_part_b_y(score):
    if score < 0.7:
        return score * 0.2
    return score - 0.56


def draw_axis_break(ax, pos, axis="x"):
    delta = 0.015
    if axis == "x":
        ax.plot([pos - delta, pos + delta], [-0.02, 0.02], transform=ax.get_xaxis_transform(), color="k", clip_on=False)
        ax.plot(
            [pos - delta + 0.02, pos + delta + 0.02],
            [-0.02, 0.02],
            transform=ax.get_xaxis_transform(),
            color="k",
            clip_on=False,
        )
    else:
        ax.plot([-0.012, 0.012], [pos - delta, pos + delta], transform=ax.get_yaxis_transform(), color="k", clip_on=False)
        ax.plot(
            [-0.012, 0.012],
            [pos - delta + 0.02, pos + delta + 0.02],
            transform=ax.get_yaxis_transform(),
            color="k",
            clip_on=False,
        )


def plot_part_b_ood(root_dir: Path, output_dir: Path):
    output_path = output_dir / "part_b_ood_generalization.png"
    all_data = []

    for folder, label in PART_B_SOURCES.items():
        path = root_dir / folder / "test_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, skipinitialspace=True)
        df.columns = df.columns.str.strip()
        row = df.iloc[-1]
        all_data.append(
            {
                "label": label,
                "params": row["parameters"] if "parameters" in row else PART_B_PARAMETER_PROXY[folder],
                "f1": row["test0_micro_f1"],
                "em": row["test0_exact_match"],
            }
        )

    if not all_data:
        raise FileNotFoundError(f"No Part B data found under {root_dir}")

    df_plot = pd.DataFrame(all_data).sort_values("params")
    df_plot["x_plot"] = df_plot["params"].apply(transform_part_b_x)
    df_plot["y_f1_plot"] = df_plot["f1"].apply(transform_part_b_y)
    df_plot["y_em_plot"] = df_plot["em"].apply(transform_part_b_y)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    color_teal = "#2A9D8F"
    color_orange = "#D6620F"

    ax.scatter(df_plot["x_plot"], df_plot["y_f1_plot"], color=color_teal, s=120, label="Micro-F1", zorder=5)
    ax.scatter(df_plot["x_plot"], df_plot["y_em_plot"], color=color_orange, s=100, marker="s", label="Exact Match", zorder=5)

    for _, row in df_plot.iterrows():
        ax.text(row["x_plot"], row["y_f1_plot"] + 0.015, row["label"], ha="center", va="bottom", fontsize=9)

    y_ticks_orig = [0.0, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks([transform_part_b_y(y) for y in y_ticks_orig], ["0.0", "0.7", "0.8", "0.9", "1.0"])

    x_ticks_orig = [5000, 10000, 700000, 11000000]
    ax.set_xticks([transform_part_b_x(x) for x in x_ticks_orig], ["5K", "10K", "0.7M", "11M"])

    draw_axis_break(ax, (transform_part_b_x(15000) + transform_part_b_x(700000)) / 2, "x")
    draw_axis_break(ax, (transform_part_b_x(800000) + transform_part_b_x(10000000)) / 2, "x")
    draw_axis_break(ax, transform_part_b_y(0.35), "y")

    ax.set_title("Part B: Out of Domain Generalization", fontsize=18, fontweight="bold", pad=20)
    ax.set_ylabel("Score", fontsize=18)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {output_path}")


def load_part_c_data(root_dir: Path, folders):
    frames = []
    for folder in folders:
        path = root_dir / folder / "metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, skipinitialspace=True)
        frame.columns = frame.columns.str.strip()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def plot_part_c_strategy(root_dir: Path, output_dir: Path):
    output_path = output_dir / "part_c_training_strategy.png"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8.5), sharey=True, dpi=150)
    fig.suptitle("Part C: Training Strategy", fontsize=22, fontweight="bold", y=0.98)
    axes = {"Transfer Strategy": ax1, "Mix Strategy": ax2}

    for title, models in PART_C_SUBPLOTS.items():
        ax = axes[title]
        for model_name, folders in models.items():
            df = load_part_c_data(root_dir, folders)
            if df is None:
                continue

            epochs = np.arange(1, len(df) + 1)
            c0 = PART_C_COLORS[(model_name, "MDC")]
            c1 = PART_C_COLORS[(model_name, "FoundIR")]

            ax.plot(epochs, df["val0_micro_f1"], color=c0, linestyle="-", lw=2.5, label=f"{model_name} MDC F1")
            ax.plot(epochs, df["val1_micro_f1"], color=c1, linestyle="-", lw=2.5, label=f"{model_name} FoundIR F1")
            ax.plot(
                epochs,
                df["val0_exact_match"],
                color=c0,
                linestyle="--",
                lw=1.8,
                alpha=0.8,
                label=f"{model_name} MDC Exact Match",
            )
            ax.plot(
                epochs,
                df["val1_exact_match"],
                color=c1,
                linestyle="--",
                lw=1.8,
                alpha=0.8,
                label=f"{model_name} FoundIR Exact Match",
            )

        ax.set_title(title, fontsize=18, fontweight="bold", pad=20)
        ax.set_xlabel("Epochs", fontsize=18)
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)

        if title == "Transfer Strategy":
            ax.axvline(x=20, color="#333333", linestyle="--", linewidth=3, alpha=0.8, zorder=1)
            bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#333333", lw=1.5, alpha=0.9)
            ax.text(10, 0.08, "Phase 1 (MDC)", ha="center", fontsize=18, fontweight="bold", bbox=bbox_props)
            ax.text(30, 0.08, "Phase 2 (FoundIR)", ha="center", fontsize=18, fontweight="bold", bbox=bbox_props)

    ax1.set_ylabel("Score", fontsize=18)
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
        order = [0, 1, 4, 5, 2, 3, 6, 7]
        valid_order = [idx for idx in order if idx < len(handles)]
        fig.legend(
            [handles[idx] for idx in valid_order],
            [labels[idx] for idx in valid_order],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fontsize=14,
            frameon=True,
            shadow=True,
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


FIGURE_RUNNERS = {
    "fig1": plot_fig1_plan_a,
    "fig2": plot_fig2_capacity_vs_performance,
    "fig3": plot_fig3_pretrain_curves,
    "fig4": plot_fig4_basechannel_curves,
    "fig5": plot_fig5_pretrain_final,
    "part-b": plot_part_b_ood,
    "part-c": plot_part_c_strategy,
}


def main():
    args = parse_args()
    root_dir = args.checkpoints_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.figure == "all":
        for runner in FIGURE_RUNNERS.values():
            runner(root_dir, output_dir)
        print(f"All figures saved under: {output_dir}")
        return

    FIGURE_RUNNERS[args.figure](root_dir, output_dir)


if __name__ == "__main__":
    main()
