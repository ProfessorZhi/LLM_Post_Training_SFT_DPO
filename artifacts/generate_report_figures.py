from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = ROOT / "submit" / "artifacts"
SCORES_DIR = ROOT / "work" / "eval" / "_FINAL_REPORT_EVIDENCE_20260524" / "scores"
LOG_DIR = ROOT / "work" / "training_logs" / "_FINAL_REPORT_EVIDENCE_20260524"

SCORE_FILES = {
    "base": "score_base_mcore_v2_1_cleanpath_20260517_134600_eval30.json",
    "v24_750": "score_v2_4_xxl_cleanpath_750step_20260518_201340_eval30.json",
    "v24_1000": "score_v2_4_xxl_cleanpath_1000step_20260518_202958_eval30.json",
    "v25_250": "score_v2_5_fnonly_filtered_250step_eval30_20260519_153017_v2_5_fnonly_filtered_250step_eval30.json",
    "v25_500": "score_v2_5_fnonly_filtered_500step_eval30_20260519_154620_v2_5_fnonly_filtered_500step_eval30.json",
    "v25_750": "score_v2_5_fnonly_filtered_750step_eval30_rescored_current_20260524.json",
    "v25_1000": "score_v2_5_fnonly_filtered_1000step_eval30_20260519_161743_v2_5_fnonly_filtered_1000step_eval30.json",
    "v25_best_500": "score_v2_5_fnonly_filtered_bestckpt_500step_20260519_232046_eval30.json",
    "dpo_20": "score_v2_5_dpo_smoke_20step_eval30_20260520_124608.json",
}

LOG_1000 = LOG_DIR / "20260519_142908_fim_sft_v2_5_fnonly_filtered_1000step_sft_1000step.outer.log"
LOG_2000 = LOG_DIR / "20260519_212954_fim_sft_v2_5_fnonly_filtered_2000step_closedloop_sft_2000step.outer.log"

TRAIN_RE = re.compile(r"iteration\s+(\d+)/\s+(\d+).*?lm loss:\s+([0-9.E+-]+)")
VAL_RE = re.compile(r"validation iteration\s+(\d+)/\s+(\d+).*?val lm loss:\s+([0-9.E+-]+)")
EARLY_STOP_RE = re.compile(r"trigger at iteration\s+(\d+)")


def load_score(name: str) -> dict:
    data = json.loads((SCORES_DIR / SCORE_FILES[name]).read_text(encoding="utf-8"))
    overall = data["results"][0]["overall"]
    return {
        "composite_score": overall["composite_score"],
        "avg_similarity": overall["avg_similarity"],
        "syntax_ok_rate": overall["syntax_ok_rate"],
        "boundary_ok_rate": overall["boundary_ok_rate"],
        "repetition_issue": overall["repetition_issue"],
        "length_outlier": overall["length_outlier"],
    }


def parse_train_losses(path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = TRAIN_RE.search(line)
        if match and "validation iteration" not in line:
            rows.append(
                {
                    "step": int(match.group(1)),
                    "loss": float(match.group(3)),
                }
            )
    df = pd.DataFrame(rows)
    df["loss_smooth_25"] = df["loss"].rolling(window=25, min_periods=1).mean()
    df["loss_smooth_30"] = df["loss"].rolling(window=30, min_periods=1).mean()
    return df


def parse_validation_losses(path: Path) -> tuple[pd.DataFrame, int | None]:
    rows: list[dict] = []
    early_stop_iter = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        val_match = VAL_RE.search(line)
        if val_match:
            rows.append({"step": int(val_match.group(1)), "val_loss": float(val_match.group(3))})
        stop_match = EARLY_STOP_RE.search(line)
        if stop_match:
            early_stop_iter = int(stop_match.group(1))
    return pd.DataFrame(rows), early_stop_iter


def ensure_dir() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def configure_style() -> None:
    sns.set_theme(
        style="whitegrid",
        context="talk",
        rc={
            "axes.facecolor": "#fbfaf7",
            "figure.facecolor": "#fbfaf7",
            "grid.color": "#ddd4ca",
            "axes.edgecolor": "#4a4036",
            "axes.labelcolor": "#2c251f",
            "xtick.color": "#5a5148",
            "ytick.color": "#5a5148",
            "font.family": "DejaVu Sans",
        },
    )


def annotate_bars(ax, decimals: int = 4, offset: float = 0.0012) -> None:
    for patch in ax.patches:
        value = patch.get_height()
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            value + offset,
            f"{value:.{decimals}f}",
            ha="center",
            va="bottom",
            fontsize=11,
            color="#2f2923",
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.6", "fc": "#fffdfa", "ec": "#c8b7a8", "lw": 1},
        )


def save_fig(fig: plt.Figure, name: str) -> Path:
    out = ARTIFACT_DIR / name
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def build_score_frames(scores: dict[str, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    milestone_df = pd.DataFrame(
        [
            {"model": "Base", "composite": scores["base"]["composite_score"], "stage": "Base"},
            {"model": "V2.4-1000", "composite": scores["v24_1000"]["composite_score"], "stage": "SFT"},
            {"model": "V2.5-750", "composite": scores["v25_750"]["composite_score"], "stage": "SFT"},
            {"model": "DPO-20", "composite": scores["dpo_20"]["composite_score"], "stage": "DPO"},
        ]
    )
    checkpoint_df = pd.DataFrame(
        [
            {"step": 250, "composite": scores["v25_250"]["composite_score"]},
            {"step": 500, "composite": scores["v25_500"]["composite_score"]},
            {"step": 750, "composite": scores["v25_750"]["composite_score"]},
            {"step": 1000, "composite": scores["v25_1000"]["composite_score"]},
        ]
    )
    return milestone_df, checkpoint_df


def figure_milestone_compare(milestone_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    palette = ["#6887ae", "#879763", "#be5b31", "#8d6ea3"]
    sns.barplot(data=milestone_df, x="model", y="composite", hue="model", palette=palette, dodge=False, legend=False, ax=ax)
    annotate_bars(ax)
    ax.set_title("Fixed Eval Milestones", loc="left", fontsize=19, weight="bold", pad=24)
    ax.text(0, 1.04, "The strongest point is still the V2.5 750-step SFT checkpoint.", transform=ax.transAxes, fontsize=11, color="#6b6055")
    ax.set_xlabel("")
    ax.set_ylabel("composite score")
    ax.set_ylim(0.27, 0.36)
    sns.despine(ax=ax, left=False, bottom=False)
    fig.subplots_adjust(top=0.86)
    return save_fig(fig, "report_fig_milestone_compare.png")


def figure_checkpoint_trend(checkpoint_df: pd.DataFrame, base_score: float, v24_score: float) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    sns.lineplot(data=checkpoint_df, x="step", y="composite", marker="o", linewidth=3.2, markersize=10, color="#be5b31", ax=ax)
    ax.axhline(base_score, linestyle="--", linewidth=2, color="#6887ae", alpha=0.95, label=f"Base {base_score:.4f}")
    ax.axhline(v24_score, linestyle="-.", linewidth=2, color="#879763", alpha=0.95, label=f"V2.4-1000 {v24_score:.4f}")
    for row in checkpoint_df.itertuples():
        ax.annotate(
            f"{row.composite:.4f}",
            (row.step, row.composite),
            textcoords="offset points",
            xytext=(0, 12 if row.step != 750 else 18),
            ha="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.6", "fc": "#fffdfa", "ec": "#be5b31", "lw": 1},
        )
    ax.set_title("V2.5 Checkpoint Trend", loc="left", fontsize=19, weight="bold", pad=24)
    ax.text(0, 1.04, "750-step beats 1000-step on the shared fixed eval set.", transform=ax.transAxes, fontsize=11, color="#6b6055")
    ax.set_xlabel("training step")
    ax.set_ylabel("composite score")
    ax.set_xticks(checkpoint_df["step"].tolist())
    ax.set_ylim(0.30, 0.355)
    ax.legend(frameon=False, loc="lower right")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.subplots_adjust(top=0.86)
    return save_fig(fig, "report_fig_checkpoint_trend.png")


def figure_sft_loss(train_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.plot(train_df["step"], train_df["loss"], color="#d7c9bd", linewidth=1.0, alpha=0.9, label="raw loss")
    ax.plot(train_df["step"], train_df["loss_smooth_25"], color="#1f6f78", linewidth=3.0, label="25-step moving average")

    first = train_df.iloc[0]
    last = train_df.iloc[-1]
    ax.annotate(
        f"start {first['loss']:.4f}",
        (first["step"], first["loss_smooth_25"]),
        textcoords="offset points",
        xytext=(10, 14),
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.22,rounding_size=0.6", "fc": "#fffdfa", "ec": "#1f6f78", "lw": 1},
    )
    ax.annotate(
        f"step 1000 {last['loss']:.4f}",
        (last["step"], last["loss_smooth_25"]),
        textcoords="offset points",
        xytext=(-86, 12),
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.22,rounding_size=0.6", "fc": "#fffdfa", "ec": "#1f6f78", "lw": 1},
    )
    ax.set_title("V2.5 1000-step Training Loss", loc="left", fontsize=19, weight="bold", pad=24)
    ax.text(0, 1.04, "The per-step curve is noisy, but the smoothed trend keeps drifting downward.", transform=ax.transAxes, fontsize=11, color="#6b6055")
    ax.set_xlabel("training step")
    ax.set_ylabel("lm loss")
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.40, 1.32)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.subplots_adjust(top=0.86)
    return save_fig(fig, "report_fig_sft1000_loss.png")


def figure_closedloop(train_df: pd.DataFrame, val_df: pd.DataFrame, early_stop_iter: int | None) -> Path:
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.plot(train_df["step"], train_df["loss_smooth_30"], color="#8d6ea3", linewidth=2.6, label="train loss (30-step MA)")
    ax.plot(val_df["step"], val_df["val_loss"], color="#be5b31", linewidth=2.8, marker="o", markersize=7, label="validation loss")

    best_row = val_df.loc[val_df["val_loss"].idxmin()]
    ax.annotate(
        f"best val @ {int(best_row['step'])}: {best_row['val_loss']:.4f}",
        (best_row["step"], best_row["val_loss"]),
        textcoords="offset points",
        xytext=(12, 16),
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.22,rounding_size=0.6", "fc": "#fffdfa", "ec": "#be5b31", "lw": 1},
    )

    if early_stop_iter is not None:
        last_val = val_df.loc[val_df["step"] == early_stop_iter, "val_loss"]
        y = float(last_val.iloc[0]) if not last_val.empty else float(val_df.iloc[-1]["val_loss"])
        ax.axvline(early_stop_iter, linestyle="--", linewidth=1.8, color="#4a4036", alpha=0.8)
        ax.annotate(
            f"early stop @ {early_stop_iter}",
            (early_stop_iter, y),
            textcoords="offset points",
            xytext=(-96, 12),
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.22,rounding_size=0.6", "fc": "#fffdfa", "ec": "#4a4036", "lw": 1},
        )

    ax.set_title("Closed-loop Validation and Early Stop", loc="left", fontsize=19, weight="bold", pad=24)
    ax.text(0, 1.04, "Validation improves until 500-step, then weakens and stops at 900-step.", transform=ax.transAxes, fontsize=11, color="#6b6055")
    ax.set_xlabel("training step")
    ax.set_ylabel("loss")
    ax.set_xlim(1, max(int(train_df["step"].max()), 900))
    ax.set_ylim(0.45, 1.35)
    ax.legend(frameon=False, loc="upper right")
    sns.despine(ax=ax, left=False, bottom=False)
    fig.subplots_adjust(top=0.86)
    return save_fig(fig, "report_fig_closedloop_validation.png")


def figure_metric_heatmap(scores: dict[str, dict]) -> Path:
    frame = pd.DataFrame(
        {
            "Base": scores["base"],
            "V2.4-1000": scores["v24_1000"],
            "V2.5-500": scores["v25_500"],
            "V2.5-750": scores["v25_750"],
            "V2.5-1000": scores["v25_1000"],
        }
    ).T[["composite_score", "avg_similarity", "syntax_ok_rate", "boundary_ok_rate"]]
    frame.columns = ["composite", "avg_similarity", "syntax_ok", "boundary_ok"]

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    sns.heatmap(frame, annot=True, fmt=".4f", cmap="YlOrBr", linewidths=0.5, linecolor="#f3ece5", cbar=False, ax=ax)
    ax.set_title("Metric Snapshot Across Key Checkpoints", loc="left", fontsize=18, weight="bold", pad=24)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.subplots_adjust(top=0.88)
    return save_fig(fig, "report_fig_metric_heatmap.png")


def interactive_dashboard(milestone_df: pd.DataFrame, checkpoint_df: pd.DataFrame) -> Path:
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=milestone_df["model"],
            y=milestone_df["composite"],
            marker_color=["#6887ae", "#879763", "#be5b31", "#8d6ea3"],
            name="milestones",
            text=[f"{v:.4f}" for v in milestone_df["composite"]],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=checkpoint_df["step"].astype(str),
            y=checkpoint_df["composite"],
            mode="lines+markers+text",
            line={"color": "#be5b31", "width": 3},
            marker={"size": 10},
            text=[f"{v:.4f}" for v in checkpoint_df["composite"]],
            textposition="top center",
            xaxis="x2",
            yaxis="y2",
            name="V2.5 checkpoints",
        )
    )
    fig.update_layout(
        template="plotly_white",
        title="LLM4SE FIM Training Report Dashboard",
        font={"family": "Arial"},
        width=1280,
        height=620,
        paper_bgcolor="#fbfaf7",
        plot_bgcolor="#fbfaf7",
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
        xaxis={"domain": [0.0, 0.44], "anchor": "y", "title": ""},
        yaxis={"domain": [0.0, 1.0], "title": "composite score", "range": [0.27, 0.36]},
        xaxis2={"domain": [0.56, 1.0], "anchor": "y2", "title": "training step"},
        yaxis2={"domain": [0.0, 1.0], "title": "composite score", "range": [0.30, 0.355]},
        showlegend=False,
    )
    out = ARTIFACT_DIR / "report_interactive_dashboard.html"
    fig.write_html(out, include_plotlyjs="cdn")
    return out


def main() -> None:
    ensure_dir()
    configure_style()

    scores = {name: load_score(name) for name in SCORE_FILES}
    milestone_df, checkpoint_df = build_score_frames(scores)
    train_1000_df = parse_train_losses(LOG_1000)
    train_2000_df = parse_train_losses(LOG_2000)
    val_df, early_stop_iter = parse_validation_losses(LOG_2000)
    if early_stop_iter is not None:
        train_2000_df = train_2000_df.loc[train_2000_df["step"] <= early_stop_iter].copy()

    outputs = [
        figure_milestone_compare(milestone_df),
        figure_checkpoint_trend(checkpoint_df, scores["base"]["composite_score"], scores["v24_1000"]["composite_score"]),
        figure_sft_loss(train_1000_df),
        figure_closedloop(train_2000_df, val_df, early_stop_iter),
        figure_metric_heatmap(scores),
        interactive_dashboard(milestone_df, checkpoint_df),
    ]

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
