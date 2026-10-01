"""Plot the threshold sweep for the blog post: runs/sweep/report.json from
python/threshold_sweep.py if present, otherwise the committed results/sweep_report.json.

Top: correct answers for the cascade and for decider-2b alone. Bottom: the
AI_CLASSIFY cost the cascade adds per 1,000 decisions, from the pre-counted
input tokens at the metered rate (warehouse time not included).

    uv run --no-project --with matplotlib==3.10.1 python python/plot_threshold_sweep.py <out.png>
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
AI_CREDITS_PER_TOKEN = 1.39 / 1e6   # the cascade's metered AI_CLASSIFY rate
USD_PER_AI_CREDIT = 2.00


def main(out):
    src = ROOT / "runs" / "sweep" / "report.json"
    if not src.exists():
        src = ROOT / "results" / "sweep_report.json"
    report = json.loads(src.read_text())
    n = report["n"]
    xs = [r["threshold"] for r in report["sweep"]]
    ys = [100 * r["correct"] / n for r in report["sweep"]]
    usd = [r["ai_classify_input_tokens"] * AI_CREDITS_PER_TOKEN * USD_PER_AI_CREDIT / n * 1000
           for r in report["sweep"]]
    base = 100 * report["decider_alone_correct"] / n

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6), dpi=200, sharex=True,
                                      gridspec_kw={"height_ratios": [3, 2]})
    top.plot(xs, ys, color="#29B5E8", linewidth=2.2, marker="o", markersize=4.5,
             label="decider-2b cascaded to AI_CLASSIFY")
    top.axhline(base, color="#5B6770", linewidth=1.6, linestyle="--", label=f"decider-2b alone ({base:.1f}%)")
    top.set_ylabel(f"Correct answers (% of {n})")
    top.yaxis.set_major_formatter(FormatStrFormatter("%.0f%%"))
    top.set_ylim(73.5, 80)
    top.legend(frameon=False, loc="lower left", fontsize=9)
    top.set_title(f"After-the-fact sweep over JevBench's {n} public decisions", fontsize=10, loc="left", color="#333333")

    bottom.plot(xs, usd, color="#11567F", linewidth=2.2, marker="o", markersize=4.5)
    bottom.set_ylabel("AI_CLASSIFY cost added\nper 1,000 decisions")
    bottom.yaxis.set_major_formatter(FormatStrFormatter("$%.2f"))
    bottom.set_ylim(0, 1.9)
    bottom.set_xlabel("Escalate to AI_CLASSIFY when decider's confidence is below")
    bottom.set_xlim(-0.02, 0.92)
    bottom.set_xticks([round(0.1 * k, 1) for k in range(10)])

    for ax in (top, bottom):
        ax.grid(axis="y", color="#E5E8EB", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main(sys.argv[1])
