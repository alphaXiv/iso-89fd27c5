import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # ISO-Merger: exact spectra, mixed functional evidence

    Specialists are useful only if we can combine them without erasing what
    each learned. ISO-Merger tries to do this by moving a model's singular
    directions while restoring the shared base model's singular values.

    **Verdict: partially reproduced.** Every tested matrix preserved its base
    spectrum, and ISO beat checkpoint averaging, but Task Arithmetic beat ISO
    on all five seeds. The two mechanism ablations were also slightly better
    than full ISO in this bounded public-task substitution.
    """)
    return


@app.cell
def _():
    data = {
        "seeds": [11, 22, 33, 44, 55],
        "base": [0.545545] * 5,
        "average": [0.609917, 0.612289, 0.607884, 0.601766, 0.597423],
        "iso": [0.626203, 0.623414, 0.621414, 0.618821, 0.611892],
        "task_arithmetic": [0.635073, 0.628367, 0.633560, 0.626660, 0.617968],
        "iso_no_restore": [0.626477, 0.623505, 0.622353, 0.620517, 0.612466],
        "iso_no_mask": [0.628033, 0.625866, 0.623909, 0.621914, 0.612783],
        "retention": {
            "Average": 0.451168,
            "ISO": 0.598259,
            "No restoration": 0.602815,
            "No mask": 0.614588,
            "Task Arithmetic": 0.646826,
        },
        "max_float64_error": 1.0412276856883838e-13,
        "max_float32_error": 1.6112400726146944e-08,
        "no_restore_drift": 1.1544127995306848e-03,
    }
    return (data,)


@app.cell
def _(data, mo):
    colors = {
        "Base": "#172554",
        "Average": "#0891b2",
        "ISO": "#2563eb",
        "Task Arithmetic": "#d97706",
    }
    means = {
        "Base": sum(data["base"]) / 5,
        "Average": sum(data["average"]) / 5,
        "ISO": sum(data["iso"]) / 5,
        "Task Arithmetic": sum(data["task_arithmetic"]) / 5,
    }
    bars = []
    for _i, (name, value) in enumerate(means.items()):
        height = (value - 0.52) / 0.13 * 240
        x = 70 + _i * 145
        bars.append(
            f'<rect x="{x}" y="{330-height:.1f}" width="95" height="{height:.1f}" '
            f'rx="5" fill="{colors[name]}"/>'
            f'<text x="{x+47}" y="{315-height:.1f}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14" font-weight="700">{100*value:.2f}%</text>'
            f'<text x="{x+47}" y="355" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{name}</text>'
        )
    chart = (
        '<svg viewBox="0 0 680 390" style="width:100%;background:white">'
        '<text x="28" y="35" font-family="sans-serif" font-size="22" font-weight="700">'
        'Balanced held-out accuracy</text>'
        '<text x="28" y="60" font-family="sans-serif" font-size="13" fill="#64748b">'
        'Mean over five independent training seeds; higher is better</text>'
        '<line x1="45" y1="330" x2="650" y2="330" stroke="#94a3b8"/>'
        + "".join(bars)
        + "</svg>"
    )
    mo.Html(chart)
    return


@app.cell
def _(mo):
    mo.md(r"""
    The headline plot is the evidence to keep in mind: ISO gains 1.45
    percentage points over a simple average, yet gives up 0.80 points to
    Task Arithmetic. The ordering holds seed by seed.

    ## Experimental substitution

    The paper used unavailable 1.5B/7B generative RL specialists. This
    reproduction instead starts from one public 4.4M-parameter
    `google/bert_uncased_L-2_H-128_A-2` checkpoint and trains matched
    full-parameter specialists on two complementary public GLUE tasks:
    SST-2 sentiment and QNLI entailment. Each specialist sees 4,096 examples
    for 160 steps. Evaluation uses both full validation sets.

    This substitution tests the released merger's mechanics and small-scale
    functional behavior. It does not reproduce the paper's absolute scores
    or its unreleased online optimizer.
    """)
    return


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        options={
            "Balanced accuracy": "accuracy",
            "Gain retention": "retention",
            "Spectrum diagnostics": "spectrum",
        },
        value="accuracy",
        label="Explore the evidence:",
    )
    metric
    return (metric,)


@app.cell
def _(data, metric, mo):
    if metric.value == "accuracy":
        rows = [
            "| Seed | Average | ISO | Task Arithmetic |",
            "|---:|---:|---:|---:|",
        ]
        for _i, seed in enumerate(data["seeds"]):
            rows.append(
                f"| {seed} | {100*data['average'][_i]:.2f}% | "
                f"{100*data['iso'][_i]:.2f}% | "
                f"{100*data['task_arithmetic'][_i]:.2f}% |"
            )
        body = "\n".join(rows)
    elif metric.value == "retention":
        body = "\n".join(
            ["| Method | Mean specialist-gain retention |", "|---|---:|"]
            + [f"| {k} | {100*v:.1f}% |" for k, v in data["retention"].items()]
        )
    else:
        body = (
            "| Diagnostic | Observed | Preregistered tolerance |\n"
            "|---|---:|---:|\n"
            f"| Worst float64 relative error | {data['max_float64_error']:.2e} | 1e-10 |\n"
            f"| Worst float32-checkpoint error | {data['max_float32_error']:.2e} | 1e-5 |\n"
            f"| No-restoration median drift | {data['no_restore_drift']:.2e} | — |"
        )
    mo.md(body)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## How ISO was implemented

    For every two-dimensional parameter, the code computes the base and
    specialist SVDs in float64, aligns singular-vector signs, projects frame
    changes onto the shared Stiefel tangent spaces, masks the trailing 10% of
    modes, solves the paper's unit-retention Gram system, retracts the merged
    frames, and reconstructs with the **base** singular values. One-dimensional
    parameters use task-vector averaging.

    Two controlled ablations change one choice each:

    - **No restoration** uses the mean specialist spectrum with the same
      merged frames.
    - **No mask** retains all modes while still restoring the base spectrum.

    The baselines are matched checkpoint-only operations. Task Arithmetic
    uses λ=1; uniform averaging is equivalent to λ=0.5.

    ## What the evidence says

    - **Mechanistic claim: aligned.** All 85 matrix checks passed. Worst
      relative error was `1.04e-13` in float64 and `1.61e-8` after float32
      checkpoint casting.
    - **Functional claim: partially aligned.** ISO beat averaging on every
      seed, but Task Arithmetic beat ISO on every seed.
    - **Ablation mechanism: not supported here.** No restoration and no mask
      each produced small, consistent improvements over full ISO.

    The likely boundary is the downscaling itself: small supervised
    classifiers can have task-vector geometry unlike generative RLVR
    specialists. A faithful full-scale reproduction still requires matched
    RLVR coding/math checkpoints and generation benchmarks.

    ## Compute and provenance

    All formal evidence ran on OpenResearch Kubernetes using NVIDIA RTX PRO
    6000 Blackwell GPUs: four GPUs per run, 16 peak concurrent, and 0.12
    elapsed wall hours for the fresh attempt. Four exact reruns confirmed
    deterministic reproduction of seeds 11, 22, 33, and 55.

    Public paper: [arXiv 2607.19331](https://arxiv.org/abs/2607.19331) ·
    [Detailed report](https://github.com/alphaXiv/iso-89fd27c5/blob/main/reports/iso-merger-reproduction/report.md) ·
    [Frozen summary data](https://github.com/alphaXiv/iso-89fd27c5/blob/main/results/summary.json)
    """)
    return


if __name__ == "__main__":
    app.run()
