import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import statistics

    return mo, statistics


@app.cell
def _(mo):
    mo.md(r"""
    # ISO-Merger: exact spectra, mixed functional retention

    When specialists learn different skills from one base model, a merge can
    erase their gains. ISO-Merger proposes combining changes to the singular
    directions of each weight matrix while restoring the base singular
    values. This notebook embeds the fresh five-seed Kubernetes evidence, so
    opening it never requires rerunning training.

    **Verdict: partially reproduced.** Spectrum preservation was exact within
    tolerance and ISO beat uniform averaging, but Task Arithmetic was better
    on the held-out mixed score. Removing restoration or the trailing-mode
    mask was also slightly beneficial.
    """)
    return


@app.cell
def _():
    rows = [
        {"seed": 11, "base": 0.5455445, "average": 0.6099168, "iso": 0.6262027, "task_arithmetic": 0.6350728, "no_restore": 0.6264773, "no_mask": 0.6280332, "ret_average": 0.4836773, "ret_iso": 0.6316024, "ret_task_arithmetic": 0.6878808, "spectrum64": 9.8029e-14, "spectrum32": 1.1548e-8},
        {"seed": 22, "base": 0.5455445, "average": 0.6122885, "iso": 0.6234140, "task_arithmetic": 0.6283672, "no_restore": 0.6235055, "no_mask": 0.6258663, "ret_average": 0.4690296, "ret_iso": 0.5838855, "ret_task_arithmetic": 0.6113199, "spectrum64": 7.6933e-14, "spectrum32": 5.4589e-9},
        {"seed": 33, "base": 0.5455445, "average": 0.6078844, "iso": 0.6214137, "task_arithmetic": 0.6335599, "no_restore": 0.6223532, "no_mask": 0.6239092, "ret_average": 0.4462943, "ret_iso": 0.5825941, "ret_task_arithmetic": 0.6586829, "spectrum64": 8.9069e-14, "spectrum32": 1.2840e-8},
        {"seed": 44, "base": 0.5455445, "average": 0.6017656, "iso": 0.6188213, "task_arithmetic": 0.6266604, "no_restore": 0.6205173, "no_mask": 0.6219144, "ret_average": 0.4216178, "ret_iso": 0.5963082, "ret_task_arithmetic": 0.6434775, "spectrum64": 9.7215e-14, "spectrum32": 5.7710e-9},
        {"seed": 55, "base": 0.5455445, "average": 0.5974233, "iso": 0.6118921, "task_arithmetic": 0.6179679, "no_restore": 0.6124655, "no_mask": 0.6127831, "ret_average": 0.4352234, "ret_iso": 0.5969072, "ret_task_arithmetic": 0.6327710, "spectrum64": 1.0412e-13, "spectrum32": 1.6112e-8},
    ]
    labels = {
        "base": "Shared base",
        "average": "Weight average",
        "iso": "ISO-Merger",
        "task_arithmetic": "Task Arithmetic",
        "no_restore": "ISO: no restoration",
        "no_mask": "ISO: no mask",
    }
    return labels, rows


@app.function
def bar_chart_svg(items, lower, upper):
    colors = ["#c8cfda", "#8b95a7", "#2575d8", "#e58a2b", "#73a6df", "#1e5da8"]
    width, height = 780, 390
    baseline, top = 315, 65
    gap = 650 / len(items)
    bars = []
    for index, (name, value) in enumerate(items):
        x = 92 + index * gap
        y = baseline - (value - lower) / (upper - lower) * (baseline - top)
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{gap * 0.62:.1f}" '
            f'height="{baseline-y:.1f}" rx="4" fill="{colors[index]}"/>'
            f'<text x="{x + gap * 0.31:.1f}" y="{y-10:.1f}" '
            'text-anchor="middle" font-family="sans-serif" font-size="15" '
            f'font-weight="700">{value*100:.2f}%</text>'
            f'<text x="{x + gap * 0.31:.1f}" y="345" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{name}</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        'xmlns="http://www.w3.org/2000/svg">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<line x1="75" x2="750" y1="315" y2="315" stroke="#ccd3dd"/>'
        + "".join(bars)
        + "</svg>"
    )


@app.cell
def _(labels, mo, rows, statistics):
    headline_keys = ["base", "average", "iso", "task_arithmetic"]
    headline_items = [
        (labels[key], statistics.mean(row[key] for row in rows))
        for key in headline_keys
    ]
    mo.vstack(
        [
            mo.md("## Primary result: balanced held-out accuracy"),
            mo.Html(bar_chart_svg(headline_items, 0.54, 0.64)),
            mo.md(
                r"""
                ISO gained **1.45 percentage points** over uniform averaging.
                Task Arithmetic was **0.80 points above ISO**, and this ordering
                held on every independent seed.
                """
            ),
        ]
    )
    return


@app.cell
def _(mo, rows, statistics):
    summary = []
    for key, label in [
        ("average", "Weight average"),
        ("iso", "ISO-Merger"),
        ("task_arithmetic", "Task Arithmetic"),
        ("no_restore", "ISO: no restoration"),
        ("no_mask", "ISO: no mask"),
    ]:
        _values = [row[key] for row in rows]
        summary.append(
            {
                "method": label,
                "mixed accuracy": f"{statistics.mean(_values):.4f}",
                "sample SD": f"{statistics.stdev(_values):.4f}",
            }
        )
    mo.vstack(
        [
            mo.md(
                r"""
                ## Why gain retention is stricter than accuracy

                For task \(t\), define retention as
                \((score_{merge,t}-score_{base,t}) /
                (score_{specialist,t}-score_{base,t})\).
                It asks what fraction of each task's acquired improvement
                survived, rather than rewarding capability already in the base.
                Across tasks and seeds: averaging retained **45.12%**, ISO
                **59.83%**, and Task Arithmetic **64.68%**.
                """
            ),
            mo.ui.table(summary, selection=None),
        ]
    )
    return


@app.cell
def _(mo, rows):
    mechanism = [
        {
            "seed": row["seed"],
            "no restoration − ISO (pp)": round((row["no_restore"] - row["iso"]) * 100, 4),
            "no mask − ISO (pp)": round((row["no_mask"] - row["iso"]) * 100, 4),
            "worst float64 spectrum error": f"{row['spectrum64']:.2e}",
            "worst float32 spectrum error": f"{row['spectrum32']:.2e}",
        }
        for row in rows
    ]
    mo.vstack(
        [
            mo.md(
                r"""
                ## Mechanism diagnostics

                All 85 matrix/seed checks passed the preregistered relative
                tolerances: \(10^{-10}\) for float64 reconstruction and
                \(10^{-5}\) after float32 casting. Yet both ablations were
                positive on every seed. The mechanism was implemented and
                changed spectra as intended, but its proposed functional benefit
                was not isolated here.
                """
            ),
            mo.ui.table(mechanism, selection=None),
        ]
    )
    return


@app.cell
def _(mo):
    metric = mo.ui.dropdown(
        options={
            "Balanced accuracy": "accuracy",
            "Specialist-gain retention": "retention",
        },
        value="accuracy",
        label="Inspect each seed:",
    )
    metric
    return (metric,)


@app.cell
def _(labels, metric, mo, rows):
    if metric.value == "accuracy":
        keys = ["average", "iso", "task_arithmetic"]
        table_rows = [
            {
                "seed": row["seed"],
                **{labels[key]: f"{row[key] * 100:.2f}%" for key in keys},
            }
            for row in rows
        ]
    else:
        table_rows = [
            {
                "seed": row["seed"],
                "Weight average": f"{row['ret_average'] * 100:.2f}%",
                "ISO-Merger": f"{row['ret_iso'] * 100:.2f}%",
                "Task Arithmetic": f"{row['ret_task_arithmetic'] * 100:.2f}%",
            }
            for row in rows
        ]
    mo.ui.table(table_rows, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Boundaries and provenance

    The public substitution is a 4.4M-parameter BERT binary classifier with
    matched 160-step full-weight specialists on GLUE SST-2 and QNLI. It is
    not a reproduction of the paper's 1.5B/7B RLVR benchmarks or its
    unreleased online optimizer.

    All evidence came from fresh OpenResearch **Kubernetes** jobs on
    **NVIDIA RTX PRO 6000 Blackwell** GPUs. Peak concurrency was **16 GPUs**;
    the launch-to-final-evidence campaign took **0.10 wall hours**, and
    successful four-GPU jobs took 68–83 seconds. Exact repeats of seeds
    11/22/33/55 returned identical measurements.

    The implementation, detailed report, figures, compact CSV, and formal
    verdict are all in the public
    [alphaXiv/iso-89fd27c5](https://github.com/alphaXiv/iso-89fd27c5)
    repository.
    """)
    return


if __name__ == "__main__":
    app.run()
