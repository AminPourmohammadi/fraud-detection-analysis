"""Drift monitoring with the Population Stability Index (PSI).

PSI compares how a column is spread across bins today with how it was
spread in the training data. 0 means identical; above 0.25 is a big shift.
"""

import numpy as np
import pandas as pd

N_BINS = 10
SMALLEST_SHARE = 0.0001   # avoids log(0) when a bin is empty
WATCH_LEVEL = 0.10
ALERT_LEVEL = 0.25


def make_edges(reference_values, n_bins=N_BINS):
    """Bin edges at the training data's deciles, open at both ends."""
    cut_points = np.linspace(0, 1, n_bins + 1)[1:-1]
    inner = np.unique(np.quantile(reference_values, cut_points))
    edges = np.concatenate([[-np.inf], inner, [np.inf]])
    return edges


def share_per_bin(values, edges):
    """Fraction of values that fall in each bin."""
    counts, _ = np.histogram(values, bins=edges)
    shares = counts / max(len(values), 1)
    return np.clip(shares, SMALLEST_SHARE, None)


def psi(reference_share, new_share):
    """Sum over bins of (new - old) * log(new / old)."""
    reference_share = np.asarray(reference_share)
    new_share = np.asarray(new_share)
    ratio = new_share / reference_share
    return float(np.sum((new_share - reference_share) * np.log(ratio)))


def build_reference(df, columns):
    """Store edges and shares for each column, taken from training data."""
    reference = {}
    for col in columns:
        edges = make_edges(df[col].values)
        shares = share_per_bin(df[col].values, edges)
        # Store only the inner edges: JSON has no infinity.
        reference[col] = {"inner_edges": edges[1:-1].tolist(),
                          "share": shares.tolist()}
    return reference


def status_for(value):
    if value >= ALERT_LEVEL:
        return "alert"
    if value >= WATCH_LEVEL:
        return "watch"
    return "ok"


def drift_report(reference, df):
    """One row per column: PSI of df against the stored reference."""
    rows = []
    for col in reference:
        inner = reference[col]["inner_edges"]
        edges = np.concatenate([[-np.inf], inner, [np.inf]])
        new_share = share_per_bin(df[col].values, edges)
        value = psi(reference[col]["share"], new_share)
        rows.append({"column": col, "psi": round(value, 4),
                     "status": status_for(value)})
    report = pd.DataFrame(rows)
    return report.sort_values("psi", ascending=False).reset_index(drop=True)
