import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import pandas as pd
import seaborn as sns


def plot_roc(y_true, y_score, savefig=None, show=True, dpi=300, colors=('firebrick', 'black')):
  """
  Plots the ROC curve.

  Parameters:
  - y_true: array-like, ground truth binary labels.
  - y_score: array-like, predicted scores or probabilities.
  - savefig: str or None, base filename to save the figure (without extension).
  - show: bool, whether to display the plot (default: True).
  - dpi: int, resolution for saving the figure (default: 300).
  - colors: tuple, (ROC curve color, diagonal line color).
  """

  fpr, tpr, _ = roc_curve(y_true, y_score)

  fig, ax = plt.subplots(figsize=(5, 3))
  ax.plot(fpr, tpr, color=colors[0], label="ROC Curve", zorder=4, clip_on=False)
  ax.plot([0, 1], [0, 1], color=colors[1], linestyle='--', label="Random Guess", zorder=3)

  ax.set_xlim([0, 1])
  ax.set_ylim([0, 1])
  ax.set_xlabel('False Positive Rate (FPR)')
  ax.set_ylabel('True Positive Rate (TPR)')
  ax.legend()
  plt.tight_layout()

  # Save figure in multiple formats if savefig is provided
  if savefig:
    for ext in ['pdf', 'png']:
      plt.savefig(f"{savefig}.{ext}", dpi=dpi, bbox_inches='tight')

  # Show the plot if enabled
  if show:
    plt.show()

  plt.close(fig)  # Prevents memory leaks in long-running scripts

def plot_score_hist_with_thresholds(
    y_true, y_score, y_score_val=None, xscale='linear',
    figsize=(5, 3), bins=None, stat="count", title=None, ax=None
):
    """
    Plot score distributions by class and mark thresholds at mean and mean+std.
    Also compute TPR/FPR at those thresholds on (y_true, y_score).

    Parameters
    ----------
    y_true : array-like
        Binary ground truth labels (0/1) for the test set.
    y_score : array-like
        Corresponding model scores/probabilities for y_true.
    y_score_val : array-like, optional
        Scores from a validation set to compute thresholds; defaults to y_score.
    xscale : {"linear", "log"}, default="linear"
        X-axis scale.
    figsize : tuple, default=(5, 3)
        Figure size.
    bins : int or array-like, optional
        Number of histogram bins (int) or explicit bin edges (array). If None, bins are auto-generated.
    stat : str, default="count"
        Seaborn histplot statistic ("count", "density", etc.).
    title : str, optional
        Plot title.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw on; creates a new one if None.

    Returns
    -------
    metrics : dict
        Contains thresholds and corresponding TPR/FPR values:
        {
            "thr_avg", "thr_avg_std",
            "tpr_thr_avg", "fpr_thr_avg",
            "tpr_thr_avg_std", "fpr_thr_avg_std"
        }.
    """
    # Input conversion and validation
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_score_val = y_score if y_score_val is None else np.asarray(y_score_val).astype(float)

    if y_score.shape[0] != y_true.shape[0]:
        raise ValueError("y_score and y_true must have the same length.")
    uniq = np.unique(y_true)
    if not np.all(np.isin(uniq, [0, 1])):
        raise ValueError("y_true must contain binary labels {0,1}.")
    eps = 1e-12 if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0 else 0.0

    # Compute thresholds
    thr_avg = float(np.mean(y_score_val))
    thr_avg_std = float(thr_avg + np.std(y_score_val))

    # Helper: TPR/FPR at a threshold
    def tpr_fpr_at(thr):
        preds = (y_score >= thr).astype(int)
        tp = np.sum((preds == 1) & (y_true == 1))
        fp = np.sum((preds == 1) & (y_true == 0))
        p, n = np.sum(y_true == 1), np.sum(y_true == 0)
        return tp / (p + eps), fp / (n + eps)

    tpr_thr_avg, fpr_thr_avg = tpr_fpr_at(thr_avg)
    tpr_thr_avg_std, fpr_thr_avg_std = tpr_fpr_at(thr_avg_std)

    # Prepare DataFrame for plotting
    hist_data = pd.DataFrame({"score": y_score, "class": y_true})

    # Figure/axes setup
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Automatic binning for log scale
    if bins is None and xscale == 'log':
        min_score = max(y_score.min(), 1e-12)  # avoid log(0)
        max_score = y_score.max()
        n_bins = 50  # default number of bins
        bins = np.logspace(np.log10(min_score), np.log10(max_score), n_bins)

    sns.histplot(
        hist_data, x="score", hue="class",
        palette={0: 'forestgreen', 1: 'firebrick'},
        stat=stat, bins=bins, ax=ax
    )

    # Threshold lines and labels
    ax.axvline(thr_avg, color="black", linestyle="--", label="avg.")
    ax.axvline(thr_avg_std, color="black", linestyle=":", label="avg.+std.")
    ymax = ax.get_ylim()[1]
    ax.text(thr_avg, 0.92 * ymax, "avg.", color="black", ha="left", va="top")
    ax.text(thr_avg_std, 0.81 * ymax, "avg.+std.", color="black", ha="left", va="top")

    # Styling
    ax.set_xscale(xscale)
    if title: ax.set_title(title)
    ax.set_xlabel("Score")
    ax.set_ylabel(stat.capitalize())
    ax.legend(title='Thresholds', loc="best")

    if created_fig:
        plt.tight_layout()
        plt.show()
        plt.close()

    return {
        "thr_avg": thr_avg, "thr_avg_std": thr_avg_std,
        "tpr_thr_avg": tpr_thr_avg, "fpr_thr_avg": fpr_thr_avg,
        "tpr_thr_avg_std": tpr_thr_avg_std, "fpr_thr_avg_std": fpr_thr_avg_std
    }
