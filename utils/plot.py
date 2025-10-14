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

def plot_score_hist_with_thresholds(y_true, y_score, y_score_val=None, xscale='linear', figsize=(5, 3), bins=None, stat="count", title=None, ax=None):
  """
  Plot a histogram of scores split by class and draw thresholds at mean and mean+std.
  Compute TPR/FPR at those thresholds on (y_true, y_score).

  Parameters
  ----------
  y_true : array-like of shape (n_samples,)
  y_score : array-like of shape (n_samples,)
      Scores/probabilities on the test set (used for both plotting and metrics).
      Binary labels in {0,1} for the test set.
  y_score_val : array-like of shape (m_samples,), optional
      Scores on a validation set to compute thresholds; if None, use y_score.
  figsize : (w, h)
      Figure size for the plot.
  bins : int, optional
      Number of histogram bins (pass None to let seaborn decide).
  stat : {"count", "density", "probability", ...}
      Seaborn histplot `stat` parameter.
  title : str, optional
      Plot title.
  ax : matplotlib.axes.Axes, optional
      Axes to plot on; if None, a new figure/axes is created.

  Returns
  -------
  metrics : dict
    {
      "thr_avg": float,
      "thr_avg_std": float,
      "tpr_thr_avg": float,
      "fpr_thr_avg": float,
      "tpr_thr_avg_std": float,
      "fpr_thr_avg_std": float,
    }
  """
  # Convert inputs
  y_true = np.asarray(y_true).astype(int)
  y_score = np.asarray(y_score).astype(float)
  if y_score_val is None:
      y_score_val = y_score
  else:
      y_score_val = np.asarray(y_score_val).astype(float)

  # Basic validations
  if y_score.shape[0] != y_true.shape[0]:
    raise ValueError("y_score and y_true must have the same length.")
  uniq = np.unique(y_true)
  if not np.array_equal(np.sort(uniq), np.array([0, 1])) and not np.array_equal(uniq, np.array([0])) and not np.array_equal(uniq, np.array([1])):
    raise ValueError("y_true must contain binary labels {0,1}.")
  if (y_true == 1).sum() == 0 or (y_true == 0).sum() == 0:
    # Still allow plotting but metrics would divide by zero; handle gracefully.
    eps = 1e-12
  else:
    eps = 0.0

  # Thresholds from validation (or test if val not provided)
  thr_avg = float(np.mean(y_score_val))
  thr_avg_std = float(thr_avg + np.std(y_score_val))

  # Helper to compute TPR/FPR at threshold
  def tpr_fpr_at(thr):
    preds = (y_score >= thr).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    p = np.sum(y_true == 1)
    n = np.sum(y_true == 0)
    tpr = tp / (p + (eps if p == 0 else 0))
    fpr = fp / (n + (eps if n == 0 else 0))
    return float(tpr), float(fpr)

  tpr_thr_avg, fpr_thr_avg = tpr_fpr_at(thr_avg)
  tpr_thr_avg_std, fpr_thr_avg_std = tpr_fpr_at(thr_avg_std)

  # Prepare DataFrame for plotting
  hist_data = pd.DataFrame({
    "score": y_score,
    "class": y_true
  })

  # Plot
  created_fig = False
  if ax is None:
    fig, ax = plt.subplots(figsize=figsize)
    created_fig = True

  sns.histplot(hist_data, x="score", hue="class", palette={0: 'forestgreen', 1: 'firebrick'}, stat=stat, bins=bins, ax=ax)

  # Place threshold lines/text; set a dynamic y position based on current ylim
  ax.axvline(x=thr_avg, color="black", linestyle="--", label="avg.")
  ax.axvline(x=thr_avg_std, color="black", linestyle=":", label="avg.+std.")
  ymax = ax.get_ylim()[1]
  y_text = 0.92 * ymax
  ax.text(x=thr_avg, y=y_text, s="avg.", color="black", ha="left", va="top")
  ax.text(x=thr_avg_std, y=y_text*0.88, s="avg.+std.", color="black", ha="left", va="top")

  ax.set_xscale(xscale)

  if title:
    ax.set_title(title)
  ax.set_xlabel("Score")
  ax.set_ylabel(stat.capitalize())
  ax.legend(title='Thresholds', loc="best")
  if created_fig:
    plt.tight_layout()
    plt.show()
    plt.close()

  metrics = {
    "thr_avg": thr_avg,
    "thr_avg_std": thr_avg_std,
    "tpr_thr_avg": tpr_thr_avg,
    "fpr_thr_avg": fpr_thr_avg,
    "tpr_thr_avg_std": tpr_thr_avg_std,
    "fpr_thr_avg_std": fpr_thr_avg_std,
  }
  return metrics
