import matplotlib
matplotlib.use("Agg")
""" NOTE: if "DejaVu Sans Display" can not be found, then math mode will not be italic.
          To fix this, go to the C:/Users/<me>/.matplotlib/fontlist-v<version>.json file
          and add the following after the "DejaVuSans-BoldOblique.ttf" entry:
    {
      "fname": "fonts\\ttf\\DejaVuSansDisplay.ttf",
      "name": "DejaVu Sans Display",
      "style": "normal",
      "variant": "normal",
      "weight": 400,
      "stretch": "normal",
      "size": "scalable",
      "__class__": "FontEntry"
    },
"""

import hotspice
import inspect
import traceback

import matplotlib.pyplot as plt
import numpy as np

from matplotlib import cycler
from matplotlib.legend import Legend
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path


page_width = 6.148*1.1 # inches

class Cycle(object):
    def __init__(self, data): self.data = data
    def __getitem__(self, i): return self.data[i % len(self.data)]
    def __repr__(self): return self.data.__repr__()
marker_cycle = Cycle(['o', 's', 'D', 'P', 'X', 'p', '*', '^']) # Circle, square, diamond, plus, cross, pentagon, star, triangle up (and repeat enough times)

fs_small = 9
fs_medium = fs_small + 1
fs_large = fs_medium + 1

def init_style(style=None):
    hotspice.plottools.init_style(small=fs_small, medium=fs_medium, large=fs_large, style=style if style is not None else "default")
    if style is None: plt.rcParams['axes.prop_cycle'] = cycler(color=["dodgerblue", "tab:red", "tab:orange", "m", "c"])
    plt.rcParams["legend.fontsize"] = fs_medium
    # plt.rcParams["pdf.use14corefonts"] = True # trigger core fonts for PDF backend
    # plt.rcParams["ps.useafm"] = True # trigger core fonts for PS backend
    # plt.rcParams["font.family"] = "Arial"
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    # rc('font',**{'family':'sans-serif','sans-serif':['Arial']})
    # rc('font',**{'family':'serif','serif':['Times']})
    # rc('text', usetex=True)

def replot_all(plot_function, subdir: str = None, **plot_kwargs):
    """ Replots all the timestamp dirs """
    script = Path(inspect.stack()[1].filename) # The caller script, i.e. the one where __name__ == "__main__"
    outdir = script.parent / (script.stem + '.out')
    if subdir: outdir = outdir / subdir
    
    def replot_dir(data_dir: Path):
        if not data_dir.is_dir(): return
        try:
            plot_function(data_dir, **plot_kwargs)
            print(f"Re-plotted {data_dir}")
        except FileNotFoundError:
            print(f"Skipping {data_dir}")
        except Exception:
            print(traceback.format_exc())
    
    replot_dir(outdir)
    for data_dir in outdir.iterdir():
        replot_dir(data_dir)
        

def label_ax(ax: plt.Axes, i: int = None, form: str = "(%s)", offset: tuple[float, float] = (0,0), fontsize: float = 11, axis_units: bool = True, **kwargs):
    """ To add a label to `ax`, pass either `i` or `form` (or both).
        If only `i` is passed, the label becomes "(a)", with the letter corresponding to index `i` (0=a, 1=b ...)
        If only `form` is passed, it is used as the complete label.
        If both `i` and `form` are passed, then the letter s corresponding to `i` is formatted using `form % s`.
        Examples:
            label_ax(ax, 1) --> "(b)"
            label_ax(ax, form="Some text.") --> "Some text."
            label_ax(ax, 3, "[%s]") --> "[d]"
        
        @param `ax` [plt.Axes]: The axis object for which the label should be drawn.
        @param `i` [int] (None): Index of the axis, which gets translated to a letter.
            If None, then `form` is assumed to be the complete string.
        @param `form` [str] ("(%s)"): The format used to represent the label with index `i`.
        @param `offset` [tuple(2)]: Tuple of two floats, determining x and y offset (in axis units).
        @param `fontsize` [float] (12): Font size of the label (default 12pt).
        Additional `**kwargs` get passed to the `ax.text()` call.
    """
    if isinstance(i, int):
        s = 'abcdefghijklmnopqrstuvwxyz'[i]
        text = form % s
    kwargs = dict(ha='left', va='bottom', color='k', weight='bold', fontfamily='DejaVu Sans') | kwargs
    t = ax.text(0 + offset[0], int(bool(axis_units)) + offset[1], text, fontsize=fontsize,
                bbox=dict(boxstyle='square,pad=3', facecolor='none', edgecolor='none'),
                transform=ax.transAxes if axis_units else ax.transData, zorder=1000, **kwargs)

def get_last_outdir(subdir: str = None):
    script = Path(inspect.stack()[1].filename) # The caller script, i.e. the one where __name__ == "__main__"
    outdir_all = script.parent / (script.stem + '.out')
    if subdir is not None: outdir_all /= subdir
    timestamped_dirs = [d.absolute() for d in outdir_all.iterdir() if d.is_dir() and d.stem.isnumeric()]
    if len(timestamped_dirs) == 0:
        raise FileNotFoundError("No automatically generated output directory could be found.")
    else:
        return sorted(timestamped_dirs)[-1]

def move_legend(leg: Legend, ax: Axes|Figure, dx: float = 0, dy: float = 0):
    if isinstance(ax, Axes):     trans = ax.transAxes
    elif isinstance(ax, Figure): trans = ax.transFigure
    else: raise ValueError("<ax> must be an Axes or Figure object.")
    bb = leg.get_bbox_to_anchor().transformed(trans.inverted())

    # Change to location of the legend.
    bb.x0 += dx
    bb.x1 += dx
    bb.y0 += dy
    bb.y1 += dy
    leg.set_bbox_to_anchor(bb, transform=trans)

# def std_of_correlated_series(series):
#     series = hotspice.utils.asnumpy(series)
#     N = series.size
#     mean = np.mean(series)
#     var = np.var(series)
    
#     # Determine autocorrelation time
#     mean_Aiplust_Ai = np.correlate(series, series, mode='full')[-N+1:]/np.arange(N-1, 0, -1)
#     autocorr_func = np.correlate(series - mean, series - mean, mode="full")[-N+1:]/(var * np.arange(N-1, 0, -1))
#     autocorr_time = 1 + 2*np.sum(autocorr_func)

#     # Example usage
#     series = np.random.randn(1000)
#     autocorr_time = autocorrelation_time(series)
#     print("Autocorrelation Time:", autocorr_time)

def std_of_correlated_series(series):
    """ WARN: This function is likely incorrect. """
    n = len(series)
    mean = np.mean(series)
    variance = np.var(series)
    
    if variance == 0:
        raise ValueError("Variance of the series is zero, cannot compute autocorrelation time.")
    
    autocorr_func = np.correlate(series - mean, series - mean, mode='full') / (variance * n)
    autocorr_func = autocorr_func[n-1:]
    
    positive_autocorr = autocorr_func[autocorr_func > 0]
    
    if len(positive_autocorr) == 0:
        raise ValueError("No positive values in autocorrelation function, check your data.")
    
    autocorr_time = 1 + 2 * np.sum(positive_autocorr[1:])
    
    if autocorr_time < 0:
        raise ValueError("Calculated autocorrelation time is negative, check your data and calculations.")
    
    N = len(series)
    N_eff = N / (2 * autocorr_time)
    return np.sqrt(variance / N_eff)
