import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from pathlib import Path
from PIL import Image

import hotspice
import thesis_utils

def plot():
    thesis_utils.init_style()
    
    figsize = (thesis_utils.page_width, thesis_utils.page_width*.5)
    fig, axes = plt.subplots(2, 3, figsize=figsize, sharex=True, sharey=True)
    
    S_ASL_values = [20, 25, 30]
    S_ASL_colors = ["#76FFE3", "#0EE667", "#128110"]
    t_values = ["1e3", "7e7"]
    
    switched = [[(1,1), (3,1), (4,5), (4,8), (4,10), (8,9), (9,3), (10,0), (10,7)],
                [(0,4), (0,7), (1,6), (1,9), (3,2), (3,5), (5,8), (6,1), (7,2), (7,8), (8,6), (9,10), (10,6), (10,8)],
                [(1,4), (1,9), (2,2), (2,6), (3,0), (3,4), (3,8), (5,8), (6,4), (7,3), (7,7), (7,9), (8,10), (9,6), (10,9)]]
    
    
    for (i, j), ax in np.ndenumerate(axes):
        ax: plt.Axes
        imagepath = Path(__file__).parent / f"SASI{S_ASL_values[j]}_t{t_values[i]}.png"
        img = np.asarray(Image.open(imagepath).convert('L'))
        ax.imshow(img, cmap="grey", extent=[0,1,0,1])
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(S_ASL_colors[j])
            spine.set_linewidth(3)
        
        ## Circles
        if i == 1:
            factor = 1/13
            for x, y in switched[j]:
                ax.add_artist(Circle(((x+1.5)*factor, 1 - (y+1.5)*factor), factor/2,
                                     edgecolor="yellow", linewidth=1, facecolor="none"))
        
        ## Titles
        fs = plt.rcParams['axes.titlesize']
        if j == 0: # First column
            t = float(t_values[i])
            t_exponent = np.floor(np.log10(t))
            ax.set_ylabel(f"$t = {t/10**t_exponent:.0f}" + r"\times" + f"10^{t_exponent:.0f}" + r"\, \mathrm{s}$", fontsize=fs)
        if i == 0: # First row
            S_ASL = S_ASL_values[j]
            ax.set_title(r"$S_\mathrm{ASI} = " + f"{S_ASL:.0f}" + r"\, \mathrm{nm}$", fontsize=fs)
    
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.1)
    
    hotspice.utils.save_results(figures={"MFM_grid": fig}, timestamped=False, copy_script=False)
    plt.show()


if __name__ == "__main__":
    plot()