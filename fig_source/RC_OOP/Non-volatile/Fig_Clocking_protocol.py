import hotspice
import math
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp
from matplotlib import lines, patches

import thesis_utils


def main_plot():
    thesis_utils.init_style()
    
    ## Create a figure that we can draw on everywhere
    figsize = (thesis_utils.page_width, 2)
    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim([0, 1])  # Set x-axis limits
    ax.set_ylim([0, figsize[1]/figsize[0]])  # Set y-axis limits to match aspect ratio
    ax.set_aspect('equal', adjustable='datalim')  # Ensure equal aspect ratio based on data limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
    print(f"Figure coordinates:\n\tx {x_min:.3f} --> {x_max:.3f}\n\ty {y_min:.3f} --> {y_max:.3f}")
    ax.axis('off')  # Disable axis lines and labels
    ax.patch.set_alpha(0)  # Transparent background
    
    ## Add the parts
    pad = 0.05
    y0 = pad/2 # Bottom of field plots
    w = ((x_max - x_min) - 4*pad)/4 # There are in total 4 figures with 4 total <pad> paddings.
    # h = w*(y_max - y_min)
    
    for bit in range(2):
        x_center_bit = 0.5 + (w + 1.3*pad)*(2*bit-1)
        ax.text(x_center_bit, y0+w+1.4*pad, f"Input cycle ${'B' if bit else 'A'}$\n(bit {bit:d})", ha="center", va="baseline", fontdict=dict(fontsize=thesis_utils.fs_large + 1))
        ax.arrow(x_center_bit - pad*.4, y0+w/2, pad*.8, 0, head_width=0.4*pad, head_length=0.3*pad, width=0.2*pad, fc="black", ec="None", length_includes_head=True)
        for step in range(2):
            x_center_step = x_center_bit + (w/2 + pad/2)*(2*step-1)
            ax.text(x_center_step, y0+w+pad/2, f"{'Second' if step else 'First'}\nsubstep", ha="center", va="baseline", fontdict=dict(fontsize=thesis_utils.fs_large))
            x0 = x_center_step - w/2
            plot_input_fields(ax, x0, y0, w, step=step, bit=bit)
    draw_line(ax, 0.5, 0, 0.5, y0 + w + 2*pad, lw=3)
    
    ## Save the result
    hotspice.utils.save_results(figures={f"Clocking_protocol": fig}, timestamped=False)

def plot_input_fields(ax: plt.Figure, x, y, w, step: int = 0, bit: int = 0, n: int = 5):
    l = w/n # Lattice spacing
    r = l*0.4 # Radius of vector symbols
    for ix in range(n):
        for iy in range(n):
            if (ix + iy) % 2 == step: continue
            up = ((bit + step) % 2) == 1
            draw_Zvector(ax, up=up, x=x+l*(ix+.5), y=y+l*(iy+.5), r=r)

    linecolor = 'gray'
    for ix in range(1,n):
        x_here = x + l*ix
        draw_line(ax, x1=x_here, y1=y, x2=x_here, y2=y+w, color=linecolor, lw=.5)
    for iy in range(1,n):
        y_here = y + l*iy
        draw_line(ax, x1=x, y1=y_here, x2=x+w, y2=y_here, color=linecolor, lw=.5)


## Helper functions
def draw_Zvector(ax: plt.Axes, up: bool, x: float, y: float, r: float, lw: float = 1, edgecolor='k', facecolor=None):
    """ If 'up' is True, then a dot is drawn, otherwise a cross, in line with the +z/-z symbols for vectors. """
    ax.add_patch(patches.Circle((x, y), r, fill=bool(facecolor), lw=lw, edgecolor=edgecolor, facecolor=facecolor))
    if up: # Draw cross
        xarr = [x - r/np.sqrt(2), x + r/np.sqrt(2)]
        yarr = [y - r/np.sqrt(2), y + r/np.sqrt(2)]
        ax.plot(xarr, yarr, lw=lw, color=edgecolor, solid_capstyle="butt")
        ax.plot(xarr, yarr[::-1], lw=lw, color=edgecolor, solid_capstyle="butt")
    else: # Draw dot
        ax.scatter([x], [y], s=((lw*200)/ax.get_figure().dpi)**2, edgecolors="none", color=edgecolor)

def draw_line(ax: plt.Axes, x1, y1, x2, y2, color='k', lw=1, style="-"):
    ax.add_artist(lines.Line2D([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style, solid_capstyle='projecting'))

if __name__ == "__main__":
    main_plot()
