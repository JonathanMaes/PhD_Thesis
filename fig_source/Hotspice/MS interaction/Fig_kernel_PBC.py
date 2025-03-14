import hotspice
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp
from matplotlib import patches

import thesis_utils


def plot_kernel_PBC(L: int = 16):
    ## Create ASI
    ASI = hotspice.ASI.IP_Pinwheel(a=1e-6, nx=L, ny=L, PBC=False)
    E_MC: hotspice.DipolarEnergy = ASI.get_energy('dipolar')
    kernel_OBC = E_MC.kernel_unitcell[0]
    ASI.PBC = True
    kernel_PBC = E_MC.kernel_unitcell[0]

    ## Create plot
    thesis_utils.init_style()
    figsize = (thesis_utils.page_width, 3)
    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=figsize)
    fig.suptitle(r"Unit cell kernel $\boldsymbol{\mathcal{K}}^\mathrm{(A)}$, with", x=0.47)
    imshow_kwargs = dict(
        cmap = plt.get_cmap("bwr"),
        vmin=-1,
        vmax=1,
        extent=(-L+.5, L-.5, -L+.5, L-.5)
    )
    ticks = [-L+1, 0, L-1]
    
    ## Open BC
    ax1: plt.Axes = axes[0,0]
    v = max(xp.max(xp.abs(kernel_OBC)), xp.max(xp.abs(kernel_PBC)))
    im1 = ax1.imshow(kernel_OBC*0, **imshow_kwargs)
    ax1.set_title("open BC")
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    
    ## Periodic BC
    ax2: plt.Axes = axes[0,1]
    im2 = ax2.imshow(kernel_PBC*0, **imshow_kwargs)
    ax2.set_title("periodic BC")
    ax2.set_xticks(ticks)
    ax2.set_yticks(ticks)
    
    ## Draw magnets
    for i, ax in enumerate(axes.flat):
        data = (kernel_OBC if i==0 else kernel_PBC)/v # Interaction strength in range -1 to 1
        size = 1.41 # Size of magnets
        def draw_magnet(x, y, angle, **ellipse_kwargs):
            ellipse_kwargs = dict(facecolor="grey", edgecolor="white", alpha=.5, linewidth=1) | ellipse_kwargs
            ax.add_artist(patches.Ellipse((x, y), size, size/2, angle=angle, **ellipse_kwargs))
        for i in np.arange(imshow_kwargs["extent"][0], imshow_kwargs["extent"][1], dtype=int):
            for j in np.arange(imshow_kwargs["extent"][2], imshow_kwargs["extent"][3], dtype=int):
                if (i+j) % 2 == 0:
                    value = (data[i-L,j-L] + 1)/2 # In range 0 to 1 for cmap
                    color = imshow_kwargs["cmap"](value)
                    if i==j==0: edgecolor = "black"
                    elif abs(i) < 2 and abs(j) < 2: edgecolor = "#EEE"
                    else: edgecolor = color
                    draw_magnet(i, j, 45 if i%2 else -45, alpha=1, facecolor="white" if i==j==0 else color, edgecolor=edgecolor)
    
    ## Add colorbar axis on the right
    X, Y, bottom = 0.89, 0.75, 0.08 # X: left edge of colorbar, Y: height of all axes, bottom: bottom of all axes
    cbar_ax: plt.Axes = fig.add_axes([X, bottom, 0.02, Y])  # [left, bottom, width, height]
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_ticks([-1, 0, 1])
    cbar.set_label("Magnetostatic interaction [a.u.]", rotation=-90, labelpad=12)  # Add a label to the colorbar

    ## Adjust layout to avoid overlapping elements
    fig.tight_layout(rect=[0, 0, X, 1])  # Leave space on the right for the colorbar
    fig.subplots_adjust(bottom=bottom, top=Y+bottom)

    ## Save
    hotspice.utils.save_results(figures={"Kernel_PBC": fig}, timestamped=False)


if __name__ == "__main__":
    plot_kernel_PBC()
