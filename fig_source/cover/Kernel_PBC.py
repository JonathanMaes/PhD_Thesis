import hotspice
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp
from matplotlib import patches

import thesis_utils


def plot_kernel_PBC(Lx: int = 32, Ly: int = 16):
    ## Create ASI
    ASI = hotspice.ASI.IP_Pinwheel(a=1e-6, nx=Lx, ny=Ly, PBC=True)
    E_MC: hotspice.DipolarEnergy = ASI.get_energy('dipolar')
    kernel_PBC = E_MC.kernel_unitcell[0]
    x_min, x_max = -Lx+.5, Lx-.5
    y_min, y_max = -Ly+.5, Ly-.5

    ## Create plot
    thesis_utils.init_style()
    d = .5
    extent=(x_min-d, x_max+d, y_min-d, y_max+d)
    figsize = (thesis_utils.page_width, thesis_utils.page_width*(extent[3] - extent[2])/(extent[1] - extent[0]))
    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(extent[:2])  # Set x-axis limits
    ax.set_ylim(extent[2:])  # Set y-axis limits to match aspect ratio
    ax.set_aspect('equal', adjustable='datalim')  # Ensure equal aspect ratio based on data limits
    imshow_kwargs = dict(cmap = plt.get_cmap("bwr"), vmin=-1, vmax=1, extent=extent)
    
    ## Draw magnets
    data = kernel_PBC/xp.max(xp.abs(kernel_PBC)) # Interaction strength in range -1 to 1
    # data = -np.clip(np.log(np.abs(data)), -1, 0)*np.sign(data)
    size = 1.41 # Size of magnets
    def draw_magnet(x, y, angle, **ellipse_kwargs):
        ellipse_kwargs = dict(facecolor="grey", edgecolor="white", alpha=.5, linewidth=1) | ellipse_kwargs
        ax.add_artist(patches.Ellipse((x, y), size, size/2, angle=angle, **ellipse_kwargs))
    for i in np.arange(x_min, x_max, dtype=int):
        for j in np.arange(y_min, y_max, dtype=int):
            if (i+j) % 2 == 0:
                value = (data[j-Ly,i-Lx] + 1)/2 # In range 0 to 1 for cmap
                color = imshow_kwargs["cmap"](value)
                if i==j==0: edgecolor = "black"
                elif abs(i) < 2 and abs(j) < 2: edgecolor = "#EEE"
                else: edgecolor = color
                draw_magnet(i, j, 45 if i%2 else -45, alpha=1, facecolor="white" if i==j==0 else color, edgecolor=edgecolor)

    ## Save
    hotspice.utils.save_results(figures={"Kernel_PBC": fig}, timestamped=False)


if __name__ == "__main__":
    Lx = 30
    plot_kernel_PBC(Lx=Lx, Ly=int(Lx*np.sqrt(2)))
