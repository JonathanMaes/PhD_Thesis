import hotspice
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp
from matplotlib import lines, patches

import thesis_utils


def main_plot():
    thesis_utils.init_style()
    
    ## Create a figure that we can draw on everywhere
    figsize = (thesis_utils.page_width/2, 2)
    fig = plt.figure(figsize=figsize)
    ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim([0, 1])  # Set x-axis limits
    ax.set_ylim([0, figsize[1]/figsize[0]])  # Set y-axis limits to match aspect ratio
    ax.set_aspect('equal', adjustable='datalim')  # Ensure equal aspect ratio based on data limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    print(f"Figure coordinates:\n\tx {x_min:.3f} --> {x_max:.3f}\n\ty {y_min:.3f} --> {y_max:.3f}")
    ax.axis('off')  # Disable axis lines and labels
    ax.patch.set_alpha(0)  # Transparent background
    
    w = 0.9
    nx, ny = 21, 7
    plot_ASI(ax, x_max/2, y_max/2, w, nx=nx, ny=ny, magnet_enlargement=2, ASI_type=hotspice.ASI.IP_Kagome, lw=1, colors=["C0", "C1", "C2", "C1", "C0", "C2"])
        
    ## Save the result
    hotspice.utils.save_results(figures={f"Unitcells": fig}, timestamped=False)


def plot_ASI(ax: plt.Figure, x, y, w, nx=5, ny=None, lw=1, magnet_enlargement=1, ASI_type: type[hotspice.Magnets] = hotspice.ASI.IP_Pinwheel, colors: dict|str = None):
    ## Argument processing
    if ny is None: ny = nx
    if colors is None: colors = [] # List of colors to use for unit cell magnets
    d = w/nx # Size of one cell
    
    ## Create ASI
    ASI = ASI_type(a=1, nx=nx, ny=ny)
    E_MC: hotspice.energies.DipolarEnergy = ASI.get_energy('dipolar')
    ASI_x = ASI.x / max(ASI.x_max, ASI.y_max)
    ASI_y = ASI.y / max(ASI.x_max, ASI.y_max)
    ASI_x_central, ASI_y_central = np.zeros(ASI.nx + 1), np.zeros(ASI.ny + 1)
    for i in range(ASI.nx + 1):
        if i == 0: ASI_x_central[i] = -ASI_x[1]/2
        elif i == ASI.nx: ASI_x_central[i] = ASI_x[-1] + (ASI_x[-1] - ASI_x[-2])/2
        else: ASI_x_central[i] = (ASI_x[i] + ASI_x[i-1])/2
    for i in range(ASI.ny + 1):
        if i == 0: ASI_y_central[i] = -ASI_y[1]/2
        elif i == ASI.ny: ASI_y_central[i] = ASI_y[-1] + (ASI_y[-1] - ASI_y[-2])/2
        else: ASI_y_central[i] = (ASI_y[i] + ASI_y[i-1])/2
    
    ## Move (x,y) to bottom left corner
    x = x - w/2
    y = y - (w/ASI.x_max*ASI.y_max)/2
    
    ## Draw magnets
    for i in range(ASI.nx):
        for j in range(ASI.ny):
            if ASI.occupation[j, i]:
                unitcell_index = E_MC.kernel_unitcell_indices[j%ASI.unitcell.y, i%ASI.unitcell.x]
                color = colors[unitcell_index] if colors is not None else colors
                draw_magnet(ax, x + ASI_x[i]*w, y + ASI_y[j]*w, l=d*magnet_enlargement, angle=ASI.angles[j, i],
                            color=color, unitcell_index=unitcell_index, round=isinstance(ASI, hotspice.ASI.OOP_ASI))
                ax.add_artist(patches.Circle((x + ASI_x[i]*w, y + ASI_y[j]*w), 0.01, color='k', linewidth=0))

    ## Draw box and unitcell
    for i in range(ASI.nx + 1):
        ucx = i % ASI.unitcell.x == 0 # x=i is at edge of a unitcell?
        if ucx: draw_line(ax, x + ASI_x_central[i]*w, y + ASI_y_central[0]*w, x + ASI_x_central[i]*w, y + ASI_y_central[-1]*w, lw=(lw*2 if ucx else lw), style="-" if ucx else ":")
        if i < ASI.nx: draw_line(ax, x + ASI_x[i]*w, y + ASI_y_central[0]*w, x + ASI_x[i]*w, y + ASI_y_central[-1]*w, lw=lw, style=":")
    
    for i in range(ASI.ny + 1):
        ucy = i % ASI.unitcell.y == 0 # y=i is at edge of a unitcell?
        if ucy: draw_line(ax, x + ASI_x_central[0]*w, y + ASI_y_central[i]*w, x + ASI_x_central[-1]*w, y + ASI_y_central[i]*w, lw=(lw*2 if ucy else lw), style="-" if ucy else ":")
        if i < ASI.ny: draw_line(ax, x + ASI_x_central[0]*w, y + ASI_y[i]*w, x + ASI_x_central[-1]*w, y + ASI_y[i]*w, lw=lw, style=":")
        

## Helper functions
def draw_magnet(ax: plt.Axes, x, y, l, angle=0, color=None, round: bool = False, unitcell_index: int = None):
    if color is None: color = 'gray'
    w = l if round else 4*l/11
    ax.add_artist(patches.Ellipse((x, y), l, w, angle=angle/np.pi*180, color=color, alpha=0.9, linewidth=0))

def draw_line(ax: plt.Axes, x1, y1, x2, y2, color='k', lw=1, style="-"):
    ax.add_artist(lines.Line2D([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style, solid_capstyle='projecting'))


if __name__ == "__main__":
    main_plot()
