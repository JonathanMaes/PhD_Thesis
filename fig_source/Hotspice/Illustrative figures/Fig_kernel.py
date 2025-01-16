import hotspice
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp
from hotspice.utils import asnumpy
from matplotlib import lines, patches

import thesis_utils


def main_plot():
    thesis_utils.init_style()
    
    ## Create a figure that we can draw on everywhere
    fig = plt.figure(figsize=(thesis_utils.page_width, 3))
    ax: plt.Axes = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim([0, 1])  # Set x-axis limits
    ax.set_ylim([0, fig.get_size_inches()[1] / fig.get_size_inches()[0]])  # Set y-axis limits to match aspect ratio
    ax.set_aspect('equal', adjustable='datalim')  # Ensure equal aspect ratio based on data limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    print(f"Figure coordinates:\n\tx {x_min:.3f} --> {x_max:.3f}\n\ty {y_min:.3f} --> {y_max:.3f}")
    ax.axis('off')  # Disable axis lines and labels
    ax.patch.set_alpha(0)  # Transparent background
    
    ## Add the parts
    style_kwargs = dict(n=5, lw=1, ASI_type=hotspice.ASI.IP_Pinwheel, colors = {(1,2): 'C0', (4,3): 'C1'})
    # style_kwargs = dict(n=15, lw=1, ASI_type=hotspice.ASI.IP_Cairo, colors = {(1,2): 'C0', (4,6): 'C1'})
    # style_kwargs = dict(n=12, lw=1, ASI_type=hotspice.ASI.IP_Kagome, colors = {(1,2): 'C0', (8,9): 'C1'})

    plot_ASI(ax, 0.01, (y_max-0.15)/2, 0.15, magnet_enlargement=1, **style_kwargs)
    for i, pos in enumerate(style_kwargs.get('colors', {}).keys()):
        plot_kernel(ax, 0.3+i*0.35, (y_max-0.3)/2, 0.3, pos=pos, **style_kwargs)
    
    ## Save the result
    ASI_type = style_kwargs.get("ASI_type", None)
    ASI_type = "" if ASI_type is None else f"_{ASI_type.__name__}"
    hotspice.utils.save_results(figures={f"Fig_kernel{ASI_type}": fig}, timestamped=False)
    plt.show()


def plot_ASI(ax: plt.Figure, x, y, w, n=5, lw=1, magnet_enlargement=1, ASI_type: type[hotspice.Magnets] = hotspice.ASI.IP_Pinwheel, colors: dict|str = None):
    ## Argument processing
    if colors is None: colors = {} # Dict, e.g. {(1,2): 'yellow', (3,2): 'red'}
    d = w/n # Size of one cell
    
    ## Create ASI
    ASI = ASI_type(a=1, nx=n, ny=n)
    E_MC: hotspice.energies.DipolarEnergy = ASI.get_energy('dipolar')
    
    ## Draw magnets
    for i in range(ASI.nx):
        for j in range(ASI.ny):
            if ASI.occupation[j, i]:
                color = colors.get((i,j), None) if isinstance(colors, dict) else colors
                unitcell_index = E_MC.kernel_unitcell_indices[j%ASI.unitcell.y, i%ASI.unitcell.x]
                draw_magnet(ax, x + (i+.5)*d, y + (j+.5)*d, l=d*magnet_enlargement, angle=ASI.angles[j, i],
                            color=color, unitcell_index=None, round=isinstance(ASI, hotspice.ASI.OOP_ASI))

    ## Draw box and unitcell
    x0, y0, x1, y1 = x, y, x + ASI.nx*d, y + ASI.ny*d
    for i in range(ASI.nx + 1):
        draw_line(ax, x + i*d, y0, x + i*d, y1, lw=(lw*2 if i % ASI.unitcell.x == 0 else lw))
        draw_line(ax, x0, y + i*d, x1, y + i*d, lw=(lw*2 if i % ASI.unitcell.y == 0 else lw))


def plot_kernel(ax: plt.Figure, x, y, w, n=5, lw=1, pos: tuple = (0, 0), magnet_enlargement=1, ASI_type: type[hotspice.Magnets] = hotspice.ASI.IP_Pinwheel, colors: dict|str = None):
    ## Argument processing
    if colors is None: colors = {}
    color = colors.get(pos, "grey") if isinstance(colors, dict) else colors
    
    ## Create ASI
    ASI = ASI_type(a=1, nx=n, ny=n)
    E_MC: hotspice.energies.DipolarEnergy = ASI.get_energy('dipolar')
    
    ## Spacing of plot grid
    kernel_unitcell_index = E_MC.kernel_unitcell_indices[pos[1]%ASI.unitcell.y, pos[0]%ASI.unitcell.x]
    if kernel_unitcell_index < 0: raise ValueError(f"There is no magnet at {pos=}")
    kernel: xp.ndarray = E_MC.kernel_unitcell[kernel_unitcell_index]
    n_kernel = kernel.shape[0] # ASI is square so kernel too
    d = w/n_kernel # Size of one cell
    
    ## Draw box and unitcells
    x0, y0, x1, y1 = x, y, x + n_kernel*d, y + n_kernel*d
    # Highlight center column and row
    ax.add_artist(patches.Rectangle(xy=(x + (ASI.nx-1)*d, y0), width=d, height=d*n_kernel, color=blend_colors(color, "w", 0.75)))
    ax.add_artist(patches.Rectangle(xy=(x0, y + (ASI.ny-1)*d), width=d*n_kernel, height=d, color=blend_colors(color, "w", 0.75)))
    # Draw all lines
    for i in range(n_kernel + 1):
        line_kwargs_x = dict(lw=(lw*2 if (i - ASI.nx + 1 + pos[0]) % ASI.unitcell.x == 0 else lw), color=color)
        line_kwargs_y = dict(lw=(lw*2 if (i - ASI.nx + 1 + pos[1]) % ASI.unitcell.y == 0 else lw), color=color)
        draw_line(ax, x + i*d, y0, x + i*d, y1, **line_kwargs_x)
        draw_line(ax, x0, y + i*d, x1, y + i*d, **line_kwargs_y)
    
    ## Draw magnets
    dx, dy = d*(ASI.nx - pos[0] - 1), d*(ASI.ny - pos[1] - 1) # Offset the ASI to the right place in the kernel to put magnet in center
    for i in range(ASI.nx):
        for j in range(ASI.ny):
            if ASI.occupation[j, i]:
                color = colors.get((i,j), None) if isinstance(colors, dict) else color
                unitcell_index = E_MC.kernel_unitcell_indices[j%ASI.unitcell.y, i%ASI.unitcell.x]
                draw_magnet(ax, x+dx + (i+.5)*d, y+dy + (j+.5)*d, l=d*magnet_enlargement, angle=ASI.angles[j, i],
                            color=color, unitcell_index=None, round=isinstance(ASI, hotspice.ASI.OOP_ASI))

    ## Draw magnets box and unitcell
    x0, y0, x1, y1 = x+dx, y+dy, x+dx + ASI.nx*d, y+dy + ASI.ny*d
    color = colors.get(pos, "grey") if isinstance(colors, dict) else colors
    for i in range(ASI.nx + 1):
        draw_line(ax, x+dx + i*d, y0, x+dx + i*d, y1, lw=(lw*2 if i % ASI.unitcell.x == 0 else lw), color=blend_colors("k", color, 0.4))
        draw_line(ax, x0, y+dy + i*d, x1, y+dy + i*d, lw=(lw*2 if i % ASI.unitcell.y == 0 else lw), color=blend_colors("k", color, 0.4))


## Helper functions
def draw_magnet(ax: plt.Axes, x, y, l, angle=0, color=None, round: bool = False, unitcell_index: int = None):
    if color is None: color = 'gray'
    w = l if round else 4*l/11
    ax.add_artist(patches.Ellipse((x, y), l, w, angle=angle/np.pi*180, color=color, alpha=0.9, linewidth=0))
    if unitcell_index is not None:
        ax.text(x, y, f"{unitcell_index:d}", ha="center", va="center_baseline", fontsize=8)

def draw_line(ax: plt.Axes, x1, y1, x2, y2, color='k', lw=1, style="-"):
    ax.add_artist(lines.Line2D([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style, solid_capstyle='projecting'))


def blend_colors(color1, color2, blend_ratio=0.5) -> tuple[float, float, float]:
    """ Blend two matplotlib colors (e.g., "k", "C0", or an RGB tuple) using the
        specified blend ratio. Returns an RGB tuple with values in range 0 to 1.
    """
    return tuple((1 - blend_ratio) * c1 + blend_ratio * c2 for c1, c2 in zip(mcolors.to_rgb(color1), mcolors.to_rgb(color2)))


if __name__ == "__main__":
    main_plot()
