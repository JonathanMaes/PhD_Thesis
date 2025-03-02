import hotspice
import math
import matplotlib.pyplot as plt
import numpy as np
import random

xp = hotspice.xp
from matplotlib import lines, patches

import thesis_utils


def main_plot():
    thesis_utils.init_style()
    
    ## Create a figure that we can draw on everywhere
    figsize = (thesis_utils.page_width, 2.4)
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
    pad = 0.01
    w = 0.4*(x_max - x_min)
    h = w*(y_max - y_min)
    plot_state(ax, x_min+pad,   y_max-pad-h, w, h, state1=1, state2=1, title=r"$E^{(i)}$")
    plot_state(ax, x_min+pad,   y_min+pad,   w, h, state1=0, state2=1, title=r"$E^{(f)}$")
    plot_state(ax, x_max-pad-w, y_max-pad-h, w, h, state1=1, state2=0, title=r"$E^{(i)} + \Delta E^{(i)}$", style="--")
    plot_state(ax, x_max-pad-w, y_min+pad,   w, h, state1=0, state2=0, title=r"$E^{(f)} + \Delta E^{(f)}$", style="--")
    
    ## Add vertical arrow
    ARROW_CENTER = False
    if ARROW_CENTER:
        offset = -0.3*(x_max-2*pad-2*w)
        ax.arrow(x_center+offset, y_max-pad-h/2, 0, y_min-y_max+2.25*pad+h, head_width=2*pad, head_length=2*pad, width=pad, fc="black", ec="None", length_includes_head=True)
        ax.text(x_center+offset+pad, y_center, r"$|\Delta P_2| \leq Q$", ha="left", va="center", rotation=0, fontsize=14)
    else:
        offset = -0.7*(x_max-2*pad-2*w)
        x = x_min+pad + w*0.2
        ax.arrow(x, y_max-pad-h, 0, y_min-y_max+2*(pad+h), head_width=2*pad, head_length=2*pad, width=pad, fc="black", ec="None", length_includes_head=True)
        ax.text(x+2*pad, y_center, r"$|\Delta P_2| \leq Q$", ha="left", va="center", rotation=0, fontsize=14)
    
    ## Add horizontal arrows
    for y, text, va in [(y_max-pad-h/2, "$P_2^{(i)}$", "bottom"), (y_min+pad+h/2, "$P_2^{(f)}$", "top")]:
        ax.arrow(pad+w, y, x_max-2*pad-2*w, 0, head_width=pad, head_length=pad, width=pad/2, fc="grey", ec="None", length_includes_head=True)
        ax.text(x_center, y - 1.5*pad if va == "top" else y + 0.5*pad, text, ha="center", va=va, fontsize=14)
    
    ## Save the result
    hotspice.utils.save_results(figures={f"MultiSwitch_proof": fig}, timestamped=False)


def plot_state(ax: plt.Figure, x, y, w, h, state1: bool = True, state2: bool = True, title: str = "", style="-"):
    draw_rectangle(ax, x, y, w, h, color="grey", style=style)
    l = w*0.2
    draw_magnet(ax, x + w*0.2, y + h*0.5, l, state=state1, color="black", arrow_color="white")
    draw_magnet(ax, x + w*0.8, y + h*0.5, l, state=state2, color="black", arrow_color="white")
    ax.text(x + w*0.5, y + h*0.8, title, ha="center", va="center", fontsize=14)
    ax.text(x + w*0.2, y + h*0.5 + 2*l/11, "1", ha="center", va="bottom", fontsize=12)
    ax.text(x + w*0.8, y + h*0.5 + 2*l/11, "2", ha="center", va="bottom", fontsize=12)
    # Small random magnets
    avoid_points = [(0.2, 0.6), (0.8, 0.6), (0.5, 0.8)]
    tiny_magnet_positions = poisson_disk_sampling(1, 1, 0.25, points=avoid_points, k=100, pad=0.1, seed=7)[len(avoid_points):]
    for dx, dy in tiny_magnet_positions:
        draw_magnet(ax, x + w*dx, y + h*dy, l*0.4, color="#CCC", arrow_color="#666")


## Helper functions
def draw_magnet(ax: plt.Axes, x, y, l, angle=0, color=None, state: bool = True, show_arrow: bool = True, arrow_color="black"):
    if color is None: color = 'gray'
    if not state: angle += np.pi
    w = 4*l/11
    ax.add_artist(patches.Ellipse((x, y), l, w, angle=angle/np.pi*180, color=color, alpha=0.9, linewidth=0))
    if show_arrow:
        arrow_x, arrow_y = np.cos(angle)*l*0.8, np.sin(angle)*l*0.8
        ax.arrow(x-arrow_x/2, y-arrow_y/2, arrow_x, arrow_y, head_width=w/2, head_length=w, fc=arrow_color, width=w/8, ec="None", length_includes_head=True)

def draw_line(ax: plt.Axes, x1, y1, x2, y2, color='k', lw=1, style="-"):
    ax.add_artist(lines.Line2D([x1, x2], [y1, y2], color=color, linewidth=lw, linestyle=style, solid_capstyle='projecting'))

def draw_rectangle(ax: plt.Axes, x, y, width, height, angle=0, color='k', lw=1, style="-", fill=False):
    ax.add_artist(patches.Rectangle((x, y), width, height, angle=angle, edgecolor=color, linewidth=lw, linestyle=style, facecolor=color if fill else 'none'))

def poisson_disk_sampling(width, height, radius, points=None, k=30, pad=0, seed=None):
    """
    Generate points using Bridson's Poisson disk sampling algorithm.

    Parameters:
        width (float): Width of the sampling box.
        height (float): Height of the sampling box.
        radius (float): Minimum distance between points.
        points (list, optional): Initial list of points to seed the sampling process.
        k (int): Number of attempts to find a valid new point for each active point.

    Returns:
        list: List of generated points.
    """
    if seed is not None:
        random.seed(seed)
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    # Initialise variables
    grid_size = radius / math.sqrt(2)
    grid_width = int(math.ceil(width / grid_size))
    grid_height = int(math.ceil(height / grid_size))
    grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
    active_list = []
    final_points = [point for point in points] if points else []

    # Helper to find grid cell coordinates
    def grid_coords(point):
        return int(point[0] // grid_size), int(point[1] // grid_size)

    # Helper to check if a point is valid
    def is_valid_point(point):
        if not (pad <= point[0] <= width - pad) or not (pad <= point[1] <= height - pad): return False
        x, y = grid_coords(point)
        for i in range(max(0, x - 2), min(grid_width, x + 3)):
            for j in range(max(0, y - 2), min(grid_height, y + 3)):
                neighbor = grid[i][j]
                if neighbor and distance(point, neighbor) < radius:
                    return False
        return True

    # Initialise with a random starting point if none provided
    if not points:
        start_point = (random.uniform(0, width), random.uniform(0, height))
        final_points.append(start_point)
    
    for point in final_points:
        active_list.append(point)
        grid_x, grid_y = grid_coords(point)
        grid[grid_x][grid_y] = point

    # Main sampling loop
    while active_list:
        current_point = active_list.pop(random.randint(0, len(active_list) - 1))
        for _ in range(k):
            angle = random.uniform(0, 2 * math.pi)
            r = random.uniform(radius, 2 * radius)
            new_point = (current_point[0] + r * math.cos(angle),
                         current_point[1] + r * math.sin(angle))
            if 0 <= new_point[0] < width and 0 <= new_point[1] < height and is_valid_point(new_point):
                active_list.append(new_point)
                final_points.append(new_point)
                grid_x, grid_y = grid_coords(new_point)
                grid[grid_x][grid_y] = new_point

    return final_points


if __name__ == "__main__":
    main_plot()
