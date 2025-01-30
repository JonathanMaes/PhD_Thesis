import os
os.environ["HOTSPICE_USE_GPU"] = "True"
import hotspice
import thesis_utils
xp = hotspice.xp


import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import colormaps, gridspec, patches
from scipy.spatial import distance
from typing import Iterable


def calculate_any_neighbours(pos, shape, center: int = 0):
    """ @param pos [xp.array(2xN)]: the array of indices (row 0: y, row 1: x)
        @param shape [tuple(2)]: the maximum indices (+1) in y- and x- direction
        @param center [int]: if 0, then the full neighbour array is returned. If nonzero, only the 
            middle region (of at most `center` cells away from the middle) is returned.
        @return [xp.array2D]: an array representing where the other samples are relative to every
            other sample. To this end, each sample is placed in the middle of the array, and all
            positions where another sample exists are incremented by 1. As such, the array is
            point-symmetric about its middle.
    """
    # Note that this function can entirely be replaced by
    # signal.convolve2d(arr, arr, mode='full')
    # but this is very slow for large, sparse arrays like we are dealing with here.
    final_array = xp.zeros((2*shape[0]-1)*(2*shape[1]-1))
    pairwise_distances = (pos.T[:,None,:] - pos.T).reshape(-1,2).T # The real distances as coordinates
    # But we need to bin them, so we have to increase everything so there are no nonzero elements, and flatten:
    pairwise_distances_flat = (pairwise_distances[0] + shape[0] - 1)*(2*shape[1]-1) + (pairwise_distances[1] + shape[1] - 1)
    pairwise_distances_flat_binned = xp.bincount(pairwise_distances_flat)
    final_array[:pairwise_distances_flat_binned.size] = pairwise_distances_flat_binned
    final_array = final_array.reshape((2*shape[0]-1, 2*shape[1]-1))
    if center == 0:
        return final_array # No need to center it at all
    else:
        pad_x, pad_y = -(shape[1] - center - 1), -(shape[0] - center - 1) # How much padding(>0)/cropping(<0) needed to center the array
        final_array = final_array[:,-pad_x:-pad_x + 2*center+1] if pad_x < 0 else xp.pad(final_array, ((0,0), (pad_x,pad_x))) # x cropping/padding
        final_array = final_array[-pad_y:-pad_y + 2*center+1,:] if pad_y < 0 else xp.pad(final_array, ((pad_y,pad_y), (0,0))) # y cropping/padding
        return final_array


def run(n_samples: int=100000, L:int=400, r:float=16, PBC:bool=True, ASI_type:type[hotspice.Magnets]=None, method:str="grid"):
    """ In this analysis, the multiple-magnet-selection algorithm of `hotspice.Magnets.select()` is analyzed.
        The spatial distribution is calculated by performing `n` runs of the `select()` method.
        Also the probability distribution of the distance between two samples is calculated,
        as well as the probablity distribution of their relative positions.
            (note that this becomes expensive to calculate for large `L`)
        @param n [int] (10000): the number of times the `select()` method is executed.
        @param Lx, Ly [int] (400): the size of the simulation in x- and y-direction. Can also specify `L` for square domain.
        @param r [float] (16): the minimal distance between two selected magnets (specified as a number of cells).
    """
    if isinstance(L, Iterable):
        if len(L) != 2: raise ValueError("L has to be an integer or a tuple of two integers (Lx, Ly).")
        Lx, Ly = L
    else:
        Lx = Ly = int(L)
    if ASI_type is None: ASI_type = hotspice.ASI.OOP_Square
    mm = ASI_type(1, Lx, ny=Ly, PBC=PBC)
    mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS
    mm.params.MULTISAMPLING_SCHEME = 'grid'
    INTEGER_BINS = False # If true, the bins are pure integers, otherwise they can be finer than this.
    ONLY_SMALLEST_DISTANCE = True # If true, only the distances to nearest neighbours are counted.
    scale: int = 3 if ONLY_SMALLEST_DISTANCE else 4 # The distances are stored up to scale*r, beyond that we dont keep in memory

    n_bins = int(r*scale+1) if INTEGER_BINS else 99
    distances_binned = xp.zeros(n_bins)
    bin_width = r*scale/(n_bins-1)
    distance_bins = np.linspace(0, r*scale-bin_width, n_bins)
    max_dist_bin = r*scale
    
    t = time.perf_counter()
    total = 0 # Number of samples
    min_dist = xp.inf # Minimal distance between 2 samples in 1 mm.select() call
    field = xp.zeros_like(mm.m) # The number of times each given cell was chosen
    field_local = xp.zeros((2*r*scale+1, 2*r*scale+1)) # Basically distances_binned, but in 2D (distribution of neighbours around a choice)
    spectrum = xp.zeros_like(field)
    while total < n_samples:
        print(total, "/", n_samples)
        match method.lower():
            case "poisson": pos = mm._select_Poisson(r=r)
            case "hybrid": pos = mm.select(r=r, poisson=True)
            case _: pos = mm.select(r=r)
        pos = mm._select_Poisson(r=r)
        total += pos.shape[1]
        choices = xp.zeros_like(field)
        choices[pos[0], pos[1]] += 1
        field += choices
        if mm.PBC:
            _, n_pos = pos.shape # The following approach is quite suboptimal, but it works :)
            all_pos = xp.zeros((2, n_pos*4), dtype=int)
            all_pos[:,:n_pos] = pos
            all_pos[0,n_pos:n_pos*2] = all_pos[0,:n_pos] + mm.ny
            all_pos[1,n_pos:n_pos*2] = all_pos[1,:n_pos]
            all_pos[0,n_pos*2:] = all_pos[0,:n_pos*2]
            all_pos[1,n_pos*2:] = all_pos[1,:n_pos*2] + mm.nx
        else:
            all_pos = pos
        if all_pos.shape[1] > 1: # if there is more than 1 sample
            if ONLY_SMALLEST_DISTANCE:
                dist_matrix = xp.asarray(distance.cdist(hotspice.utils.asnumpy(all_pos.T), hotspice.utils.asnumpy(all_pos.T)))
                dist_matrix[dist_matrix==0] = np.inf
                distances = xp.min(dist_matrix, axis=1)
            else:
                distances = xp.asarray(distance.pdist(hotspice.utils.asnumpy(all_pos.T)))
            min_dist = min(min_dist, xp.min(distances))
            near_distances = distances[distances < max_dist_bin]
            if near_distances.size != 0:
                bin_counts = xp.bincount(xp.clip(xp.floor(near_distances/r*(n_bins/scale)), 0, n_bins-1).astype(int))
                distances_binned[:bin_counts.size] += bin_counts
            field_local += calculate_any_neighbours(all_pos, (mm.ny*(1+mm.PBC), mm.nx*(1+mm.PBC)), center=r*scale)
            spectrum += xp.log(xp.abs(xp.fft.fftshift(xp.fft.fft2(choices)))) # Not sure if this should be done always or only if more than 1 sample exists
        
    field_local[r*scale, r*scale] = 0 # set center value to zero
    t = time.perf_counter() - t
    
    print(f"Time required to select {n_samples} magnets: {t:.3f}s.")
    print(f"--- ANALYSIS RESULTS ---")
    print(f"Total number of actual samples: {total}")
    print(f"Empirical minimal distance between two samples in a single selection: {min_dist:.2f} (r={r})")
    
    ## Save
    hotspice.utils.save_results(parameters={"r": r, "L": L, "scale": scale, "n_samples": n_samples,
                                            "bin_width": bin_width, "max_dist_bin": max_dist_bin,
                                            "ONLY_SMALLEST_DISTANCE": ONLY_SMALLEST_DISTANCE, "PBC": PBC,
                                            "dx": mm.dx, "dy": mm.dy},
                                data={"total": total, "distance_bins": distance_bins, "distances_binned": distances_binned, "field_local": field_local, "field": field, "spectrum": spectrum})
    plot()


def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    
    r, L, scale = params["r"], params["L"], params["scale"]
    bin_width, max_dist_bin, ONLY_SMALLEST_DISTANCE = params["bin_width"], params["max_dist_bin"], params['ONLY_SMALLEST_DISTANCE']
    dx, dy = params["dx"], params["dy"]
    distance_bins, distances_binned, field_local, field, spectrum = data["distance_bins"], data["distances_binned"], data["field_local"], data["field"], data["spectrum"]
    total = data["total"]
    
    ## Initialise plot
    thesis_utils.init_style()
    fig = plt.figure(figsize=(thesis_utils.page_width, thesis_utils.page_width/2))
    gs = fig.add_gridspec(2, 4, width_ratios=[2, .5, 1, 1], height_ratios=[1,1], hspace=0.7, wspace=0)
    
    cmap = colormaps['viridis'].copy()
    cmap.set_under(color='black')

    # PLOT 1: HISTOGRAM OF (NEAREST) NEIGHBOURS
    # ax1 = fig.add_subplot(gs[0,2:])
    ax1 = fig.add_subplot(2, 2, 2)
    if ONLY_SMALLEST_DISTANCE:
        color_left = 'C1'
        ax1.set_xlabel("Distance to nearest neighbour (binned)")
        data1_left = hotspice.utils.asnumpy(distances_binned/xp.sum(distances_binned))/(bin_width/r)
        ax1.fill_between(distance_bins, data1_left, step='post', edgecolor=color_left, facecolor=color_left, alpha=0.7)
        ax1.set_ylabel("Probability density [$r^{-1}$]", color=color_left)
        ax1.tick_params(axis='y', labelcolor=color_left)

        ax1_right = ax1.twinx() # instantiate a second axes that shares the same x-axis
        color_right = 'C0'
        ax1_right.set_ylabel("Cumulative probability", color=color_right)
        data1_right = hotspice.utils.asnumpy(xp.cumsum(distances_binned/xp.sum(distances_binned)))
        ax1_right.bar(distance_bins, data1_right, align='edge', width=bin_width, color=color_right)
        ax1_right.tick_params(axis='y', labelcolor=color_right)
        ax1.set_zorder(ax1_right.get_zorder() + 1)
        ax1.patch.set_visible(False)

        ax1.set_ylim([0, ax1.get_ylim()[1]])
        ax1_right.set_ylim([0, ax1_right.get_ylim()[1]]) # might want to set this to [0, 1]
    else:
        ax1.bar(distance_bins, hotspice.utils.asnumpy(distances_binned), align='edge', width=bin_width)
        ax1.set_xlabel("Distance to any other sample (binned)")
        ax1.set_title("Inter-sample distances")
        ax1.set_ylabel("# occurences")
    ax1.set_xlim([0, max_dist_bin])
    ax1.set_xticks([r*n for n in range(scale+1)])
    ax1.set_xticklabels(["0", "r"] + [f"{n}r" for n in range(2, scale+1)])
    ax1.axvline(r, color='black', linestyle=':', linewidth=1, label=None)

    # PLOT 2: PROBABILITY DENSITY OF NEIGHBOURS AROUND ANY SAMPLE
    # ax2 = fig.add_subplot(gs[:,0])
    ax2 = fig.add_subplot(1, 2, 1)
    data2 = hotspice.utils.asnumpy(field_local)/total
    im2 = ax2.imshow(data2, vmin=1e-10, vmax=max(2e-10, np.max(data2)), extent=[-.5-r*scale, .5+r*scale, -.5-r*scale, .5+r*scale], interpolation_stage='rgba', interpolation='nearest', cmap=cmap)

    ax2.add_patch(plt.Circle((0, 0), 0.707, linewidth=0.5, fill=False, color='white'))
    ax2.add_patch(patches.Ellipse((0, 0), 2*r/min(dx), 2*r/min(dy), linewidth=1, fill=False, color='white', linestyle=':'))
    c2 = plt.colorbar(im2, extend='min', location="top")
    c2.ax.set_title("Prob. dens. of neighbours\naround any sample", fontsize=11)

    gs_inner = gridspec.GridSpecFromSubplotSpec(1, 3, width_ratios=[1, 0.5, 1], subplot_spec=gs[1,2:])

    # PLOT 3: PROBABILITY OF CHOOSING EACH CELL
    # ax3 = fig.add_subplot(gs_inner[0,0])
    ax3 = fig.add_subplot(2, 4, 7)
    data3 = hotspice.utils.asnumpy(field)
    im3 = ax3.imshow(data3, vmin=1e-10, origin='lower', interpolation_stage='rgba', interpolation='none', cmap=cmap)
    # ax3.set_yticks([])
    # ax3.set_title("# choices for each cell")
    c3 = plt.colorbar(im3, extend='min', location="right")
    ax3.set_title("Samples per cell", fontsize=11)

    # PLOT 4: PERIODOGRAM
    # ax4 = fig.add_subplot(gs_inner[0,2])
    ax4 = fig.add_subplot(2, 4, 8)
    freq = hotspice.utils.asnumpy(xp.fft.fftshift(xp.fft.fftfreq(L, d=1))) # use fftshift to get ascending frequency order
    data4 = hotspice.utils.asnumpy(spectrum)/total
    extent = [-.5+freq[0], .5+freq[-1], -.5+freq[0], .5+freq[-1]]
    im4 = ax4.imshow(data4, extent=extent, interpolation_stage='rgba', interpolation='none', cmap='gray')
    zoom = 4
    ax4.set_xlim([-1/zoom, 1/zoom])
    ax4.set_ylim([-1/zoom, 1/zoom])
    ax4.set_xticks([-1/(zoom+1), 0, 1/(zoom+1)])
    ax4.set_yticks([-1/(zoom+1), 0, 1/(zoom+1)])
    ax4.yaxis.tick_right()
    ax4.set_title("Periodogram")
    # plt.colorbar(im4)
    
    # fig.suptitle(f"{L}x{L} grid, r={r} cells: ({total} samples)\nPBC {'en' if PBC else 'dis'}abled")
    fig.tight_layout()
    fig.subplots_adjust(left=0.03, right=0.9, bottom=0.07, wspace=0.2)
    
    def move_ax(ax: plt.Axes, dx=0, dy=0, dw=0, dh=0):
        pos = ax.get_position()
        ax.set_position([pos.x0+dx, pos.y0+dy, pos.x1-pos.x0+dw, pos.y1-pos.y0+dh])
    move_ax(ax1, dy=-0.03, dx=0.02, dh=0.12) # Cumprob
    move_ax(ax4, dx=0.05) # Periodogram
    
    thesis_utils.label_ax(ax1, 1, offset=(0.02, -0.15)) # Cumprob
    thesis_utils.label_ax(ax2, 0, offset=(-0.2, 0.35)) # Main imshow
    thesis_utils.label_ax(ax3, 2, offset=(-0.5, 0.08)) # Samples per cell
    thesis_utils.label_ax(ax4, 3, offset=(-0.4, 0.08)) # Periodogram
    
    hotspice.utils.save_results(figures={f'MultiSwitch_select_{os.path.basename(data_dir)}': fig}, outdir=data_dir, copy_script=False)

if __name__ == "__main__":
    PBC = True
    # run(L=400, n_samples=1000000, r=16, ASI_type=hotspice.ASI.OOP_Square, PBC=PBC, method="grid")
    thesis_utils.replot_all(plot)
