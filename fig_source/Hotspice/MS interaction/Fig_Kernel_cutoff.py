import matplotlib.pyplot as plt
import numpy as np
import time

import os
os.environ['HOTSPICE_USE_GPU'] = 'True' # This is beneficial for Metropolis' convolutions
import hotspice
import thesis_utils

from hotspice.utils import asnumpy, J_to_eV
from matplotlib import colormaps, colors

xp = hotspice.xp


def run(mm: hotspice.Magnets=None, n: int = 10000, L: int = 400, Lx: int = None, Ly: int = None, cutoff: int = 16, pattern: str = None):
    """ In this analysis, the difference between using either a truncated hotspice.DipolarEnergy() kernel,
        or using the full dipolar kernel, is analyzed.
        
        @param n [int] (10000): the number of times the energy is updated using a reduced kernel.
        @param Lx, Ly [int] (400): the size of the simulation in x- and y-direction. Can also specify `L` for square domain.
        @param cutoff [int] (16): the size of the reduced kernel. TODO: could it be interesting to sweep this?
    """
    if Lx is None: Lx = L
    if Ly is None: Ly = L
    if mm is None: mm = hotspice.ASI.OOP_Square(1e-6, nx=Lx, ny=Ly, PBC=True) # Large spacing to get many Metropolis switches
    Lx, Ly = mm.nx, mm.ny

    if mm.get_energy('dipolar', verbose=False) is None: mm.add_energy(hotspice.DipolarEnergy())
    mm.params.REDUCED_KERNEL_SIZE = cutoff
    mm.params.SIMULTANEOUS_SWITCHES_CONVOLUTION_OR_SUM_CUTOFF = 0 # Need convolution method to use truncated kernel
    mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS # Néel collapses to update_single(), which has no cutoff. Furthermore, using Metropolis samples way more magnets, especially if Q=np.inf.
    mm.PBC = True
    if pattern is not None: mm.initialize_m(pattern)

    steps_done = 0
    interesting_iterations = np.arange(n) + 1
    switches = np.zeros_like(interesting_iterations, dtype=int)
    absdiff_avg = np.zeros_like(switches, dtype=float)
    absdiff_max = np.zeros_like(switches, dtype=float)
    cutoffs = np.zeros_like(switches, dtype=int)
    t = time.perf_counter()
    for i, next_stop in enumerate(interesting_iterations):
        print(i, n)
        if i < mm.n:
            mm.update(r=mm.dx.sum()/4)
        else:
            for _ in range(next_stop - steps_done):
                mm.update(Q=1)
        steps_done = next_stop

        E_incremented = mm.get_energy('dipolar').E.copy() # The approximative kernel after `n` runs
        mm.get_energy('dipolar').update() # Completely recalculate the dipolar energy from scratch
        E_recalculated = mm.get_energy('dipolar').E.copy()
        E_diff = E_recalculated - E_incremented
        E_absdiff = xp.abs(E_diff)
        absdiff_avg[i] = xp.mean(E_absdiff)
        absdiff_max[i] = xp.max(E_absdiff)
        cutoffs[i] = cutoff # This is here in case we want to sweep `cutoff`
        switches[i] = mm.switches
        mm.get_energy('dipolar').E = E_incremented
    t = time.perf_counter() - t

    print(f"Time required for {n} iterations ({mm.switches} switches): {t:.3f}s.")
    print(f"--- ANALYSIS RESULTS ---")
    print(f"max inc: {xp.max(xp.abs(E_incremented))}")
    print(f"max rec: {xp.max(xp.abs(E_recalculated))}")
    print("-"*4)
    print(f"avg diff: {xp.mean(xp.abs(E_diff[mm.occupation != 0]))}")
    print(f"max diff: {xp.max(xp.abs(E_diff[mm.occupation != 0]))}")

    ## Save
    hotspice.utils.save_results(parameters={'T': mm.T_avg, 'E_B': mm.E_B_avg, 'iterations': n, 'cutoff': cutoff, 'dx': mm.dx, 'dy': mm.dy, 'Lx': Lx, 'Ly': Ly, 'ASI_type': hotspice.utils.full_obj_name(mm), 'PBC': mm.PBC, 'pattern': pattern},
                                data={'iteration': interesting_iterations, 'switches': switches, 'absdiff_avg': absdiff_avg, 'absdiff_max': absdiff_max, 'E_absdiff': E_absdiff, 'E_incremented': E_incremented, 'E_recalculated': E_recalculated})
    plot()
    

def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    
    meV = lambda value: J_to_eV(value)*1000
    unit = "meV"
    
    switches, absdiff_avg, absdiff_max = data['switches'], meV(data['absdiff_avg']), meV(data['absdiff_max'])
    switches, absdiff_avg, absdiff_max = np.append(0, switches), np.append(absdiff_avg[0], absdiff_avg), np.append(absdiff_max[0], absdiff_max)
    cutoff = params['cutoff']
    E_absdiff = meV(asnumpy(data['E_absdiff']))
    E_incremented = meV(asnumpy(data['E_incremented']))
    E_recalculated = meV(asnumpy(data['E_recalculated']))
    
    ## Initialise plot
    thesis_utils.init_style()
    fig = plt.figure(figsize=(thesis_utils.page_width, 3.6))
    ratio = 2
    gs = fig.add_gridspec(2, 5, width_ratios=[ratio, ratio+1, .5, ratio+1, 1], height_ratios=[1,1], hspace=.7, wspace=0.1)
    
    cmap = colormaps.get_cmap('viridis')
    vmin = min(E_recalculated.min(), E_incremented.min())
    vmax = max(E_recalculated.max(), E_incremented.max())
    
    def no_ticks(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    ## PLOT 1: THE EXACT ENERGY PROFILE
    ax1: plt.Axes = fig.add_subplot(gs[1,0])
    im1 = ax1.imshow(E_recalculated, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
    ax1.set_title(r"Exact $E_\mathrm{MS}$", fontsize=11)
    no_ticks(ax1)

    ## PLOT 2: THE TRUNCATED ENERGY PROFILE
    ax2: plt.Axes = fig.add_subplot(gs[1,1])
    im2 = ax2.imshow(E_incremented, vmin=vmin, vmax=vmax, origin='lower', cmap=cmap)
    ax2.set_title(f"Approximation", fontsize=11)
    no_ticks(ax2)

    ## PLOT 3: THE ABSOLUTE DIFFERENCE
    ax3: plt.Axes = fig.add_subplot(gs[1,-2])
    im3 = ax3.imshow(E_absdiff, origin='lower', cmap=cmap)
    c3 = plt.colorbar(im3)
    c3.ax.set_ylabel(f"Magnetostatic\ninteraction\nenergy [{unit}]", rotation=270, fontsize=10)
    c3.ax.get_yaxis().labelpad = 35
    ax3.set_title(r"Absolute error $E_\mathrm{err}$", fontsize=11)
    no_ticks(ax3)

    ## PLOT 4: THE TIME-DEPENDENCE
    ax4: plt.Axes = fig.add_subplot(gs[0,:])
    ax4.plot(switches, absdiff_avg, color='C0', label=r"$\langle E_\mathrm{err} \rangle$")
    ax4.plot(switches, absdiff_max, color='C1', label=r"max($E_\mathrm{err}$)")
    ax4.set_xscale('log')
    ax4.set_xlim([1, np.max(switches)])
    ax4.set_ylim([0, ax4.get_ylim()[1]])
    ax4.set_xlabel("Switches")
    ax4.set_ylabel(f"Absolute error [{unit}]")
    ax4.set_title(f"Kernel truncated to {2*cutoff+1}x{2*cutoff+1}", pad=20)
    ax4.legend()
    thesis_utils.label_ax(ax4, 0, offset=(-0.1, 0.2))
    thesis_utils.label_ax(ax4, 1, offset=(-0.1, -1.6)) # Put (b) on ax4 to have same x-coordinate as (a)

    ## LAYOUT PLOT AND SAVE
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, right=0.97, bottom=0.02)
    
    ## POSITION EXACT-APPROX COLORBAR
    pad = 0.01
    width = 0.01
    x0 = ax2.get_position().x1 + pad
    y0 = ax2.get_position().y0
    Y = ax2.get_position().y1 - y0
    cb_ax = fig.add_axes([x0, y0, width, Y])
    c2 = fig.colorbar(im2, cax=cb_ax)
    
    ## Add - and =
    y = (ax2.get_position().y0 + ax2.get_position().y1)/2
    xmin = (ax1.get_position().x1 + ax2.get_position().x0)/2
    xequals = (cb_ax.get_position().x1 + ax3.get_position().x0)/2 + 0.01
    fs = 20
    fig.text(xmin, y, "$-$", va="center", ha="center", fontsize=fs, weight="bold")
    fig.text(xequals, y, "$=$", va="center", ha="center", fontsize=fs, weight="bold")
    
    hotspice.utils.save_results(figures={'Kernel_cutoff': fig}, outdir=data_dir, copy_script=False)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


if __name__ == "__main__":
    # run(hotspice.ASI.IP_Pinwheel(2e-6, 100, T=1e6, E_B=5e-22), n=2000, cutoff=20, pattern='uniform') # As many switches as possible
    # run(mm := hotspice.ASI.IP_Pinwheel(2e-6, 100, T=300, E_B=5e-22), n=3*mm.n, cutoff=20, pattern='uniform') # Reasonable values
    # run(hotspice.ASI.IP_Pinwheel(1e-6, 200, T=500, E_B=5e-22), n=10000, cutoff=20, pattern='AFM') # Reasonable values with low T
    # run(hotspice.ASI.IP_Pinwheel(2e-6, 100, T=80, E_B=5e-22), n=1, cutoff=20, pattern='uniform') # Procedurally generated modern art
    # run(hotspice.ASI.OOP_Square(2e-6, 100, T=300, E_B=5e-22), n=1000, cutoff=20, pattern='AFM') # OOP_Square
    # run(hotspice.ASI.IP_Kagome(4e-6, 128, T=500, E_B=5e-22, PBC=True), n=10000, cutoff=20, pattern='uniform') # IP_Kagome
    
    thesis_utils.replot_all(plot)
