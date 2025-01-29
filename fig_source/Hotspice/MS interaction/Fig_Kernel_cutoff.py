import matplotlib.pyplot as plt
import numpy as np
import time

from matplotlib import colormaps

import os
os.environ['HOTSPICE_USE_GPU'] = 'True' # This is beneficial for Metropolis' convolutions
import hotspice
import thesis_utils

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
    
    switches, absdiff_avg, absdiff_max = data['switches'], data['absdiff_avg'], data['absdiff_max']
    cutoff = params['cutoff']
    E_absdiff, E_incremented, E_recalculated = data['E_absdiff'], data['E_incremented'], data['E_recalculated']
    
    ## Plot
    thesis_utils.init_style()
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(thesis_utils.page_width, 2.5))
    ax1: plt.Axes = axes[0,0]
    cmap = colormaps['viridis'].copy()
    hotspice.plottools.init_style()
    fig = plt.figure(figsize=(8, 7))

    # PLOT 1: THE RECALCULATED ENERGY PROFILE
    ax1 = fig.add_subplot(2, 2, 1)
    im1 = ax1.imshow(hotspice.utils.asnumpy(E_recalculated), origin='lower', cmap=cmap)
    c1 = plt.colorbar(im1)
    c1.ax.get_yaxis().labelpad = 15
    c1.ax.set_ylabel(f"Dipolar energy [J]", rotation=270, fontsize=10)
    ax1.set_title("Exact solution")

    # PLOT 2: THE TRUNCATED ENERGY PROFILE
    ax2 = fig.add_subplot(2, 2, 3, sharex=ax1, sharey=ax1)
    im2 = ax2.imshow(hotspice.utils.asnumpy(E_incremented), origin='lower', cmap=cmap)
    c2 = plt.colorbar(im2)
    c2.ax.get_yaxis().labelpad = 15
    c2.ax.set_ylabel(f"Dipolar energy [J]", rotation=270, fontsize=10)
    ax2.set_title(f"Approximation")

    # PLOT 3: THE ABSOLUTE DIFFERENCE
    ax3 = fig.add_subplot(2, 2, 2, sharex=ax1, sharey=ax1)
    im3 = ax3.imshow(hotspice.utils.asnumpy(E_absdiff), origin='lower', cmap=cmap)
    c3 = plt.colorbar(im3)
    c3.ax.get_yaxis().labelpad = 15
    c3.ax.set_ylabel(f"Energy [J]", rotation=270, fontsize=10)
    ax3.set_title(r"Absolute error $E_{err}$")

    # PLOT 4: THE TIME-DEPENDENCE
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(switches, absdiff_avg, color='C0', label=r"$\langle E_{err} \rangle$")
    ax4.plot(switches, absdiff_max, color='C1', label=r"max($E_{err}$)")
    ax4.set_xscale('log')
    ax4.set_xlim([1, np.max(switches)])
    ax4.set_ylim([0, ax4.get_ylim()[1]])
    ax4.set_xlabel("Switches")
    ax4.set_ylabel("Absolute error [J]")
    ax4.legend()

    plt.suptitle(f"{params['Lx']}x{params['Ly']} {params['ASI_type']}, truncated {2*cutoff+1}x{2*cutoff+1} kernel\n{params['iterations']} steps ({switches[-1]} switches)")
    plt.gcf().tight_layout()

    hotspice.utils.save_results(figures={'Kernel_cutoff': fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    # run(hotspice.ASI.IP_Pinwheel(2e-6, 100, T=1e6, E_B=5e-22), n=2000, cutoff=20, pattern='uniform') # As many switches as possible
    run(hotspice.ASI.IP_Pinwheel(2e-6, 100, T=300, E_B=5e-22), n=1000, cutoff=20, pattern='uniform') # Reasonable values
    # run(hotspice.ASI.IP_Pinwheel(1e-6, 200, T=500, E_B=5e-22), n=10000, cutoff=20, pattern='AFM') # Reasonable values with low T
    # run(hotspice.ASI.IP_Pinwheel(2e-6, 100, T=80, E_B=5e-22), n=1, cutoff=20, pattern='uniform') # Procedurally generated modern art
    # run(hotspice.ASI.OOP_Square(2e-6, 100, T=300, E_B=5e-22), n=1000, cutoff=20, pattern='AFM') # OOP_Square
    # run(hotspice.ASI.IP_Kagome(4e-6, 128, T=500, E_B=5e-22, PBC=True), n=10000, cutoff=20, pattern='uniform') # IP_Kagome
    
    thesis_utils.replot_all(plot)
