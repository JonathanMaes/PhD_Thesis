import gc
import time

import matplotlib.pyplot as plt
import numpy as np

import os
os.environ["HOTSPICE_USE_GPU"] = "true"
import hotspice

import thesis_utils

def get_performance(mm: hotspice.Magnets, t_min: float = 1, n_min: int = 1, verbose: bool = False):
    """ In this analysis, the performance of Hotspice for the geometry in `mm` is measured,
        by simply calculating the number of attempted switches per second, real switches
        per second, and Monte Carlo steps per second, when calling `mm.update()` successively.
        @param mm [hotspice.Magnets]: the Magnets object with the desired size and parameters.
        @param t_min [float] (1): the minimal number of seconds during which the performance is monitored.
        @param n_min [int] (1): the minimal number of `mm.update()` calls whose performance is monitored.
    """
    if hotspice.config.USE_GPU:
        hotspice.utils.free_gpu_memory()
    else:
        gc.collect()
    i, t0 = -1, time.perf_counter()
    while (i := i + 1) < n_min or time.perf_counter() - t0 < t_min: # Do this for at least `n_min` iterations and `t_min` seconds
        mm.update()
    dt = time.perf_counter() - t0
    if verbose:
        GPU_text = f"[{hotspice.utils.get_gpu_memory()['free']} free on GPU] " if hotspice.config.USE_GPU else ""
        print(f"{GPU_text}Time required for {i} runs ({mm.switches} switches) of Magnets.select() on {mm.nx}x{mm.ny} grid: {dt:.3f}s.")
    return {'attempts/s': mm.attempted_switches/dt, 'switches/s': mm.switches/dt, 'MCsteps/s': mm.MCsteps/dt}


def performance_sweep(L_range, ASI_type: type[hotspice.Magnets] = hotspice.ASI.OOP_Square, verbose: bool = False, **kwargs):
    L_range: np.ndarray = np.asarray(L_range)
    N = np.zeros_like(L_range)
    attempts_per_s = np.zeros_like(L_range, dtype='float')
    switches_per_s = np.zeros_like(L_range, dtype='float')
    MCsteps_per_s = np.zeros_like(L_range, dtype='float')
    for i, L in enumerate(L_range):
        try:
            try: del mm
            except NameError: pass
            mm = ASI_type(1e-6, L, ny=L, **kwargs)
            mm.params.UPDATE_SCHEME = hotspice.Scheme.METROPOLIS
            mm.update() # Prevents an outlier at the first N
            x = get_performance(mm, verbose=verbose)
            N[i], attempts_per_s[i], switches_per_s[i], MCsteps_per_s[i] = mm.n, x['attempts/s'], x['switches/s'], x['MCsteps/s']
        except Exception as e:
            print(e)
            if verbose: print(f"Could not initialize {ASI_type} for L={L}.")
            continue
    nz = N.nonzero()
    L_range, N, attempts_per_s, switches_per_s, MCsteps_per_s = L_range[nz], N[nz], attempts_per_s[nz], switches_per_s[nz], MCsteps_per_s[nz]
    
    parameters = {'GPU': hotspice.config.USE_GPU, "scheme": mm.params.UPDATE_SCHEME, 'T': mm.T_avg, 'E_B': mm.E_B_avg, 'dx': mm.dx, 'dy': mm.dy, 'ASI_type': hotspice.utils.full_obj_name(mm), 'PBC': mm.PBC}
    data = {"L": L_range, "N": N, 'attempts/s': attempts_per_s, 'switches/s': switches_per_s, 'MCsteps/s': MCsteps_per_s}
    
    ## Save
    hotspice.utils.save_results(parameters=parameters, data=data)
    plot()

def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    L, N = data["L"], data["N"]
    GPU = params["GPU"]
    
    ## We need two axes, because Matplotlib does not natively support broken axes.
    hotspice.plottools.init_style()
    fig = plt.figure(figsize=(thesis_utils.page_width/2, 3.5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.plot(N, data['attempts/s'], label="Samples / sec")
    ax1.plot(N, data['switches/s'], label="Switches / sec")
    ax1.plot(N, data['MCsteps/s'], label="MC sweeps / sec")
    ax1.set_xlim([N.min(), N.max()])
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel("Number of magnets")
    if not GPU: ax1.set_ylabel("Throughput [$s^{-1}$]")

    ax2.set_xscale('log')
    ax2.set_xlim([L.min(), L.max()])
    ticks = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
    ax2.set_xticks(ticks[np.where((ticks >= ax2.get_xlim()[0]) & (ticks <= ax2.get_xlim()[1]))])
    ax2.xaxis.grid(linestyle=':', zorder=1)
    ax2.set_xlabel("Cells in x- and y-direction")
    
    if not GPU: ax2.legend(*ax1.get_legend_handles_labels(), loc="lower left")
    plt.gcf().tight_layout()
    PU = "GPU" if GPU else "CPU"
    hotspice.utils.save_results(figures={f'Performance_{PU}': fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    if hotspice.config.USE_GPU: # for GPU
        L_range = np.concatenate([np.arange(1, 100, 1), np.arange(100, 400, 5), np.arange(400, 600, 10), np.arange(600, 1001, 25)])
    else: # for CPU
        L_range = np.concatenate([np.arange(1, 100, 1), np.arange(100, 251, 5)])

    # performance_sweep(L_range=L_range, T=100, PBC=False, verbose=True, pattern="random")
    # plot()
    
    thesis_utils.replot_all(plot)
