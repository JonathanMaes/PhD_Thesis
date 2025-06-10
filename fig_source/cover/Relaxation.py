""" IN THIS SCRIPT, THE THERMAL RELAXATION OF OOP_SQUARE ASI IS SIMULATED, WHEN STARTING FROM A UNIFORM STATE.
    OOP_Square prefers an AFM state, so spontaneous switching will occur to move towards this ground state.
    The question is whether this relaxation is exponential in time, and how its depends on system parameters.
    
    The main parameters that are varied within this simulation are:
        - E_EA: The effective OOP anisotropy of the individual magnets
        - E_MC: The strength of the dipole-dipole interaction
    Other parameters which can be adjusted, but are not 'swept' throughout this simulation, are:
        - E_B_std: The disorder in the system.
          NOTE: For each relaxation (i.e. each single line in the plots), a completely new and unique 
                <Magnets> object is created, so any features present throughout all the individual lines are
                certainly due to the system parameters rather than a unique random choice of the E_B array.
        - size: The system is a <size>x<size> square-lattice array of OOP magnets.
    
    => OUTPUT: plots of NN correlation and m_avg as a function of time, for various (E_EA, E_MC) combinations.
"""
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import convolve

import hotspice
import thesis_utils

from thermally_active_ASI import get_thermal_mm


def run(t_max: float = 1e4, MCsweeps: float = 4., samples: int = 20):
    """ The simulation will run for `MCsweeps` Monte Carlo sweeps (using Néel updates), or until `t_max` is reached.
        For each combination of E_EA and E_MC, `samples` relaxations will be performed.
    """
    mm_kwargs = {'E_MC_ratio': 5, 'E_EA_ratio': 20, 'E_B_std': 0.05, 'size': 11, 'J_ratio': 0}
    magnet_size_ratios = 0 # If 0, point dipoles are used, otherwise magnets are circular with diameter <magnet_size_ratios*a>

    ## RUN SIMULATIONS
    # Prepare and execute sweep over (varx, vary)
    samples, N_switches = int(samples), int(MCsweeps*get_thermal_mm(**mm_kwargs).n)
    times = np.full((samples, N_switches), np.nan) # NaNs are not plotted, so we pre-fill with those
    m_avgs = np.copy(times)
    corr_NN, corr_2NN, corr_3NN = np.copy(times), np.copy(times), np.copy(times)
    for s in range(samples):
        if magnet_size_ratios != 0: mm_kwargs |= {"magnet_size_ratio": magnet_size_ratios}
        mm = get_thermal_mm(**mm_kwargs)
        mm.initialize_m('uniform')
        for switch in range(N_switches):
            mm.update(t_max=t_max)
            times[s,switch] = mm.t
            m_avgs[s,switch] = mm.m_avg
            corr_NN[s,switch] = mm.correlation_NN()
            corr_2NN[s,switch] = mm.correlation_NN(N=2)
            corr_3NN[s,switch] = mm.correlation_NN(N=3)
            if mm.t >= t_max:
                if (mm.switches % 2) != (mm_kwargs['size'] % 2) == 0: continue # Don't want to stop at a moment where an odd number of magnets have flipped
                times[s,switch:] = mm.t
                m_avgs[s,switch:] = mm.m_avg
                corr_NN[s,switch:] = mm.correlation_NN()
                corr_2NN[s,switch:] = mm.correlation_NN(N=2)
                corr_3NN[s,switch:] = mm.correlation_NN(N=3)
                break
    
    ## Save
    hotspice.utils.save_results(
        parameters={
            "t_max": t_max, "MCsweeps": MCsweeps, "samples": samples, "magnet_size_ratios": magnet_size_ratios
        } | mm_kwargs,
        data={"times": times, "m_avg": m_avgs, "corr_NN": corr_NN, "corr_2NN": corr_2NN, "corr_3NN": corr_3NN, "mm": mm}
    )
    plot()


def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)

    t_max, samples = params["t_max"], params["samples"]
    times, m_avgs, corr_NN, corr_2NN, corr_3NN = data["times"], data["m_avg"], data["corr_NN"], data["corr_2NN"], data["corr_3NN"]
    
    ## Initialise plot
    thesis_utils.init_style(style="default")

    ## PLOT 1: BUNCH OF SUBPLOTS SHOWING metric(t) or metric(switches), WITHOUT AVERAGING THE SAMPLES
    # Plot parameters/switches
    HIGHER_CORRS =    True # Plots q_2NN and q_3NN as well. NOTE: these are not implemented in paper-quality, just in the legend
    PLOT_INDIVIDUAL = False
    ZOOMED =          True

    # Colors
    C0_dark, C1_dark = colors.rgb_to_hsv(colors.to_rgb("C0")), colors.rgb_to_hsv(colors.to_rgb("C1"))
    C0_dark[-1] *= .7
    C1_dark[-1] *= .85
    C0_dark, C1_dark = colors.to_hex(colors.hsv_to_rgb(C0_dark)), colors.to_hex(colors.hsv_to_rgb(C1_dark))
    # Setup figure
    
    ## Create plot
    figsize = (thesis_utils.page_width, thesis_utils.page_width*.5)
    fig = plt.figure(figsize=figsize)
    pad = 0.001
    ax: plt.Axes = fig.add_axes([pad, pad, 1-2*pad, 1-2*pad])
    ax.set_axis_off()
    ax.set_xscale('log')
    # ax.set_xlim([0,1])  # Set x-axis limits
    ax.set_ylim([-0.05, 1.05])  # Set y-axis limits to match aspect ratio
    
    # Plot the data
    if PLOT_INDIVIDUAL: # MANY LINES
        m_avg = np.abs(m_avgs[:,:]) # Because we want it to be in range [0,1].
        q_NN, q_2NN, q_3NN = (1 - corr_NN[:,:])/2, (1 - corr_2NN[:,:])/2, (1 - corr_3NN[:,:])/2
        p1, = ax.plot(np.mean(times, axis=0), np.mean(m_avg, axis=0), label=r"$m_\mathrm{avg}$", color="C0")
        p2, = ax.plot(np.mean(times, axis=0), np.mean(q_NN, axis=0), label=r"$q_\mathrm{NN}$", color="C1")
        if HIGHER_CORRS:
            p3, = ax.plot(np.mean(times, axis=0), np.mean(q_2NN, axis=0), label=r"$q_\mathrm{2NN}$", color="C3")
            p4, = ax.plot(np.mean(times, axis=0), np.mean(q_3NN, axis=0), label=r"$q_\mathrm{3NN}$", color="C6")
        alpha = max(0.9**(samples/2+2), 0.04)
        lw = 3
        for s in range(samples): # Show the various samples as lighter lines
            ax.plot(times[s,:], m_avg[s,:], color="C0", alpha=alpha, linewidth=lw)
            ax.plot(times[s,:], q_NN[s,:], color="C1", alpha=alpha, linewidth=lw)
            if HIGHER_CORRS:
                ax.plot(times[s,:], q_2NN[s,:], color="C3", alpha=alpha, linewidth=lw)
                ax.plot(times[s,:], q_3NN[s,:], color="C6", alpha=alpha, linewidth=lw)
    else: # STATISTICAL AVERAGES ETC.
        x_vals = times[:,:]
        m_avg = np.abs(m_avgs[:,:]) # Because we want it to be in range [0,1].
        q_NN = (1 - corr_NN[:,:])/2
        q_2NN = (1 - corr_2NN[:,:])/2
        q_3NN = (1 - corr_3NN[:,:])/2
        
        def mean_std(times, values, t_samples): # Returns 2D array with dimensions (time, relaxations)
            print(times.shape)
            n_relaxations, T = times.shape
            M = t_samples.shape[0]
            vals = np.empty((M, n_relaxations))
            for i, t in enumerate(t_samples):
                vals[i,:] = values[np.arange(n_relaxations), np.argmin(np.abs(times - t), axis=1)]
                # mean[i], std[i] = np.mean(vals), np.std(vals)
                # perc_low[i], perc_high[i] = np.percentile(vals, 2), np.percentile(vals, 98)
            return vals

        ds = [{"var": m_avg, "label": r"$m_\mathrm{avg}$", "color": "#1E64C8"}, {"var": q_NN, "label": r"$q_\mathrm{NN}$", "color": "C1"}] # UGent blue
        if HIGHER_CORRS: ds += [{"var": q_2NN, "label": r"$q_\mathrm{2NN}$", "color": "C3"}, {"var": q_3NN, "label": r"$q_\mathrm{3NN}$", "color": "C6"}]
        X = np.logspace(np.log10(np.min(x_vals)), np.log10(np.max(x_vals)), 200)
        lns: list[plt.Line2D] = []
        for i, d in enumerate(ds):
            zorder = 100 - i
            # std, perc_low, perc_high = mean_std(x_vals, d["var"], X)
            vals = mean_std(x_vals, d["var"], X)
            mean, std = np.mean(vals, axis=1), np.std(vals, axis=1)
            lns.append(ax.plot(X, mean, label=d["label"], color=d["color"], zorder=zorder)[0])
            ax.fill_between(X, mean - std, mean + std, color=d["color"], edgecolor="none", alpha=0.5, zorder=zorder)
            for perc in range(1, 10, 3):
            # for perc in [1, 3, 5, 10, 25]:
            # for perc in range(1, 50, 5):
                perc_low = np.percentile(vals, perc, axis=1)
                perc_high = np.percentile(vals, 100-perc, axis=1)
                ax.fill_between(X, smooth(perc_low), smooth(perc_high), color=d["color"], edgecolor="none", alpha=0.15, zorder=zorder)

    ax.set_xlim(times[0,0], times[0,-1])
    if ZOOMED: # To obscure the "bounding box"
        t0, t1 = np.log10(np.min(times)), np.log10(np.max(times))
        dt = t1-t0
        ax.set_xlim(10**(t0 + 0.1*dt), 10**(t1 - 0.05*dt))
        ax.set_ylim([0.05, 0.95])
    ## SAVE PLOT
    hotspice.utils.save_results(figures={"OOP_relaxation": fig}, outdir=data_dir, copy_script=False)

def smooth(signal):
    mask = np.ones(3)
    return convolve(signal, mask, mode='same')/convolve(np.ones_like(signal), mask, mode='same')

if __name__ == "__main__":
    # run(MCsweeps=40, samples=100)
    plot(thesis_utils.get_last_outdir())
    # thesis_utils.replot_all(plot)
