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
import matplotlib.ticker as ticker
import numpy as np

import hotspice
import thesis_utils

from thermally_active_ASI import get_thermal_mm


def run(t_max: float = 1e4, MCsweeps: float = 4., samples: int = 20):
    """ The simulation will run for `MCsweeps` Monte Carlo sweeps (using Néel updates), or until `t_max` is reached.
        For each combination of E_EA and E_MC, `samples` relaxations will be performed.
    """
    mm_kwargs = {'E_B_std': 0.05, 'size': 11, 'J_ratio': 0}
    magnet_size_ratios = 0 # If 0, point dipoles are used, otherwise magnets are circular with diameter <magnet_size_ratios*a>
    
    # Sweep-related params
    varx_name, vary_name = "E_MC_ratio", "E_EA_ratio" # NN Dipole-dipole interaction, and energy barrier, in multiples of kBT
    # varx_values, vary_values = [0, 2.5, 10, 40], [0, 20, 40]
    varx_values, vary_values = [1.25, 2.5, 10], [40, 30, 20] #! These are the paper values.
    
    ## RUN SIMULATIONS
    # Sanitize input
    if varx_name not in (allowed_varx_vary := ("E_MC_ratio", "E_EA_ratio", "J_ratio", "gradient")) or vary_name not in allowed_varx_vary:
        raise ValueError(f"varx_name and vary_name must be any of {allowed_varx_vary}")
    for name in (varx_name, vary_name): mm_kwargs.pop(name, None) # None avoids KeyError
    vary_values, varx_values = np.asarray(vary_values), np.asarray(varx_values)
    # Prepare and execute sweep over (varx, vary)
    samples, N_switches = int(samples), int(MCsweeps*get_thermal_mm(**mm_kwargs).n)
    times = np.full((samples, vary_values.size, varx_values.size, N_switches), np.nan) # NaNs are not plotted, so we pre-fill with those
    m_avgs = np.copy(times)
    corr_NN, corr_2NN, corr_3NN = np.copy(times), np.copy(times), np.copy(times)
    final_states = np.zeros((vary_values.size, varx_values.size, mm_kwargs['size'], mm_kwargs['size']))
    for i, vary in enumerate(vary_values):
        for j, varx in enumerate(varx_values):
            for s in range(samples):
                mm_kwargs_here = mm_kwargs | {varx_name: varx, vary_name: vary}
                if magnet_size_ratios != 0: mm_kwargs_here |= {"magnet_size_ratio": magnet_size_ratios[i]}
                mm = get_thermal_mm(**mm_kwargs_here)
                mm.initialize_m('uniform')
                for switch in range(N_switches):
                    mm.update(t_max=t_max)
                    times[s,i,j,switch] = mm.t
                    m_avgs[s,i,j,switch] = mm.m_avg
                    corr_NN[s,i,j,switch] = mm.correlation_NN()
                    corr_2NN[s,i,j,switch] = mm.correlation_NN(N=2)
                    corr_3NN[s,i,j,switch] = mm.correlation_NN(N=3)
                    if mm.t >= t_max:
                        if (mm.switches % 2) != (mm_kwargs['size'] % 2) == 0: continue # Don't want to stop at a moment where an odd number of magnets have flipped
                        times[s,i,j,switch:] = mm.t
                        m_avgs[s,i,j,switch:] = mm.m_avg
                        corr_NN[s,i,j,switch:] = mm.correlation_NN()
                        corr_2NN[s,i,j,switch:] = mm.correlation_NN(N=2)
                        corr_3NN[s,i,j,switch:] = mm.correlation_NN(N=3)
                        break
                # if s == 0: # Random system and outcome (s=0 is already as random as it gets)
                #     final_states[i,j,:,:] = mm.m
                furthest = np.nanmax(times[:,i,j,-1]) # Largest time that a sample for these values of varx and vary has reached
                if np.isclose(furthest, times[s,i,j,-1]): # Only save state if this one progressed the furthest yet out of all the samples
                    final_states[i,j,:,:] = mm.m
    
    ## Save
    hotspice.utils.save_results(
        parameters={
            "varx_name": varx_name, "vary_name": vary_name, "varx_values": varx_values, "vary_values": vary_values,
            "t_max": t_max, "MCsweeps": MCsweeps, "samples": samples, "magnet_size_ratios": magnet_size_ratios
        } | mm_kwargs,
        data={"times": times, "m_avg": m_avgs, "corr_NN": corr_NN, "corr_2NN": corr_2NN, "corr_3NN": corr_3NN, "final_states": final_states, "mm": mm}
    )
    plot()


def plot(data_dir=None):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)

    varx_name, vary_name = params["varx_name"], params["vary_name"]
    varx_values, vary_values = np.asarray(params["varx_values"]), np.asarray(params["vary_values"])
    t_max, samples = params["t_max"], params["samples"]
    times, m_avgs, corr_NN, corr_2NN, corr_3NN = data["times"], data["m_avg"], data["corr_NN"], data["corr_2NN"], data["corr_3NN"]
    final_states, mm = data["final_states"], data["mm"]
    
    ## Initialise plot
    thesis_utils.init_style(style="default")

    ## PLOT 1: BUNCH OF SUBPLOTS SHOWING metric(t) or metric(switches), WITHOUT AVERAGING THE SAMPLES
    CLIP_RANGES =          True # If True, axes and magnitudes are clipped to realistic ranges for the OOP ASI paper.
    # Plot parameters/switches
    HIGHER_CORRS =         False # Plots q_2NN and q_3NN as well. NOTE: these are not implemented in paper-quality, just in the legend
    SAME_XAXIS_RANGE =     True # If True, then zooming/panning keeps all x-axes in the same range throughout the figure
    DOWNARROW =            False # If true, all displayed kBT values are shown with an arrow.
    SHOW_STATES =          True # Whether to show the states at t_max.
    # Some useful parameters for all figures
    d = {"E_MC_ratio": lambda E_MC: r"NN magnetostatic coupling $E_\mathrm{MC}$" + f"\n{E_MC:g}" + r"$\,k_\mathrm{B}\mathrm{T}$",
        "E_EA_ratio": lambda E_EA: r"Net OOP anisotropy $E_\mathrm{EA}$" + f"\n{E_EA:g}" + r"$\,k_\mathrm{B}\mathrm{T}$",
        "J_ratio": lambda J: "Exchange interaction\n" + r"$J = " + f"{J:g} k_BT$",
        "gradient": lambda g: f"gradient = {g*100:g}%"}
    varx_text, vary_text = d[varx_name], d[vary_name]
    pad = 0
    fontsize_headers = 11
    fontsize_labels = 10
    fontsize_ticks = 9
    linewidths = 1
    # Colors
    C0_dark, C1_dark = colors.rgb_to_hsv(colors.to_rgb("C0")), colors.rgb_to_hsv(colors.to_rgb("C1"))
    C0_dark[-1] *= .7
    C1_dark[-1] *= .85
    C0_dark, C1_dark = colors.to_hex(colors.hsv_to_rgb(C0_dark)), colors.to_hex(colors.hsv_to_rgb(C1_dark))
    # Setup figure
    figsize = (thesis_utils.page_width, thesis_utils.page_width*.8)
    fig, axes = plt.subplots(vary_values.size, varx_values.size, figsize=figsize, sharex=SAME_XAXIS_RANGE, sharey=True)
    fig.subplots_adjust(left=0.16, top=0.88, bottom=0.09, wspace=0.12, hspace=0.2) #! Already here, because insets require knowledge of final aspect ratio
    fig.suptitle(varx_text(0).split("\n")[0] if len(varx_text(0).split("\n")) > 1 else "", fontsize=fontsize_headers, x=0.53) # 0.03 to the right to be centered on the subplots
    fig.supylabel(vary_text(0).split("\n")[0] if len(vary_text(0).split("\n")) > 1 else "", fontsize=fontsize_headers, x=0.005, y=0.47) # 0.005 to not be right at the edge but not too far to the right either
    fig.supxlabel("Elapsed time (s)", fontsize=fontsize_labels, x=0.53)
    
    insets_x = np.array([[0.8, 0.6, 0.8], [0.45, 0.35, 0.8], [0.8, 0.8, 0.8]])
    insets_y = np.ones((3,3))*.5
    insets_d = 0.4
    
    for i, vary in enumerate(vary_values):
        for j, varx in enumerate(varx_values):
            # Setup axes
            ax: plt.Axes = np.asarray(axes).flat[i*len(varx_values) + j]
            ax_right: plt.Axes = ax.twinx()
            ax.set_xscale('log')
            ax_right.spines['left'].set_color(C0_dark)
            ax.tick_params(axis='y', colors=C0_dark, labelsize=fontsize_ticks)
            ax.tick_params(axis='x', labelsize=fontsize_ticks)
            ax_right.spines['right'].set_color(C1_dark)
            ax_right.tick_params(axis='y', colors=C1_dark, labelsize=fontsize_ticks)
            for a in [ax, ax_right]: # Thicker lines
                for axis in ['top','bottom','left','right']: a.spines[axis].set_linewidth(linewidths)
                a.tick_params(width=linewidths)
            if i == 0:
                text = varx_text(varx).split("\n")[-1]
                ax.annotate(text + "\n" + r"$\downarrow$"*DOWNARROW, fontsize=fontsize_headers, xy=(0.5, 1), xytext=(0, 2*pad),
                            xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
            if j == 0:
                text = vary_text(vary).split("\n")[-1]
                ax.annotate(text + "\n" + r"$\downarrow$"*DOWNARROW, fontsize=fontsize_headers, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 4*pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points', ha='center', va='center', rotation=90)
                if i == len(vary_values)//2: ax.set_ylabel(r"Average magnetisation $m_\mathrm{avg}$", color=C0_dark, fontsize=fontsize_labels)
                else: ax.set_ylabel(r"Average magnetisation $m_\mathrm{avg}$", alpha=0, fontsize=fontsize_labels) # Otherwise the row-labels are no longer aligned with eachother
            elif j == varx_values.size - 1:
                if i == len(vary_values)//2: ax_right.set_ylabel(r"Local AFM parameter $q_\mathrm{NN}$", color=C1_dark, fontsize=fontsize_labels, rotation=270, labelpad=2*pad+6, va='baseline')
            if j < varx_values.size - 1:
                ax_right.set_yticklabels([])
            if CLIP_RANGES:
                ax.set_xticks([1e-9, 1e-3, 1e3])
                ax.set_xticks([10**i for i in range(-10, 7)], minor=True)
                ax.tick_params('x', length=5, width=2, which='major')
                ax.tick_params('x', length=3, width=1, which='minor')
                ax.xaxis.set_minor_formatter(ticker.NullFormatter())
            # Plot the data
            x_vals = times[:,i,j,:]
            m_avg = m_avgs[:,i,j,:]
            q_NN = (1 - corr_NN[:,i,j,:])/2
            q_2NN = (1 - corr_2NN[:,i,j,:])/2
            q_3NN = (1 - corr_3NN[:,i,j,:])/2
            if CLIP_RANGES: m_avg = np.abs(m_avg) # Because we want it to be in range [0,1]. No need to do this for metric 2 since it is always in [0,1] or [-1,1] and the axis gets updated accordingly.
            
            def mean_std(t_plot, sample_times, sample_values):
                print(sample_times.shape)
                n_samples, T = sample_times.shape
                M = t_plot.shape[0]
                mean, std, perc_low, perc_high = np.empty(M), np.empty(M), np.empty(M), np.empty(M)
                for i, t in enumerate(t_plot):
                    indices = np.argmin(np.abs(sample_times - t), axis=1)
                    vals = sample_values[np.arange(n_samples), indices]
                    mean[i], std[i] = np.mean(vals), np.std(vals)
                    perc_low[i], perc_high[i] = np.percentile(vals, 1), np.percentile(vals, 99)
                return mean, std, perc_low, perc_high

            ds = [{"var": m_avg, "label": r"$m_\mathrm{avg}$", "color": "C0"}, {"var": q_NN, "label": r"$q_\mathrm{NN}$", "color": "C1"}]
            if HIGHER_CORRS: ds += [{"var": q_2NN, "label": r"$q_\mathrm{2NN}$", "color": "C3"}, {"var": q_3NN, "label": r"$q_\mathrm{3NN}$", "color": "C6"}]
            X = np.logspace(np.log10(np.min(x_vals)), np.log10(np.max(x_vals)), 200)
            lns: list[plt.Line2D] = []
            for d in ds:
                mean, std, perc_low, perc_high = mean_std(X, x_vals, d["var"])
                lns.append(ax.plot(X, mean, label=d["label"], color=d["color"])[0])
                ax.fill_between(X, mean - std, mean + std, color=d["color"], edgecolor="none", alpha=0.5)
                ax.fill_between(X, perc_low, perc_high, color=d["color"], edgecolor="none", alpha=0.5)
            if CLIP_RANGES:
                for a in [ax, ax_right]:
                    a.set_xlim(xmin=max(1e-10, np.min(x_vals)/2), xmax=min(t_max, np.max(x_vals)*2))
                    a.set_ylim([0,1])

            t = ax.text(0, 1, f" \n  {i*len(varx_values) + j + 1:d}  ",
                        bbox=dict(boxstyle='square,pad=0', facecolor='#000', edgecolor='#000'), color='w',
                        fontsize=fontsize_ticks*.85, fontfamily='DejaVu Sans', weight='bold', linespacing=0.01, ha='left', va='bottom', transform=ax.transAxes, zorder=-1)
            if any(phase_1_finished := (np.mean(m_avg, axis=0) < 0.2)): # Don't draw the green shading if m_avg doesn't even reach the threshold
                plt.axvspan(np.mean(x_vals, axis=0)[np.argmax(phase_1_finished)], t_max, facecolor='g', alpha=0.1)
            
            ## Add final magnetisation state to this axis
            if SHOW_STATES:
                x, y = insets_x[i,j], insets_y[i,j]
                cutout_data = final_states[i,j,:,:]
                inset_ax(ax, x, y, insets_d, cutout_data)

    # Finish the figure
    if HIGHER_CORRS: # Then a legend is advised
        fig.legend(lns, [l.get_label() for l in lns], loc='upper left', ncol=2)

    ## SAVE PLOT
    hotspice.utils.save_results(figures={"OOP_relaxation": fig}, outdir=data_dir, copy_script=False)


def inset_ax(ax: plt.Axes, x, y, d, state: np.array):
    # Aspect ratio
    fig = ax.figure
    figsize = fig.get_size_inches()
    ar = figsize[1]/figsize[0]
    dx = d*ar if ar < 1 else d
    dy = d if ar < 1 else d/ar
    
    # To axis size
    pos = ax.get_position()  # Bbox with (x0, y0, width, height) in figure coords
    dx *= pos.height
    dy *= pos.height

    # Compute the lower left corner of the inset bounding box
    center_fig = (pos.x0 + x * pos.width, pos.y0 + y * pos.height)
    inset_left = center_fig[0] - dx / 2.
    inset_bottom = center_fig[1] - dy / 2.
    inset_rect = [inset_left, inset_bottom, dx, dy]

    # Create the inset axis at the computed location and set its aspect ratio to equal
    inax: plt.Axes = fig.add_axes(inset_rect)
    inax.set_aspect('equal')
    inax.imshow(state, vmin=0, vmax=1, cmap=colors.ListedColormap(['black', 'none']), aspect='auto', origin='lower', alpha=1)
    inax.set_xticks([])
    inax.set_yticks([])
    # ax.add_patch(plt.Rectangle((x-dx, y-dy), 2*dx, 2*dy, fill=False, color="gray", linewidth=1))
    return inax


if __name__ == "__main__":
    # run(MCsweeps=40, samples=200)
    plot(thesis_utils.get_last_outdir())
    # thesis_utils.replot_all(plot)
