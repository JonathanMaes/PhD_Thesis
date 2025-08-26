""" IN THIS SCRIPT, THE TIME-EVOLUTION WHEN APPLYING A UNIFORM STIMULUS IS SIMULATED.
    USAGE: change simulation parameters under the heading "## DEFINE PARAMETERS"
           (varx_name and vary_name can be any of "frequency", "dutycycle" or "magnitude")
           change plotting preferences under the heading "# Plot parameters/switches"
           All other parts of the code will do their job based on those settings.

    We take an OOP_Square ASI with the following fixed parameters:
        - E_B = 10 kBT,
        - NN DD = 2.5 kBT,
    because this combination has a relaxation taking place over very few OoM, while still having a decent DD interaction.
    
    The main parameters that are varied within this simulation are:
        - frequency: how fast subsequent pulses are applied
        - magnitude: the magnitude of the external magnetic field used to apply the input
        - duty cycle: which fraction of the input period is used to actually apply the field
    This will allow us to make a map of which combinations create useful behavior in the ASI.
    All other ASI parameters can be adjusted, but are not 'swept' throughout this simulation.
    This way, the ASI is always the same, yielding a consistent baseline.
    The system parameters were chosen based on results from relaxation_EEA_EMC.py.
    
    => OUTPUT: plots of NN correlation and m_avg as a function of time, for various (frequency, magnitude) combinations.
"""
import copy
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

import hotspice
import thesis_utils

from thermally_active_ASI import get_thermal_mm, UniformInputter


def run(max_inputs: int = 25, MCsweeps: float = 4., samples: int = 20):
    """ After `MCsweeps*mm.n` switches or `max_inputs` input values, the time-evolution is stopped.
        For each combination of varx and vary, `samples` relaxations will be performed.
    """
    mm_kwargs = {'E_EA_ratio': 20, 'E_MC_ratio': 2.5, 'E_B_std': 0.05, 'size': 11}
    inputter_kwargs = {'datastream': hotspice.io.RandomScalarDatastream(), 'dutycycle': 1, 'n': 2}
    
    # Sweep-related params
    varx_name, vary_name = "magnitude", "frequency"
    varx_values = [0.0002, 0.0003, 0.0004] # Magnitude [T]
    vary_values = [1e3, 1e5] # Frequency [Hz]

    ## RUN SIMULATIONS
    # Sanitize input
    if varx_name not in (allowed_varx_vary := ("magnitude", "dutycycle", "frequency")) or vary_name not in allowed_varx_vary:
        raise ValueError(f"varx_name and vary_name must be any of {allowed_varx_vary}")
    for name in (varx_name, vary_name): mm_kwargs.pop(name, None) # None avoids KeyError
    vary_values, varx_values = np.asarray(vary_values), np.asarray(varx_values)
    # Prepare and execute sweep over (varx, vary)
    samples, N_switches = int(samples), int(MCsweeps*get_thermal_mm(**mm_kwargs).n)
    times = np.full((samples, vary_values.size, varx_values.size, N_switches), np.nan) # NaNs are not plotted, so we pre-fill with those
    m_avgs = np.copy(times)
    corr_NN, corr_2NN, corr_3NN = np.copy(times), np.copy(times), np.copy(times)
    input_values = np.full(max_inputs, np.nan)
    for i, vary in enumerate(vary_values):
        for j, varx in enumerate(varx_values):
            print(i, j)
            inputter =  UniformInputter(**{varx_name: varx, vary_name: vary}, **inputter_kwargs)
            for s in range(samples):
                mm = get_thermal_mm(**mm_kwargs)
                mm.initialize_m('AFM')
                inputter.datastream.reset_rng() # This allows <samples> to be >1 even if the datastream is random.
                switch, running = 0, True
                for inp in range(max_inputs): # Maximum <max_inputs> input values
                    if not running: break
                    input_values[inp] = inputter.datastream.get_next()[0]
                    for _ in inputter.input_single(mm, input_values[inp], stepwise=True):
                        times[s,i,j,switch] = mm.t
                        m_avgs[s,i,j,switch] = mm.m_avg
                        corr_NN[s,i,j,switch] = mm.correlation_NN()
                        corr_2NN[s,i,j,switch] = mm.correlation_NN(N=2)
                        corr_3NN[s,i,j,switch] = mm.correlation_NN(N=3)
                        if (switch := switch+1) >= N_switches: # Maximum
                            running = False
                            break
    
    ## Save
    hotspice.utils.save_results(
        parameters={
            "varx_name": varx_name, "vary_name": vary_name, "varx_values": varx_values, "vary_values": vary_values,
            "max_inputs": max_inputs, "MCsweeps": MCsweeps, "samples": samples,
            "input_values": input_values
        } | mm_kwargs | inputter_kwargs,
        data={"times": times, "m_avg": m_avgs, "corr_NN": corr_NN, "corr_2NN": corr_2NN, "corr_3NN": corr_3NN, "mm": mm, "inputter": inputter}
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
    input_values = params["input_values"]
    times, m_avgs, corr_NN, corr_2NN, corr_3NN = data["times"], data["m_avg"], data["corr_NN"], data["corr_2NN"], data["corr_3NN"]
    inputter = data["inputter"]
    
    ## Initialise plot
    thesis_utils.init_style(style="default")

    # Plot parameters/switches
    HIGHER_CORRS =         False # Plots q_2NN and q_3NN as well. NOTE: these are not implemented in paper-quality, just in the legend
    SAME_XAXIS_RANGE =     False # If True, then all x-axes have the same range throughout the figure
    X_LOGARITHMIC =        False # If True, the x-axis is plotted on a log scale
    TIME_UNIT_PREFIX =     "m" # The x-axis SI prefix
    # Some useful parameters for all figures
    from hotspice.utils import appropriate_SIprefix
    d = {"frequency": lambda freq: r"Input frequency $f$" + f"\n{appropriate_SIprefix(freq)[0]:.3g} {appropriate_SIprefix(freq)[1]}Hz",
         "dutycycle": lambda duty: r"Duty cycle" + f"\n{duty*100:.0f}%",
         "magnitude": lambda magn: r"Input field magnitude $B$" + f"\n{appropriate_SIprefix(magn)[0]:.3g} {appropriate_SIprefix(magn)[1]}T"}
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
    figsize = (thesis_utils.page_width, thesis_utils.page_width*.54)
    fig, axes = plt.subplots(vary_values.size, varx_values.size, figsize=figsize, sharex=SAME_XAXIS_RANGE, sharey=True)
    axes_flat = np.asarray(axes).flat
    fig.subplots_adjust(left=0.18, top=0.82, bottom=0.14, wspace=0.12, hspace=(hspace:=0.45)) #! Already here, because insets require knowledge of final aspect ratio
    fig.suptitle(varx_text(0).split("\n")[0] if len(varx_text(0).split("\n")) > 1 else "", fontsize=fontsize_headers, x=0.53) # 0.03 to the right to be centered on the subplots
    fig.supylabel(vary_text(0).split("\n")[0] if len(vary_text(0).split("\n")) > 1 else "", fontsize=fontsize_headers, x=0.005, y=0.47) # 0.005 to not be right at the edge but not too far to the right either
    fig.supxlabel(f"Elapsed time [{TIME_UNIT_PREFIX}s]", fontsize=fontsize_labels, x=0.53)

    time_mul = hotspice.utils.SIprefix_to_mul(TIME_UNIT_PREFIX)
    
    for i, vary in enumerate(vary_values):
        for j, varx in enumerate(varx_values):
            # Setup axes
            ax: plt.Axes = axes_flat[i*len(varx_values) + j]
            ax_right: plt.Axes = ax.twinx()
            ax.set_xscale('log' if X_LOGARITHMIC else 'linear')
            ax.set_ylim([0,1])
            ax_right.set_ylim([0,1])
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
                ax.annotate(text + "\n", fontsize=fontsize_headers, xy=(0.5, 1), xytext=(0, 2*pad),
                            xycoords='axes fraction', textcoords='offset points', ha='center', va='baseline')
            if j == 0:
                text = vary_text(vary).split("\n")[-1]
                ax.annotate(text + "\n", fontsize=fontsize_headers, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 4*pad - fontsize_labels*4, 0),
                            xycoords='axes fraction', textcoords='offset points', ha='center', va='center', rotation=90)
                if i == len(vary_values)//2:
                    label = ax.set_ylabel(r"Average magnetisation $m_\mathrm{avg}$", color=C0_dark, fontsize=fontsize_labels, ha="center")
                    if len(vary_values) % 2 == 0: label.set_y(1 + hspace/2)
                else: ax.set_ylabel(r"Average magnetisation $m_\mathrm{avg}$", alpha=0, fontsize=fontsize_labels, ha="center") # Otherwise the row-labels are no longer aligned with eachother
            elif j == varx_values.size - 1:
                if i == len(vary_values)//2:
                    label = ax_right.set_ylabel(r"Local AFM parameter $q_\mathrm{NN}$", color=C1_dark, fontsize=fontsize_labels, rotation=270, labelpad=2*pad+6, va='baseline')
                    if len(vary_values) % 2 == 0: label.set_y(1 + hspace/2)
            if j < varx_values.size - 1:
                ax_right.set_yticklabels([])
            # Plot the data
            x_vals = times[:,i,j,:]/time_mul
            m_avg = m_avgs[:,i,j,:]
            q_NN = (1 - corr_NN[:,i,j,:])/2
            q_2NN = (1 - corr_2NN[:,i,j,:])/2
            q_3NN = (1 - corr_3NN[:,i,j,:])/2
            
            def mean_std(t_plot, sample_times, sample_values):
                print(sample_times.shape)
                n_samples, T = sample_times.shape
                M = t_plot.shape[0]
                mean, std, perc_low, perc_high = np.empty(M), np.empty(M), np.empty(M), np.empty(M)
                for i, t in enumerate(t_plot):
                    indices = np.nanargmin(np.abs(sample_times - t), axis=1)
                    vals = sample_values[np.arange(n_samples), indices]
                    mean[i], std[i] = np.mean(vals), np.std(vals)
                    perc_low[i], perc_high[i] = np.percentile(vals, 1), np.percentile(vals, 99)
                return mean, std, perc_low, perc_high

            ds = [{"var": m_avg, "label": r"$m_\mathrm{avg}$", "color": "C0"}, {"var": q_NN, "label": r"$q_\mathrm{NN}$", "color": "C1"}]
            if HIGHER_CORRS: ds += [{"var": q_2NN, "label": r"$q_\mathrm{2NN}$", "color": "C3"}, {"var": q_3NN, "label": r"$q_\mathrm{3NN}$", "color": "C6"}]
            X = np.logspace(np.log10(np.nanmin(x_vals)), np.log10(np.nanmax(x_vals)), 200) if X_LOGARITHMIC else np.linspace(np.nanmin(x_vals), np.nanmax(x_vals), 200)
            lns: list[plt.Line2D] = []
            for d in ds:
                mean, std, perc_low, perc_high = mean_std(X, x_vals, d["var"])
                lns.append(ax.plot(X, mean, label=d["label"], color=d["color"])[0])
                ax.fill_between(X, mean - std, mean + std, color=d["color"], edgecolor="none", alpha=0.5)
                ax.fill_between(X, perc_low, perc_high, color=d["color"], edgecolor="none", alpha=0.5)
            ax.set_xlim([0, np.max(X)])

            t = ax.text(0, 1, f" \n  {i*len(varx_values) + j + 1:d}  ",
                        bbox=dict(boxstyle='square,pad=0', facecolor='#000', edgecolor='#000'), color='w',
                        fontsize=fontsize_ticks*.85, fontfamily='DejaVu Sans', weight='bold',
                        linespacing=0.01, ha='left', va='bottom', transform=ax.transAxes, zorder=-1)
            
            # Show the normal logarithmic relaxation as it would be without an Inputter
            x_fit = np.sort(x_vals.reshape(-1))
            # if np.allclose(params['E_EA_ratio'], 20) and np.allclose(params['E_MC_ratio'], 2.5):
            #     log_fit = -0.1684*np.log10(t_fit := x_fit*time_mul) - 0.3573 # formula from fitting log-curve to relaxation of EB=20 and DD=2.5
            #     ok = np.logical_and(log_fit >= 0, log_fit <= 1)
            #     ax.plot(x_fit[ok], log_fit[ok], linestyle=':', color=lns[0].get_color(), linewidth=1)
            # Plot where each new input cycle began
            f = varx if varx_name == "frequency" else (vary if vary_name == "frequency" else inputter.frequency_max)
            ax.vlines(np.arange(0, np.nanmax(x_fit), (1/f)/time_mul), 0, 1, linestyles='dotted', colors='gray', alpha=0.2, linewidth=1)
            dt = (1/f)/time_mul
            input_values_padded = np.append(input_values, 0)
            ax.fill_between(np.arange(0, len(input_values_padded)*dt, dt), np.zeros(len(input_values_padded)), input_values_padded, step='post', alpha=0.2, color="grey", edgecolor="none")
            # for posidx, input_value in enumerate(input_values):
            #     ax.fill_between([posidx*dt, (posidx+1)*dt], [0, 0], [input_value, input_value], alpha=0.2, color="grey", edgecolor="none")
            ax.set_xlim([0, 18*dt])
    # Finish the figure
    if HIGHER_CORRS: # Then a legend is advised
        fig.legend(lns, [l.get_label() for l in lns], loc='upper left', ncol=2)

    ## SAVE PLOT
    hotspice.utils.save_results(figures={"OOP_relaxation_input": fig}, outdir=data_dir, copy_script=False)


if __name__ == "__main__":
    # run(MCsweeps=40, samples=200)
    plot(thesis_utils.get_last_outdir())
    # thesis_utils.replot_all(plot)
