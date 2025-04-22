import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from joblib import Parallel, delayed
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from psutil import cpu_count

import sys
sys.path.append(os.path.abspath(".")) # Otherwise local imports (like thermally_active_ASI) can not be resolved
from thermally_active_ASI_old import get_thermal_mm
from signals import *
import hotspice
xp = hotspice.xp

import thesis_utils
DEV = False # Set to False when this script works well


def determine_MSE(return_experiment: bool = False,
                  DD_ratio: float = 2.5, J_ratio: float = 0.0, E_B_ratio: float = 20., E_B_std: float = 0.05,
                  size: int = 20, gradient: float = 0.1, res_x: int = None,
                  signal: Signal = Signals.SINE, target: Signal = Signals.SAW, offset: float = 0,
                  frequency_log: float = 2.9, magnitude: float = 3.4e-4, magnitude_min: float = 0,
                  num_periods: int = None):
    """ Determines the MSE for a given system.
        If <return_experiment> is True, the experiment is returned (instead of MSE) without averaging (<n_avg> is ignored).
        (Unused parameters that could be specified, but significantly change the system: DD_exponent, J_ratio)
        
        When optimizing, the only parameters that make sense to vary during optimization are:
            DD_ratio, (E_B_ratio|frequency_log), E_B_std, gradient, magnitude
        because we know that increasing size means better performance, res_x should be min(size, 30) to avoid overfitting,
        and the signal/target/offset should be fixed because we are optimizing for a given signal transformation.
    """
    if res_x is None: res_x = size if size <= 20 else 10
    if num_periods is None: num_periods=20 if signal is not Signals.MACKEYGLASS else 100
    mm = get_thermal_mm(E_B_ratio=E_B_ratio, DD_ratio=DD_ratio, J_ratio=J_ratio, E_B_std=E_B_std, gradient=gradient, size=size)
    experiment = SignalTransformationExperiment(mm, signal, target, frequency=10**frequency_log, magnitude=(magnitude_min, magnitude), offset=offset, res_x=res_x, res_y=1, use_constant=True)
    experiment.run(num_periods=num_periods, samples_per_period=20, verbose=True)
    experiment.calculate_all()
    return experiment if return_experiment else experiment.MSE_reservoir


def run_optimization(file: str, n_iter: int = 8, n_avg: int = 5, static_params: dict = None, variables: dict = None, initial_guess: dict = None, ONLY_REDRAW_BEST: bool = False, save: bool = True):
    if any(k in static_params.keys() for k in variables.keys()):
        raise ValueError("Duplicate keys between static_params and variables.")
    filebase = file.removesuffix('.json')
    load_previous = os.path.exists(filebase + '.json') if filebase is not None else False
    os.makedirs(os.path.dirname(filebase), exist_ok=True)
    
    def determine_MSE_wrapper(return_experiment: bool = False, **kwargs):
            print(f"Running with parameters {kwargs}")
            if return_experiment: return determine_MSE(return_experiment=True, **static_params, **kwargs)
            else: return 1/np.mean([determine_MSE(**static_params, **kwargs) for _ in range(n_avg)]) # 1/MSE because MSE must be minimized but optimizer maximizes this output value
        
    if not ONLY_REDRAW_BEST:
        import bayes_opt
        # BayesianOptimization maximizes return value, but we want to minimize MSE, so inverting MSE does the trick.
        optimizer = bayes_opt.BayesianOptimization(
            f=determine_MSE_wrapper,
            pbounds=variables,
            verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
            random_state=np.random.randint(1000000),
            allow_duplicate_points=True # Shouldn't happen, but it doesn't hurt either
        )

        if load_previous:
            try:
                bayes_opt.util.load_logs(optimizer, logs=[filebase + ".json"])
                print(f"New optimizer is now aware of {len(optimizer.space)} points.")
            except FileNotFoundError:
                print("No previous logs found.")
                load_previous = False
    
        logger = bayes_opt.logger.JSONLogger(path=filebase, reset=False)
        optimizer.subscribe(bayes_opt.Events.OPTIMIZATION_STEP, logger)

        if not load_previous and initial_guess is not None:
            optimizer.probe(params=initial_guess)
        optimizer.maximize(init_points=0 if load_previous else 5, n_iter=n_iter)
        best_guess = optimizer.max
    else:
        static_params.setdefault('num_periods', 100 if DEV else 1000) # To get train and test very close to each other
        parameters, data = hotspice.utils.load_results(os.path.dirname(filebase))
        best_guess = data['best_guess']
    
    ## Recalculate the best guess to save more details about it
    print(f"Best guess: {best_guess}")
    best_experiments: list[SignalTransformationExperiment] = [determine_MSE_wrapper(return_experiment=True, **best_guess['params']) for _ in range(n_avg)]
    results = {
        "MSE_reservoir_train":        [e.MSE_reservoir_train for e in best_experiments],
        "MSE_reservoir_test": (MSE := [e.MSE_reservoir       for e in best_experiments]),
        "MSE_rawinput_train":         [e.MSE_rawinput_train  for e in best_experiments],
        "MSE_rawinput_test":          [e.MSE_rawinput        for e in best_experiments]
        }
    if not ONLY_REDRAW_BEST:
        data = {"best_guess": best_guess | {"results": results}, "all_iterations": optimizer.res}
        parameters = {"n_avg": n_avg, "static_params": static_params, "variables": variables}
    else:
        data["best_guess"]["results"] = results
    best_experiment = best_experiments[np.argmin(MSE)]
    if save:
        hotspice.utils.save_results(
            outdir=os.path.dirname(filebase),
            parameters=parameters,
            data=data,
            figures=best_experiment.plot(
                N_cycles=20 if static_params.get('signal', Signals.SINE) is Signals.MACKEYGLASS else 10,
                show=False
            )
        )
    return best_experiment


def multithreaded_MG_offsets(offsets=None):
    if offsets is None: offsets = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4]
    
    def runner(offset):
        print('running offset', offset)
        dir_base = os.path.splitext(__file__)[0]
        dir_end = f"offset{offset:2.1f}/iterations.json"
        run_optimization(
            n_iter=100, n_avg=5, ONLY_REDRAW_BEST=True,
            file=f"{dir_base}/MG_11x11_noGrad/{dir_end}",
            static_params={'size': 11, 'res_x': 11, 'E_B_ratio': 20, 'DD_ratio': 2.5, 'gradient': 0,
                           'signal': Signals.MACKEYGLASS, 'target': Signals.MACKEYGLASS, 'offset': offset},
            variables={'frequency_log': (2,5), 'magnitude': (0, 0.7e-3), 'magnitude_min': (-0.7e-3, 0)},
            initial_guess={'frequency_log': 2, 'magnitude': 2e-4, 'magnitude_min': -3.5e-4}
        )
        run_optimization(
            n_iter=100, n_avg=5, ONLY_REDRAW_BEST=True,
            file=f"{dir_base}/MG_11x11/{dir_end}",
            static_params={'size': 11, 'res_x': 11, 'E_B_ratio': 20, 'DD_ratio': 2.5,
                           'signal': Signals.MACKEYGLASS, 'target': Signals.MACKEYGLASS, 'offset': offset},
            variables={'frequency_log': (2,5), 'magnitude': (0, 0.7e-3), 'magnitude_min': (-0.7e-3, 0), 'gradient': (0, 0.3)},
            initial_guess={'frequency_log': 2, 'magnitude': 2e-4, 'magnitude_min': -3.5e-4, 'gradient': .1}
        )
        run_optimization(
            n_iter=100, n_avg=5, ONLY_REDRAW_BEST=True,
            file=f"{dir_base}/MG_20x20_resx10/{dir_end}",
            static_params={'size': 20, 'res_x': 10, 'E_B_ratio': 20, 'DD_ratio': 2.5,
                           'signal': Signals.MACKEYGLASS, 'target': Signals.MACKEYGLASS, 'offset': offset},
            variables={'frequency_log': (2,5), 'magnitude': (0, 0.7e-3), 'magnitude_min': (-0.7e-3, 0), 'gradient': (0, 0.3)},
            initial_guess={'frequency_log': 2, 'magnitude': 2e-4, 'magnitude_min': -3.5e-4, 'gradient': .1}
        )
        print(f"Finished optimization of offset={offset}.")

    # NOTE: if this goes woefully slow, comment out the time.sleep in joblib/parallel.py in method Parallel._retrieve()
    Parallel(n_jobs=cpu_count(logical=False), backend='loky')(delayed(runner)(offset) for offset in offsets) # 'loky' overcomes GIL


def plot_MSE_offset(ax: plt.Axes, params: list, data: list, legend_cols: int = 0, fig: plt.Figure = None):
    """ <params> is a list of dictionaries, containing the information of params.json in each of the offset directories. """
    fontsize_axes = thesis_utils.fs_small
    fontsize_legend = thesis_utils.fs_small
    
    offsets = np.asarray([p['static_params']['offset'] for p in params])
    results = [d['best_guess']['results'] for d in data]
    LoD_to_DoL = lambda LoD: {k:[D[k] for D in LoD] for k in LoD[0].keys()} # list-of-dicts to dict-of-lists
    results_avg = LoD_to_DoL([{k: np.mean(v) for k,v in result.items()} for result in results])
    results_std = LoD_to_DoL([{k: np.std(v, ddof=True) for k,v in result.items()} for result in results])
    
    ax.plot(offsets, results_avg["MSE_rawinput_test"], "rh-", label="Input scaling (test)")
    ax.plot(offsets, results_avg["MSE_rawinput_train"], "r:", alpha=0.5, label="Input scaling (train)")
    ax.errorbar(offsets, results_avg["MSE_reservoir_test"], yerr=results_std["MSE_reservoir_test"], marker="H", color="b", label=f"Reservoir (test)")
    ax.errorbar(offsets, results_avg["MSE_reservoir_train"], yerr=results_std["MSE_reservoir_train"], linestyle=":", color='b', alpha=0.5, label=f"Reservoir (train)")
    ax.set_ylabel("MSE", fontsize=fontsize_axes)
    ax.set_xlabel(r"Offset $h/f_\mathrm{MG}$", fontsize=fontsize_axes)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='both', labelsize=fontsize_axes)
    if len(idx0 := np.where(offsets == 0)[0]): # offset=0 shows the minimum possible MSE, because there only thermal noise creates discrepancy
        ax.axhline(results_avg["MSE_reservoir_test"][idx0[0]], linestyle=':', color='lightgray')
        
    ## Create legend
    if legend_cols:
        line_input = Line2D([], [], color='r', linestyle='none', marker='h', markersize=10, label="Input scaling")
        line_reservoir = Line2D([], [], color='b', linestyle='none', marker='H', markersize=10, label="Reservoir")
        line_test = Line2D([], [], color='k', linestyle='-', label='Test set')
        line_train = Line2D([], [], color='k', linestyle=':', alpha=0.5, label='Train set')
        # line_test = ax.errorbar([], [], yerr=[], color='k', linestyle='-', label='Test set')
        # line_train = ax.errorbar([], [], yerr=[], color='k', linestyle=':', label='Train set')
        ax.legend(handles=[line_input, line_reservoir, line_test, line_train], ncol=legend_cols, loc='lower center', bbox_to_anchor=(0.5, 1.2), fontsize=fontsize_legend)

    if fig is not None: # If a figure is passed, compute and return the figure‐coordinate of the rightmost test‐curve point.
        return tuple(fig.transFigure.inverted().transform(ax.transData.transform((offsets[-1], results_avg["MSE_reservoir_test"][-1]))))


def plot_waveform(exp: SignalTransformationExperiment, ax: plt.Axes, train=False, N_cycles: int = None, show_signal=True, samples_per_period=None, show_MSE_raw: bool = True):
    ## Plot font sizes
    fontsize_axes = thesis_utils.fs_small
    fontsize_legend = thesis_utils.fs_small
    alpha_signal, alpha_target = 0.5, 1.0
    
    ## Determine the best-looking part of the data to show in the plot
    if samples_per_period is None: samples_per_period = np.argmax((exp.t - np.min(exp.t)) >= 1/exp.frequency)
    N_idx = np.argmax((exp.t - np.min(exp.t)) >= (N_cycles*.8)/exp.frequency) # Number of indices per N_cycles*0.8 (legend hides 20% of plot)
    N_idx_full = np.argmax((exp.t - np.min(exp.t)) >= N_cycles/exp.frequency) # Number of indices per N_cycles (to know the full plot length)
    offset = int(samples_per_period/7)
    # Train set:
    MSEs = [exp.MSE(exp.fit_reservoir[i:i+N_idx], exp.target_train[i:i+N_idx]) for i in range(exp.target_train.size - N_idx_full)]
    best_start_idx = np.argmin(MSEs[offset::samples_per_period])*samples_per_period + offset
    train_slice = (best_start_idx, best_start_idx + N_idx_full)
    # Test set:
    MSEs = [exp.MSE(exp.prediction_reservoir[i:i+N_idx], exp.target_test[i:i+N_idx]) for i in range(exp.target_test.size - N_idx_full)]
    best_start_idx = np.argmin(MSEs[offset::samples_per_period])*samples_per_period + offset + exp.target_train.size
    test_slice = (best_start_idx, best_start_idx + N_idx_full)
    
    ## Draw the plots
    t_rescaled, t_unit = hotspice.utils.appropriate_SIprefix(exp.t)
    t_scale = 10**hotspice.utils.SIprefix_to_magnitude[t_unit]
    ax.set_xlabel(f"Time ({t_unit}s)", fontsize=fontsize_axes, labelpad=-0.06)
    ax.set_ylim([-0.1, 1.1])
    if not train:
        t = exp.get_test(t_rescaled)
        if N_cycles is not None:
            ax.set_xlim(exp.t[test_slice[0]]/t_scale, exp.t[test_slice[1]]/t_scale)
            # ax1.set_xlim([np.min(t), min(np.max(t), np.min(t) + N_cycles/self.frequency/t_scale)])
        line_target, = ax.plot(t, exp.get_test(exp.target_values), "k--", alpha=alpha_target)
        line_input_pred, = ax.plot(t, exp.prediction_rawinput, "r", alpha=0.5)
        if show_MSE_raw: line_input_pred.set_label(f"{1/exp.MSE_rawinput:.2f}")
        line_reservoir, = ax.plot(t, exp.prediction_reservoir, "dodgerblue", label=f"{1/exp.MSE_reservoir:.2f}")
        if show_signal: line_input, = ax.plot(t, exp.signal_test, "k:", alpha=alpha_signal)
        ax.legend(title=r"1/MSE", fontsize=fontsize_legend, loc='lower right', ncol=1, title_fontproperties={'weight':'bold', 'size': fontsize_legend-2}, handlelength=1, handletextpad=0.4)
        ax.tick_params(axis='both', labelsize=fontsize_axes)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    else:
        t = exp.get_train(t_rescaled)
        if N_cycles is not None:
            ax.set_xlim(exp.t[train_slice[0]]/t_scale, exp.t[train_slice[1]]/t_scale)
            # ax2.set_xlim([np.min(t), min(np.max(t), np.min(t) + N_cycles/self.frequency/t_scale)])
        line_target, = ax.plot(t, exp.get_train(exp.target_values), "k--", alpha=alpha_target)
        line_input_pred, = ax.plot(t, exp.fit_rawinput, "r", alpha=0.5)
        if show_MSE_raw: line_input_pred.set_label(f"{1/exp.MSE_rawinput:.2f}")
        line_reservoir, = ax.plot(t, exp.fit_reservoir, "dodgerblue", label=f"{1/exp.MSE(exp.fit_reservoir, exp.target_train):.2f}")
        if show_signal: line_input, = ax.plot(t, exp.signal_train, "k:", alpha=alpha_signal)
        ax.legend(title=r"1/MSE", fontsize=fontsize_legend, loc='lower right', ncol=1, title_fontproperties={'weight':'bold', 'size': fontsize_legend-2}, handlelength=1, handletextpad=0.4)
        ax.tick_params(axis='both', labelsize=fontsize_axes)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    
    if show_signal:
        handles = [line_input, line_target, line_input_pred, line_reservoir]
        labels = ["Input signal", "Target", "Prediction with input only", "Prediction with ASI"]
    else:
        handles = [line_target, line_input_pred, line_reservoir]
        labels = ["Target", "Prediction with input only", "Prediction with ASI"]
    return handles, labels


def plot():
    folders = ["MG_11x11_noGrad", "MG_11x11", "MG_20x20_resx10"]
    offsets = [0., 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    fs_title = thesis_utils.fs_large
    arrow_kwargs = dict(arrowstyle='-|>', mutation_scale=14, linewidth=2, shrinkA=5, shrinkB=1, color='k')
    
    dir_base = os.path.splitext(__file__)[0]
    fig = plt.figure(figsize=(thesis_utils.page_width, 4.5))
    gs = fig.add_gridspec(2, 6, height_ratios=[2,1.5], hspace=0.9, right=0.99, top=0.87, bottom=0.08, left=0.1)
    axMSE1 = fig.add_subplot(gs[0,0:2])
    axMSE2 = fig.add_subplot(gs[0,2:4])
    axMSE3 = fig.add_subplot(gs[0,4:6])
    axwave1 = fig.add_subplot(gs[1,0:3])
    axwave2 = fig.add_subplot(gs[1,3:6])
    
    thesis_utils.label_ax(axMSE1, 0, offset=(-0.2, 0.3), ha="right", va="baseline")
    thesis_utils.label_ax(axwave1, 1, offset=(-0.2/3*2 + 0.006, 0.3), ha="right", va="baseline")
    
    axMSE1.set_ylabel("1/MSE")
    axMSE1.set_title(r"$11 \times 11$ no gradient", fontsize=fs_title)
    params, data = zip(*(hotspice.utils.load_results(f"{dir_base}/{folders[0]}/offset{offset:.1f}") for offset in offsets))
    coord_top = plot_MSE_offset(axMSE1, params, data, fig=fig)
    arrow_end_y = axwave1.get_position().y1
    fig.patches.append(FancyArrowPatch(coord_top, (coord_top[0], arrow_end_y), transform=fig.transFigure, **arrow_kwargs))
    
    axMSE2.set_yticklabels([])
    axMSE2.set_title(r"$11 \times 11$ with gradient", fontsize=fs_title)
    params, data = zip(*(hotspice.utils.load_results(f"{dir_base}/{folders[1]}/offset{offset:.1f}") for offset in offsets))
    plot_MSE_offset(axMSE2, params, data, legend_cols=4)
    axMSE2.set_ylabel("")
    
    axMSE3.set_yticklabels([])
    axMSE3.set_title(r"$20 \times 20$ with gradient", fontsize=fs_title)
    params, data = zip(*(hotspice.utils.load_results(f"{dir_base}/{folders[2]}/offset{offset:.1f}") for offset in offsets))
    coord_top = plot_MSE_offset(axMSE3, params, data, fig=fig)
    axMSE3.set_ylabel("")
    arrow_end_y = axwave2.get_position().y1
    fig.patches.append(FancyArrowPatch(coord_top, (coord_top[0], arrow_end_y), transform=fig.transFigure, **arrow_kwargs))
    
    ## WAVEFORMS
    offset = 1.4
    N_cycles = 20
    dir_end = f"offset{offset:2.1f}/iterations.json"
    
    ## LEFT WAVEFORM (11x11 no gradient)
    # Load experiment, or create and save experiment
    exp: SignalTransformationExperiment = determine_MSE(return_experiment=True, num_periods=10)
    filename = os.path.join(dir_base, f"{folders[0]}_waveform.pkl")
    if os.path.exists(filename):
        with open(filename, 'rb') as infile:
            d = pickle.load(infile)
            exp.N, exp.t, exp.frequency, exp.signal_values, exp.target_values, exp.readout_values = d["N"], d["t"], d["frequency"], d["signal_values"], d["target_values"], d["readout_values"]
            exp.train_estimator()
    else:
        exp: SignalTransformationExperiment = run_optimization(
            n_iter=100, n_avg=5, ONLY_REDRAW_BEST=True, save=False,
            file=f"{dir_base}/{folders[0]}/{dir_end}",
            static_params={'size': 11, 'res_x': 11, 'E_B_ratio': 20, 'DD_ratio': 2.5, 'gradient': 0,
                        'signal': Signals.MACKEYGLASS, 'target': Signals.MACKEYGLASS, 'offset': offset},
            variables={'frequency_log': (2,5), 'magnitude': (0, 0.7e-3), 'magnitude_min': (-0.7e-3, 0)},
            initial_guess={'frequency_log': 2, 'magnitude': 2e-4, 'magnitude_min': -3.5e-4}
        )
        with open(filename, 'wb') as outfile:
            d = {"N": exp.N, "t": exp.t, "frequency": exp.frequency, "signal_values": exp.signal_values, "target_values": exp.target_values, "readout_values": exp.readout_values}
            pickle.dump(d, outfile)
    handles, labels = plot_waveform(exp, axwave1, N_cycles=N_cycles, show_MSE_raw=True)
    
    ## RIGHT WAVEFORM (20x20 with gradient)
    axwave2.set_yticklabels([])
    # Load experiment, or create and save experiment
    exp: SignalTransformationExperiment = determine_MSE(return_experiment=True, num_periods=10)
    filename = os.path.join(dir_base, f"{folders[2]}_waveform.pkl")
    if os.path.exists(filename):
        with open(filename, 'rb') as infile:
            d = pickle.load(infile)
            exp.N, exp.t, exp.frequency, exp.signal_values, exp.target_values, exp.readout_values = d["N"], d["t"], d["frequency"], d["signal_values"], d["target_values"], d["readout_values"]
            exp.train_estimator()
    else:
        exp: SignalTransformationExperiment = run_optimization(
            n_iter=100, n_avg=5, ONLY_REDRAW_BEST=True, save=False,
            file=f"{dir_base}/{folders[2]}/{dir_end}",
            static_params={'size': 20, 'res_x': 10, 'E_B_ratio': 20, 'DD_ratio': 2.5,
                        'signal': Signals.MACKEYGLASS, 'target': Signals.MACKEYGLASS, 'offset': offset},
            variables={'frequency_log': (2,5), 'magnitude': (0, 0.7e-3), 'magnitude_min': (-0.7e-3, 0), 'gradient': (0, 0.3)},
            initial_guess={'frequency_log': 2, 'magnitude': 2e-4, 'magnitude_min': -3.5e-4, 'gradient': .1}
        )
        with open(filename, 'wb') as outfile:
            d = {"N": exp.N, "t": exp.t, "frequency": exp.frequency, "signal_values": exp.signal_values, "target_values": exp.target_values, "readout_values": exp.readout_values}
            pickle.dump(d, outfile)
    plot_waveform(exp, axwave2, N_cycles=N_cycles, show_MSE_raw=False)
    
    ## Legend
    fontsize_legend = thesis_utils.fs_small
    bottom = axwave1.get_position().y1
    left = axMSE1.get_position().x1
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(left, bottom), ncol=2, fontsize=fontsize_legend)

    hotspice.utils.save_results(figures={"MG": fig}, copy_script=False, timestamped=False, outdir=os.path.splitext(__file__)[0])

if __name__ == "__main__":
    # multithreaded_MG_offsets() # Used to generate the BO iterations
    plot()
