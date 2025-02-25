import argparse
import bayes_opt
import colorsys
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle

from joblib import Parallel, delayed
from matplotlib import colormaps
from matplotlib.colors import to_rgb, cnames
from matplotlib.patches import Rectangle
from pathlib import Path
from thermally_active_ASI import get_thermal_mm

import os
os.environ["HOTSPICE_USE_GPU"] = "False"
import hotspice

import thesis_utils

parser = argparse.ArgumentParser()
parser.add_argument("--only-plot", action="store_true", help="Pass this argument to prevent any additional iterations from being calculated, the plots are just recalculated.")
parser.add_argument("--point-dipoles", action="store_true", help="If specified, the improved finite magnet approximation is used, otherwise infinitesimal Ising spins are considered.")
parser.add_argument("--parabolic-barrier", action="store_true", help="If specified, the improved parabolic energy barrier approximation is used, otherwise the simplest barrier calculation is used.")
args = parser.parse_args()

JUST_REPLOT: bool = args.only_plot
FINITE_MAGNET_SIZE: bool = not args.point_dipoles
PARABOLIC_E_B: bool = args.parabolic_barrier
OUTDIR = f"{hotspice.utils.get_caller_script().stem}.out/{'finite_magnets' if FINITE_MAGNET_SIZE else 'infinitesimal_magnets'}-{'parabolic_barrier' if PARABOLIC_E_B else 'simple_barrier'}"
print(f"Saving results to {OUTDIR}")


def compare_exp_sim(exp_data: dict, E_EA: float = 65, E_MC: float = 10, J: float = 0, n_avg: int = 200, magnet_size_ratio: float = 0, plot: bool = False, ax: plt.Axes = None):
    """ Returns a similarity-score if `plot == False`, otherwise returns a Matplotlib `Figure` object. """
    AX_PROVIDED = ax is not None
    T_MAX = np.max(exp_data["times"])*(2 if not plot else 1e10) # To be sure to pass the experimental time enough
    mm_kwargs = {'E_B_std': 0.05, 'size': 11, 'J_ratio': J, "E_EA_ratio": E_EA, "E_MC_ratio": E_MC, "magnet_size_ratio": magnet_size_ratio}
    MCsweeps = 4 # After <MCsweeps*mm.n> Néel switching attempts, the decay is stopped.
    
    ## RUN <samples> SIMULATIONS
    samples, N_switches = int(n_avg), int(MCsweeps*get_thermal_mm(**mm_kwargs).n)
    times = np.full((samples, N_switches), np.nan) # NaNs are not plotted, so we pre-fill with those
    m_avgs, corr_NN, corr_2NN, corr_3NN = np.copy(times), np.copy(times), np.copy(times), np.copy(times)
    all_diffs = []
    for s in range(samples):
        mm = get_thermal_mm(**mm_kwargs) # Need to always make a new mm, in order to get multiple E_B_std samples (otherwise no good averaging in final result)
        mm.params.ENERGY_BARRIER_METHOD = 'parabolic' if PARABOLIC_E_B else 'simple'
        mm.initialize_m('uniform')
        for switch in range(N_switches):
            idx = switch if mm.t <= T_MAX else slice(switch, None) # Set all subsequent values if T_MAX is passed
            times[s,idx] = mm.t
            m_avgs[s,idx] = mm.m_avg
            corr_NN[s,idx] = mm.correlation_NN()
            corr_2NN[s,idx] = mm.correlation_NN(N=2)
            corr_3NN[s,idx] = mm.correlation_NN(N=3)
            if mm.t >= T_MAX: break
            mm.update(t_max=T_MAX) # Do this last, so that the first element in all arrays is the initial value
            
        q_NN, q_2NN, q_3NN = (1 - corr_NN[:,:])/2, (1 - corr_2NN[:,:])/2, (1 - corr_3NN[:,:])/2
        sim_data = {'times': times[s,:],
                    'm_avg': m_avgs[s,:],
                    'q_NN' : q_NN[s,:],
                    'q_2NN': q_2NN[s,:],
                    'q_3NN': q_3NN[s,:]}
        
        # Store the difference between this simulation run and the experiment
        sim_data_at_times = {k: [] for k in sim_data.keys() if k != 'times'}
        for exp_time in exp_data["times"]:
            i = np.argmax(sim_data["times"] - exp_time > 0) # Should be >= 1 due to the fact that the first element in each array is the starting value
            with np.errstate(divide='ignore'):
                ts_log = np.log10(sim_data["times"]) if sim_data["times"][i-1] != 0 else sim_data["times"] # Only take log if there is no zero in a required position
            frac = (np.log10(exp_time) - ts_log[i-1])/(ts_log[i] - ts_log[i-1])
            for k in sim_data_at_times.keys():
                sim_data_at_times[k].append(frac*sim_data[k][i] + (1-frac)*sim_data[k][i-1])
        
        all_diffs.append(np.array([[sim_data_at_times[k][i] - exp_data[k][i] for i, _ in enumerate(exp_data["times"])] for k in sim_data_at_times.keys()]))
    ## Summarize the differences between all the samples' metrics and the experimental metrics into one BO value
    all_diffs = np.asarray(all_diffs) # So an array of shape (n_avg, n_metrics=4, n_times).
    # Calculate the average and std of the differences
    all_diffs_avg = np.mean(all_diffs, axis=0) # Shape (n_metrics=4, n_times)
    all_diffs_std = np.std(all_diffs, axis=0) # Shape (n_metrics=4, n_times)
    
    # Score is how likely this outcome is, assuming a normal distribution
    prob = np.exp(-.5*(all_diffs_avg/all_diffs_std)**2) # No normalization to get area 1, we are just interested in the relative probability of this outcome
    total_prob = np.prod(prob) # Total probability is all probabilities multiplied together
    
    ## Return BO value or plt figure (depending on `plot` argument).
    if not plot:
        with np.errstate(divide='ignore'):
            return np.log10(total_prob) # To make more balanced for the BO procedure
    else:
        # Plotting parameters and switches
        HIGHER_CORRS = True # Plots q_2NN and q_3NN as well. NOTE: these are not implemented in paper-quality, just in the legend
        fontsize_ticks = 20
        linewidths = 1.5
        # Setup figure
        if ax is None:
            figsize = (4, 4)
            fig, ax = plt.subplots(1, 1, figsize=figsize, sharex=True, sharey=True)
        # Plot the averages
        m_avg = np.abs(m_avgs[:,:]) # Because we want it to be in range [0,1].
        q_NN, q_2NN, q_3NN = (1 - corr_NN[:,:])/2, (1 - corr_2NN[:,:])/2, (1 - corr_3NN[:,:])/2
        
        def mean_std(t_plot, sample_times, sample_values):
            n_samples, T = sample_times.shape
            M = t_plot.shape[0]
            mean, std, perc_low, perc_high = np.empty(M), np.empty(M), np.empty(M), np.empty(M)
            for i, t in enumerate(t_plot):
                indices = np.argmin(np.abs(sample_times - t), axis=1)
                vals = sample_values[np.arange(n_samples), indices]
                mean[i], std[i] = np.mean(vals), np.std(vals)
                perc_low[i], perc_high[i] = np.percentile(vals, 1), np.percentile(vals, 99)
            return mean, std, perc_low, perc_high
        
        nonzero_times = times[:,1:]
        ds = [{"var": m_avg, "label": r"$m_\mathrm{avg}$", "color": "C0"}, {"var": q_NN, "label": r"$q_\mathrm{NN}$", "color": "C1"}]
        if HIGHER_CORRS: ds += [{"var": q_2NN, "label": r"$q_\mathrm{2NN}$", "color": "C3"}, {"var": q_3NN, "label": r"$q_\mathrm{3NN}$", "color": "C6"}]
        X = np.logspace(np.log10(np.min(nonzero_times)), np.log10(np.max(nonzero_times)), 200)
        lns: list[plt.Line2D] = []
        fbs: list[plt.PolyCollection] = []
        for d in ds:
            mean, std, perc_low, perc_high = mean_std(X, times, d["var"])
            lns.append(ax.plot(X, mean, label=d["label"], color=d["color"])[0])
            fbs.append(ax.fill_between(X, mean - std, mean + std, color=d["color"], edgecolor="none", alpha=0.5))
            # ax.fill_between(X, perc_low, perc_high, color=d["color"], edgecolor="none", alpha=0.5)

        # Format axes
        ax.set_xscale('log')
        if not AX_PROVIDED:
            for axis in ['x', 'y']: ax.tick_params(axis=axis, labelsize=fontsize_ticks)
            for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(linewidths)
        ax.tick_params(width=linewidths)
        ax.set_xlim(xmin=1e-1, xmax=1e11)
        ax.set_ylim([0,1])
        ax.set_xticks([1, 1e3, 1e6, 1e9])
        ax.set_yticks([0, .5, 1])
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: "10$^\mathregular{" + f"{np.log10(x):.0f}" + "}$")) # For nicer alignment of exponent
        for label in ax.get_xticklabels() + ax.get_yticklabels(): label.set_fontname('Arial') # For compatibility with Alex' plots
        for quantity, color in zip(["m_avg", "q_NN", "q_2NN", "q_3NN"], ["C0", "C1", "C3", "C6"]):
            ax.scatter(exp_data["times"], exp_data[quantity], s=25, color=color, edgecolors="black", zorder=3)
        ax.vlines(exp_data["times"], 0, 1, linestyle=':', color='k')
        ax.hlines(np.linspace(0, 1, 11), 0.1, 1e20, linestyle=':', color='gray', alpha=0.5, linewidths=1)
        if AX_PROVIDED:
            return list(zip(lns, fbs))
        # Finish the figure
        if HIGHER_CORRS: # Then a legend is advised
            fig.legend(lns, [l.get_label() for l in lns], loc='upper center', ncol=4, frameon=False)
        fig.subplots_adjust(left=0.215, top=0.83, wspace=0.12, hspace=0.2)
        return fig

def get_exp_data(S_ASI: float):
    # Set the experiment values that should be reproduced
    data = {"times": (1e3, 7e7)} # (1000s, 26months)
    match S_ASI:
        case 20:
            # mm_kwargs |= {"E_EA_ratio": 65, "E_MC_ratio": 14.5}
            data |= {"m_avg": (0.471, 0.306), "q_NN": (0.436, 0.529), "q_2NN": (0.448, 0.506), "q_3NN": (0.386, 0.445)}
        case 25:
            # mm_kwargs |= {"E_EA_ratio": 65, "E_MC_ratio": 9.5}
            data |= {"m_avg": (0.719, 0.488), "q_NN": (0.278, 0.501), "q_2NN": (0.285, 0.360), "q_3NN": (0.225, 0.355)}
        # case 25:
        #     data = {"times": (240, 360, 2100, 2460, 9420, 9780, 18720),
        #             "m_avg": (0.785124, 0.768595, 0.768595, 0.752066, 0.752066, 0.735537, 0.735537),
        #             "q_NN": (0.218320, 0.226584, 0.226584, 0.242424, 0.242424, 0.260331, 0.260331),
        #             "q_2NN": (0.235537, 0.252066, 0.252066, 0.264463, 0.264463, 0.278926, 0.278926),
        #             "q_3NN": (0.173554, 0.181818, 0.181818, 0.199036, 0.199036, 0.212810, 0.212810)
        #     }
        case 30:
            # mm_kwargs |= {"E_EA_ratio": 65, "E_MC_ratio": 9}
            data |= {"m_avg": (0.752, 0.488), "q_NN": (0.249, 0.459), "q_2NN": (0.252, 0.382), "q_3NN": (0.202, 0.387)}
        case 35:
            # mm_kwargs |= {}
            data |= {"m_avg": (0.818, 0.653), "q_NN": (0.187, 0.328), "q_2NN": (0.198, 0.316), "q_3NN": (0.164, 0.247)}
        case 40:
            # mm_kwargs |= {"E_EA_ratio": 98, "E_MC_ratio": 16}
            data |= {"m_avg": (0.828, 0.669), "q_NN": (0.176, 0.295), "q_2NN": (0.169, 0.310), "q_3NN": (0.175, 0.308)}
        case _:
            raise ValueError(f"{S_ASI = }nm is not supported.")
    return data

def set_lightness(color, lightness=0.5):
    c = colorsys.rgb_to_hls(*to_rgb(color))
    return colorsys.hls_to_rgb(c[0], lightness, c[2])

def plot_all_iterations(path: str|Path, legend_title: str = r"$S_\mathrm{ASI}$", only_subdirs: list[str|Path] = None):
    """ Path can be a file (iterations.json) or a directory containing multiple directories with iterations.json in them. Only plots the first recursive subdirectory-level. """

    path = Path(path).absolute()
    if path.is_dir():
        subdirs = [path / Path(subdir) for subdir in only_subdirs] if only_subdirs is not None else path.iterdir()
        subdirs = [subdir for subdir in subdirs if Path.exists(subdir / 'iterations.json')] # Only check subdirs if they have an iterations.json
        files_iterations_json = [subdir / 'iterations.json' for subdir in subdirs]
    else:
        files_iterations_json = [path] if path.name == 'iterations.json' else []

    min_prob = 1e-10 if path.is_dir() else 1e-20
    cmap = colormaps['rainbow']
    varnames_readable = {"E_EA": "Net OOP anisotropy\n" + r"$E_\mathrm{EA}$ [$k_\mathrm{B}T$]",
                        "E_MC": "NN MS coupling\n" + r"$E_\mathrm{MC}$ [$k_\mathrm{B}T$]",
                        "J": "Exchange coupling\n" + r"$J$ [$k\mathrm{B}T$]"}
    for i, file in enumerate(files_iterations_json):
        with open(file, 'r') as inFile:
            iterations = [json.loads(line) for line in inFile]
        varnames = [k for k in iterations[0]['params'].keys()]
        iterations = [iteration for iteration in iterations[1:] if iteration['target'] > np.log10(min_prob)] # Only keep the somewhat decent iterations
        
        ## SUBPLOT FOR EACH VARIABLE, DIFFERENT OFFSETS ARE DIFFERENT COLORS, EACH SUBPLOT IS MSE AS FUNCTION OF VARIABLE
        if i == 0:
            fig, axes = plt.subplots(1, len(varnames), figsize=(thesis_utils.page_width, 4))
        for v, ax in enumerate(axes): # v is index of varnames
            ax: plt.Axes
            varname = varnames[v]
            
            probs = np.asarray([10**iteration['target'] for iteration in iterations])
            param_values = np.asarray([iteration['params'][varname] for iteration in iterations])
            probs, param_values = probs[order := probs.argsort()[::-1]], param_values[order] # Sort as function of prob (descending)
            
            color = to_rgb(cmap(i/(len(files_iterations_json)-1)) if path.is_dir() else 'k')
            alphas = np.array([.1 + .9*(1 - j/(len(probs) + 1))**10 for j, _ in enumerate(probs)])
            if not iterations: alphas = [1] # Otherwise empty array won't have a symbol in the legend
            color = [(*set_lightness(color, 0.5 if path.is_dir() else 0), alpha) for alpha in alphas] # Apply alpha to see more easily
            ax.scatter(param_values, probs, s=5+30*(probs/(np.max(probs) if iterations else 1))**.25,
                       c=color, edgecolors='none',
                       label=None if v != 0 else (file.parent.name if not file.parent.name.endswith('nm') else (file.parent.name[:-2] + "$\,$nm")))
            if iterations: ax.vlines([param_values[0]], [min_prob], [probs[0]], colors=[color[0]], linestyles='dashed')
            
            if i == 0: # Set some labels etc.
                ax.set_yscale('log')
                ax.set_ylim([min_prob, 1])
                ax.grid(color='gray', linestyle=':', linewidth=1, alpha=0.5)
                if v == 0: ax.set_ylabel(r"Fit quality $\eta$", fontsize=12)
                else: ax.set_yticklabels([])
                if varname == "J": ax.add_artist(Rectangle((-100, 0), 100, 1, alpha=0.2, color='gray', zorder=1)) # Nonphysical J values
                ax.set_xlabel(varnames_readable.get(varname, varname), fontsize=12)
                try: # Try to find the file that contains information about the BO bounds
                    with open(file.parent / "params.json") as inFile: ax.set_xlim(json.load(inFile)["variables"][varname])
                except Exception: pass

    fig.tight_layout()
    if path.is_dir():
        lgd = fig.legend(loc='center right', ncol=1, title=legend_title)
        for h in lgd.legend_handles: h._sizes = [30]
        fig.subplots_adjust(left=0.11, right=0.84, wspace=0.2)
    return fig


def run_optimization(S_ASI: float, n_iter: int = 8, static_params: dict = None, variables: dict = None, initial_guess: dict = None, max_total_iter: int = np.inf):
    # Process/sanitize input arguments
    exp_data = get_exp_data(S_ASI=S_ASI)
    msr = 170/(170 + S_ASI) if FINITE_MAGNET_SIZE else 0
    if static_params is None: static_params = {}
    if any(k in static_params.keys() for k in variables.keys()):
        raise ValueError("Duplicate keys between static_params and variables.")
    
    # Determine output directories and files
    file = f"{OUTDIR}/{S_ASI:.0f}nm/iterations.json"
    filebase = file.removesuffix('.json')
    os.makedirs(os.path.dirname(filebase), exist_ok=True)
    if not os.path.exists(file):
        if (pkl_file := Path(file).parent / "data.pkl").exists():
            with open(pkl_file, "rb") as picklefile:
                data = pickle.load(picklefile)
            with open(file, "w") as iterationsfile:
                for iteration in data["all_iterations"]:
                    iterationsfile.write(json.dumps(iteration) + "\n")
    
    # Define minimization function
    def BO_optimizer_wrapper(plot=False, **kwargs):
        print(f"Running {S_ASI:.0f}nm with parameters {kwargs}")
        if plot: return compare_exp_sim(exp_data=exp_data, n_avg=200, plot=plot, magnet_size_ratio=msr, **kwargs)
        try:
            return max((min_val := -100), compare_exp_sim(exp_data=exp_data, n_avg=200, plot=plot, magnet_size_ratio=msr, **kwargs))
        except Exception:
            return min_val
    
    # Initialize optimizer
    optimizer = bayes_opt.BayesianOptimization(
        f=BO_optimizer_wrapper,
        pbounds=variables,
        random_state=np.random.randint(1000000),
        verbose=2, # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        allow_duplicate_points=True # Can happen when using joblib.Parallel
    )
    load_previous = os.path.exists(filebase + '.json') if filebase is not None else False
    if load_previous:
        try:
            bayes_opt.util.load_logs(optimizer, logs=[filebase + ".json"])
            print(f"New optimizer is now aware of {len(optimizer.space)} points.")
        except FileNotFoundError:
            load_previous = False
            print("No previous logs found.")
    logger = bayes_opt.logger.JSONLogger(path=filebase, reset=False) # Generates iterations.json
    optimizer.subscribe(bayes_opt.Events.OPTIMIZATION_STEP, logger)
    if initial_guess is not None and not load_previous: optimizer.probe(params=initial_guess)
    
    # Perform <n_iter> iterations, or stop when iterations.json contains <max_total_iter> iterations.
    n_iter = np.clip(n_iter, a_min=0, a_max=max_total_iter - len(optimizer.space))
    n_init = 0 if load_previous else min(5, max_total_iter)
    optimizer.maximize(init_points=n_init, n_iter=max(0, n_iter - n_init))
    # n_init = n_iter
    # optimizer.maximize(init_points=n_init, n_iter=0)
    
    # Recalculate the best guess to save more details about it
    print(f"Best guess: {optimizer.max}")
    hotspice.utils.save_results(
        outdir=os.path.dirname(filebase),
        parameters={"static_params": static_params, "variables": variables, "result_best_guess": optimizer.max},
        data={"best_guess": optimizer.max, "all_iterations": optimizer.res},
        figures=(
            BO_optimizer_wrapper(plot=True, **optimizer.max['params']),
            plot_all_iterations(file)
        )
    )
    plt.close()


if __name__ == "__main__":
    ## Check if code works as expected (to some extent)
    # assert abs(-1.5 - compare_exp_sim(exp_data=get_exp_data(25), E_EA=65, E_MC=9.5, J=0, n_avg=200)) < 0.5  # Should be around -1.5 without big fluctuations
    
    ## Run BO
    n_iter = 20 if not JUST_REPLOT else 0
    max_total_iter = 500
    ranges = {"E_EA": (20, 150), "E_MC": (3, 30), "J": (-20, 20)}
    funcs = [
        lambda: run_optimization(S_ASI=20, n_iter=n_iter, max_total_iter=max_total_iter,
                        variables=ranges, initial_guess={"E_EA": 65, "E_MC": 14.5, "J": 0}),
        lambda: run_optimization(S_ASI=25, n_iter=n_iter, max_total_iter=max_total_iter,
                        variables=ranges, initial_guess={"E_EA": 65, "E_MC": 10, "J": 0}),
        lambda: run_optimization(S_ASI=30, n_iter=n_iter, max_total_iter=max_total_iter,
                        variables=ranges, initial_guess={"E_EA": 65, "E_MC": 9, "J": 0}),
        # lambda: run_optimization(S_ASI=35, n_iter=n_iter, max_total_iter=max_total_iter,
        #                 variables=ranges, initial_guess={"E_EA": 65, "E_MC": 9, "J": 0}),
        lambda: run_optimization(S_ASI=40, n_iter=n_iter, max_total_iter=max_total_iter,
                        variables=ranges, initial_guess={"E_EA": 98, "E_MC": 16, "J": 0})
    ]
    N = len(funcs)
    funcs *= int(max_total_iter // n_iter) if not JUST_REPLOT else 1 # Combined with a low `n_iter`, this can be used to update the plot from time to time while the iteration run.
    Parallel(n_jobs=min(8, N), backend='loky')(delayed(f)() for f in funcs) # Plotting will fail with this though
    
    ## Post-processing
    hotspice.utils.save_results(outdir=OUTDIR, figures=plot_all_iterations(OUTDIR), copy_script=False)
