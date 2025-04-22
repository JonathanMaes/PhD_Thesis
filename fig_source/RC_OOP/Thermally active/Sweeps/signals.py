import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from collections import deque
from matplotlib.axes import Axes
from typing import Callable
# from sympy import simplify
# from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_application, function_exponentiation

from thermally_active_ASI_old import UniformInputter
import hotspice
from hotspice.utils import asnumpy
from hotspice.experiments import SweepMetricPlotparams
xp = hotspice.xp


__all__ = ['Signal', 'MackeyGlass', 'Signals', 'SignalTransformationExperiment']

class Signal: # To ensure uniformity between the signal and target functions
    # TODO: have some fun with sympy to simplify expressions for all the __dunder__ methods
    def __init__(self, funfunfunction: Callable, name: str = None, /):
        self.f = funfunfunction
        # try:
        #     self.name = str(simplify(parse_expr(name, transformations=standard_transformations + (implicit_application, function_exponentiation))))
        # except:
        self.name = name
    def __call__(self, time: float) -> float: return self.f(time)
    def plot(self, t=2, *, N: int = 201, ax: Axes = None, show=True):
        if ax is None: ax = plt
        ax.plot(times := np.linspace(0, t, N), self(times), label=self.name.replace("-", "−") if self.name is not None else None)
        if show: plt.show()
    def __repr__(self): return self.name
    # Some utility functions to get transformed versions of a Signal
    def speedx(self, factor: float, name: str = None):
        """ Returns a Signal b(t) which is a(factor*t). So frequency increased by <factor>. """
        return Signal(lambda t: self(factor*t), f"({self.name})({factor:.2g}*t)" if name is None else name)
    def offset(self, dt: float, name: str = None):
        """ Returns a Signal b(t) which is a(t - dt). So the signal shifted <dt> to the right. """
        return Signal(lambda t: self(t - dt), f"({self.name})(t{'-' if dt > 0 else '+'}{abs(dt):.2g})" if name is None else name)
    def Kminus1Kminus2(self, dt: float, name: str = None):
        """ Returns a Signal b(t) which is a(t - dt)*a(t - 2*dt). """
        return Signal(lambda t: self(t - dt)*self(t - 2*dt), f"({self.name})(t-{dt:.2g})*({self.name})(t-{2*dt:.2g})" if name is None else name)
    def __add__(self, other):
        return Signal(lambda t: self(t) + other(t), f"({self.name})+({other.name})") if isinstance(other, Signal) else Signal(lambda t: self(t) + other, f"({self.name})+{other:.2g}")
    def __radd__(self, other): return self + other
    def __sub__(self, other):
        return Signal(lambda t: self(t) - other(t), f"({self.name})-({other.name})") if isinstance(other, Signal) else Signal(lambda t: self(t) - other, f"({self.name})-{other:.2g}")
    def __rsub__(self, other):
        return Signal(lambda t: other(t) - self(t), f"({other.name})-({self.name})") if isinstance(other, Signal) else Signal(lambda t: other - self(t), f"{other:.2g}-({self.name})")
    def __mul__(self, other):
        return Signal(lambda t: self(t)*other(t), f"({self.name})*({other.name})") if isinstance(other, Signal) else Signal(lambda t: self(t)*other, f"{other:.2g}*({self.name})")
    def __rmul__(self, other): return self*other
    def __truediv__(self, other):
        return Signal(lambda t: self(t)/other(t), f"({self.name})/({other.name})") if isinstance(other, Signal) else Signal(lambda t: self(t)/other, f"({self.name})/{other:.2g}")
    def __rtruediv__(self, other):
        return Signal(lambda t: other(t)/self(t), f"({other.name})/({self.name})") if isinstance(other, Signal) else Signal(lambda t: other/self(t), f"{other:.2g}/({self.name})")
    def __neg__(self):
        return Signal(lambda t: -self(t), f"-{self.name}")

class MackeyGlass:
    def __init__(self, tau=23.0, beta=0.2, gamma=0.1, n=10.0, steps_per_tau: int = 1000, x0=None, discard: int = 25):
        """ Generate Mackey-Glass time series using the discrete approximation of Grassberger & Procaccia (1983).
            Adapted from https://github.com/manu-mannattil/nolitsa/blob/0e3cfd59c82c21c42da55cd24ee944b44aa0d9ad/nolitsa/data.py#L223.
            @param tau, beta, gamma, n [float] (23, 0.2, 0.1, 10): Standard parameters of the Mackey-Glass oscillator.
            @param steps_per_tau [int] (1000): Number of discrete steps into which the interval between <t> and <t+tau> should
                be divided. This results in a time step of <tau/steps_per_tau> and an <steps_per_tau + 1> dimensional map.
            @param x0 [np.ndarray] (random): Array of length <steps_per_tau> for the initial condition of the discrete map.
            @param discard [int] (25): Number of <steps_per_tau>-steps to discard in order to eliminate transients.
                A total of <steps_per_tau*discard> steps will be discarded.
        """
        if tau < 17: raise ValueError("tau must be > 17.")
        self.tau, self.beta, self.gamma, self.n = tau, beta, gamma, n
        deque_len = steps_per_tau + 1
        if x0 is None:
            x0 = np.random.random(deque_len)
        elif isinstance(x0, list|np.ndarray):
            x0 = np.asarray(x0)
        else:
            x0 = x0*np.ones(deque_len)
        if len(x0) != deque_len: raise ValueError(f"x0 must have length {deque_len} (steps_per_tau + 1).")
        self.deque = deque(x0, maxlen=deque_len)
        self.history = []
        
        self.steps_per_tau = steps_per_tau
        self.sample: int = 20
        self.discard = discard

        self.A = (2*self.steps_per_tau - self.gamma*self.tau)/(2*self.steps_per_tau + self.gamma*self.tau)
        self.B = self.beta*self.tau/(2*self.steps_per_tau + self.gamma*self.tau)
        
        min_val, max_val = np.nan, np.nan
        for i in range(self.steps_per_tau*self.discard): # Extinguish transients and establish rescale range
            val = self.next(_record_history=False)
            if i < self.steps_per_tau: continue
            if not (min_val <= val): min_val = val
            if not (max_val >= val): max_val = val
        self.scaling_range = (max_val - min_val)*1.03
        self.scaling_min = min_val - 0.01*self.scaling_range
        self.history = []
        
    def next(self, _record_history=True):
        for _ in range(self.sample):
            nextvalue = self.A*self.deque[-1] + self.B*(self.deque[0]/(1 + self.deque[0]**self.n) + self.deque[1]/(1 + self.deque[1]**self.n))
            self.deque.append(nextvalue) # deque so oldest value gets pushed out
        if _record_history:
            nextvalue = (self.deque[-1] - self.scaling_min)/self.scaling_range # Move values (probably) in range [0, 1]
            self.history.append(nextvalue)
        return nextvalue
    
    def __call__(self, time: float) -> float:
        if isinstance(time, np.ndarray): return np.asarray([self(t) for t in time]) # Problem is that self.history must be appendable, so it can not be a numpy array and thus can not be indexed by array.
        if time < 0: raise ValueError("Can only get Mackey-Glass at times >= 0.")
        time_n = time*self.steps_per_tau/self.sample # Need points at self.history[floor(time_n)] and self.history[ceil(time_n)]
        time_n_min, time_n_plus, time_n_frac = math.floor(time_n), math.ceil(time_n), math.modf(time_n)[0]
        additional_steps_needed = time_n_plus - (len(self.history) - 1)
        if additional_steps_needed > 0:
            for _ in range(additional_steps_needed): self.next()
        value = time_n_frac*self.history[time_n_plus] + (1 - time_n_frac)*self.history[time_n_min] # Linear interpolation
        return value


class SignalsMeta(type):
    def __iter__(self): return iter([s for s in self.__dict__.values() if isinstance(s, Signal)])
class Signals(metaclass=SignalsMeta):
    SINE = Signal(lambda t: (1+np.sin(2*np.pi*t))/2, "sin")
    ABSSINE = Signal(lambda t: np.abs(np.sin(np.pi*t)), "abssin")
    SINESQUARED = Signal(lambda t: np.sin(2*np.pi*t)**2, "sin²")
    SAW = Signal(lambda t: np.remainder(t, 1), "saw")
    SQUARE = Signal(lambda t: np.heaviside(.5 - Signals.SAW(t), .5), "Square")
    MACKEYGLASS = Signal(MackeyGlass(x0=0.5).__call__, "MackeyGlass") # x0=.5 for consistency
    MSO12 = Signal(lambda t: sum(np.sin(2*np.pi*t*phi) for phi in [.2, .331, .42, .51, .63, .74, .85, .97, 1.08, 1.19, 1.27, 1.32])/12, "MSO12") # From "Hierarchical architectures in reservoir computing systems" in J. Moon et al.
    
    @classmethod
    def preview(cls):
        fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
        ax: Axes = axes[0,0]
        ax.set_title("Standard signals")
        for signal in cls: signal.plot(ax=ax, show=False)
        ax.legend()
        ax = axes[0,1]
        ax.set_title("Transformations of sin(t)")
        transformations = [cls.SINE, cls.SINE.Kminus1Kminus2(0.01), cls.SINE.offset(0.05)]
        for signal in transformations: signal.plot(ax=ax, show=False)
        ax.legend()
        ax = axes[1,0]
        ax.set_title("Random Mackey-Glasses")
        for _ in range(5): Signal(MackeyGlass().__call__).plot(ax=ax, show=False)
        plt.show()


class SignalTransformationExperiment(hotspice.experiments.Experiment): # An Experiment() implementation of the signal tf task.
    def __init__(self, mm: hotspice.Magnets, signal: Signal, target: Signal, frequency: float = 1, magnitude: float = 1, offset: float = 0, res_x: int = None, res_y: int = None, use_constant: bool = True):
        """ Performs a signal transformation from <signal> to <target> by training an OLS estimator on the reservoir output.
            The performance is measured by MSE: the 2 main MSEs are MSE_reservoir and MSE_rawinput (for the test set).
                The train MSEs can also be calculated, but are not a good measure of the performance of the reservoir.
                They can be used to see whether the OLS is overfitting (e.g. due to too many readout nodes).

            @param mm [hotspice.Magnets]: The Magnets object representing the ASI, with all parameters already configured.
            @param signal [Signal]: The input signal. It will be offset by <offset> (default no offset).
            @param target [Signal]: The intended output signal, that the OLS should approximate using reservoir readouts.
                NOTE: <signal> and <target> should have the same timescale.
            @param offset [float]: Shifts the <signal> and <target> in time w.r.t. each other.
                such that <signal(t)> will be mapped to <target(t+offset)>.
                SO POSITIVE OFFSET MEANS PREDICTION, NEGATIVE OFFSET IS RECALLING PREVIOUS VALUES (if signal==target)
            @param frequency [float] (1): The real frequency at which the <signal> is applied.
                This is because <signal> and <target> are assumed to have a period of 1 second.
                NOTE: this is not <self.inputter.frequency>, because that is dynamically determined by samples_per_second.
        
        NOTE: <frequency> is the frequency of the <signal> and <target>, not the frequency of the <self.inputter>.
                  The UniformInputter is generated dynamically based on this frequency and the sample rate specified in self.run().
        """
        self.inputter = UniformInputter(magnitude=magnitude) # Just a constant inputter. Frequency will be set in self.run().
        self.outputreader = hotspice.io.RegionalOutputReader(mm.nx if res_x is None else res_x,
                                                             1 if res_y is None else res_y,
                                                             mm) #! This assumes the gradient is along the x-direction.
        super().__init__(self.inputter, self.outputreader, mm) # Just to be sure to conform to the expected Experiment structure
        if offset < 0: signal = signal.offset(offset) # Always want to offset forward in time (neg offset), because backwards
        elif offset > 0: target = target.offset(-offset) # is not possible for Mackey-Glass due to iterative implementation
        self.signal, self.target = signal, target
        self.frequency, self.magnitude = frequency, magnitude #! Can not immediately pass an <inputter>, because samples_per_second will influence the actual <inputter.frequency> required.
        self.use_constant = use_constant

    def run(self, num_periods: float = 20, samples_per_period: float = 20, warmup_periods: float = 5, verbose: bool = False):
        """ Performs the necessary Hotspice simulations to apply <self.signal> for several periods, and records the readout.
            @param num_periods [float] (20): The number of periods of <self.signal> that will be applied.
                (So the simulation will run for <num_periods/frequency> seconds.)
            @param samples_per_period [float] (20): The number of substeps per period. A number >20 is usually ok.
            @param warmup_periods [float] (5): A few periods will be simulated without recording the result.
                This is to eliminate transients. 5 periods is usually enough, depending on the kind of input signal.
        """
        self.inputter.set_frequency(self.frequency*samples_per_period)
        self.N, N_warmup = math.floor(samples_per_period*num_periods), math.floor(samples_per_period*warmup_periods)
        times_warmup, times = np.split(np.arange(self.N + N_warmup)/samples_per_period, [N_warmup]) #! Fraction of <signal> period! Not true time (that is self.t)

        self.t = times/self.frequency # The real time
        self.signal_values, self.target_values = self.signal(times), self.target(times)
        self.readout_values = np.zeros((self.N, self.outputreader.n))
        self.inputter.input(self.mm, values=self.signal(times_warmup)) # WARMUP: Some unrecorded periods to eliminate transients
        for i in range(self.N):
            if verbose and i % 100 == 0: print(f"Period {i//samples_per_period} of {num_periods}: {self.mm.switches} switches occurred ({self.mm.MCsteps} MC steps)")
            self.inputter.input(self.mm, values=self.signal_values[i])
            self.readout_values[i,:] = asnumpy(self.outputreader.read_state())

    def train_estimator(self, train_fraction: float = 0.6, use_constant: bool = None):
        """ Using the recorded readout from <self.run()>, an estimator is trained to assess the ASI performance.
            @param train_fraction [float] (0.6): this fraction of the recorded input-output values will be used to
                train the OLS estimator. The other <1-train_fraction> will be the test set for performance evaluation.
            @param use_constant [bool] (True): whether to include an additional constant when training the OLS estimator.
                This is particularly important for the raw input 'prediction': the added constant will allow the <signal>
                to also be vertically offset, in addition to otherwise only a scaling, when fitting it to the <target>,
                often drastically improving the raw input prediction's MSE.
                It also slightly improves the reservoir prediction, but not as drastically as the raw input prediction.
        """
        if use_constant is not None: self.use_constant = use_constant
        self.N_train = int(train_fraction*self.N)
        if self.N_train < 10 or (self.N - self.N_train) < 10: raise ValueError(f"train_fraction or test_fraction is too small for the number of samples (N={self.N})")
        
        self.signal_train, self.signal_test = xp.split(self.signal_values, [self.N_train])
        self.readout_train, self.readout_test = xp.split(sm.add_constant(self.readout_values) if self.use_constant else self.readout_values, [self.N_train])
        self.target_train, self.target_test = xp.split(self.target_values, [self.N_train])
        
        # Uses regularized fit as in "Reconfigurable Training and Reservoir Computing in an Artificial Spin-Vortex Ice via Spin-Wave Fingerprinting")
        signal_train, signal_test = xp.split(sm.add_constant(self.signal_values) if self.use_constant else self.signal_values, [self.N_train])
        self.OLS_rawinput = sm.OLS(self.target_train, signal_train).fit_regularized(alpha=0.001, L1_wt=0)
        self.OLS_reservoir = sm.OLS(self.target_train, self.readout_train).fit_regularized(alpha=0.001, L1_wt=0)
        self.fit_rawinput = self.OLS_rawinput.predict(signal_train) # Train prediction based on raw input
        self.fit_reservoir = self.OLS_reservoir.predict(self.readout_train) # Train prediction based on reservoir readout
        self.prediction_rawinput = self.OLS_rawinput.predict(signal_test) # Test prediction based on raw input
        self.prediction_reservoir = self.OLS_reservoir.predict(self.readout_test) # Test prediction based on reservoir readout
        
        return self.MSE_rawinput, self.MSE_reservoir
    
    def get_train(self, arr):
        if len(arr) != self.N: raise ValueError(f"Array must have length {self.N} to get the train set.")
        return arr[:self.N_train]
    
    def get_test(self, arr):
        if len(arr) != self.N: raise ValueError(f"Array must have length {self.N} to get the test set.")
        return arr[self.N_train:]

    def MSE(self, arr, target): # TODO: perhaps use NRMSE instead (see "Hierarchical architectures in reservoir computing systems")
        return np.mean((arr - target)**2)
    @property
    def MSE_rawinput(self):
        return self.MSE(self.prediction_rawinput, self.target_test)
    @property
    def MSE_reservoir(self):
        return self.MSE(self.prediction_reservoir, self.target_test)
    @property
    def MSE_rawinput_train(self):
        return self.MSE(self.fit_rawinput, self.target_train)
    @property
    def MSE_reservoir_train(self):
        return self.MSE(self.fit_reservoir, self.target_train)
    
    def calculate_all(self, **kwargs):
        """ (Re)calculates all the metrics in the <self.results> dict. """
        self.train_estimator()
        self.results = {
            "MSE_reservoir": self.MSE_reservoir,
            "MSE_rawinput": self.MSE_rawinput,
            "MSE_reservoir_train": self.MSE_reservoir_train,
            "MSE_rawinput_train": self.MSE_rawinput_train
        }

    def to_dataframe(self) -> pd.DataFrame:
        """ Creates a Pandas dataframe from the saved results of self.run(). """
        return pd.DataFrame({'t': self.t, 'signal_values': self.signal_values, 'target_values': self.target_values, 'readout_values': list(self.readout_values)})

    def load_dataframe(self, df: pd.DataFrame):
        """ Loads the data from self.to_dataframe() to the current object.
            Might return the most important columns of data, but this is not required.
        """
        self.t = df['t']
        self.signal_values = df['signal_values']
        self.target_values = df['target_values']
        self.readout_values = xp.asarray([xp.asarray(readout) for readout in df['readout_values']])
        self.N = self.t.size

    @classmethod
    def dummy(cls, mm: hotspice.Magnets = None):
        """ Creates a minimalistic working SignalTransformationExperiment instance.
            @param mm [hotspice.Magnets] (None): if specified, this is used as Magnets()
                object. Otherwise, a minimalistic hotspice.ASI.OOP_Square() instance is used.
        """
        if mm is None: mm = hotspice.ASI.OOP_Square(1, 11, energies=(hotspice.DipolarEnergy(), hotspice.ZeemanEnergy()))
        return cls(mm, Signals.SINE, Signals.SAW)
    
    def get_plot_metrics():
        from hotspice.experiments import SweepMetricPlotparams
        # return { ## MSE
        #     "MSE_reservoir": SweepMetricPlotparams('ASI (test)', lambda data: data["MSE_reservoir"], min_value=0, max_value=(lambda data: data["MSE_reservoir_train"].max()), contours=[lambda data: data["MSE_rawinput"].max()], lower_is_better=True), # This can be ultra bad, so set max value to max of MSE_reservoir_train
        #     "MSE_reservoir_train": SweepMetricPlotparams('ASI (train)', lambda data: data["MSE_reservoir_train"], min_value=0, contours=[lambda data: data["MSE_rawinput"].max()], lower_is_better=True), # This will never be ultra bad, so no max_value needed
        #     "MSE_rawinput": SweepMetricPlotparams('Raw input (test)', lambda data: data["MSE_rawinput"], min_value=0) # Last, because it is usually constant so we can just cut it off when putting figure in report
        # }
        max_func = lambda data: max(1/data["MSE_reservoir"].max(), 1/data["MSE_rawinput"].max()) # Do not use train for max, because it can overfit
        return { ## 1/MSE
            "MSE_reservoir": SweepMetricPlotparams('ASI reservoir (test set)', lambda data: 1/data["MSE_reservoir"], min_value=0, contours=[lambda data: 1/data["MSE_rawinput"].min()], lower_is_better=False), # This can be ultra bad, so set max value to max of MSE_reservoir_train
            "MSE_reservoir_train": SweepMetricPlotparams('ASI reservoir (training set)', lambda data: 1/data["MSE_reservoir_train"], min_value=0, contours=[lambda data: 1/data["MSE_rawinput"].min()], lower_is_better=False), # This will always be the best, so no max_value needed
            "MSE_rawinput": SweepMetricPlotparams('Raw input (test)', lambda data: 1/data["MSE_rawinput"], min_value=0, max_value=(lambda data: max_func(data)), omit_if_constant=True) # Last, because it is usually constant so we can just cut it off when putting figure in report
        }

    def plot(self, N_cycles: int = None, show=True, show_signal=True, samples_per_period=None, fig_height=3, ax: plt.Axes = None):
        ## Plot font sizes
        fontsize_header = 14
        fontsize_axes = 16
        fontsize_legend = 14
        
        ## Determine the best-looking part of the data to show in the plot
        if samples_per_period is None: samples_per_period = np.argmax((self.t - np.min(self.t)) >= 1/self.frequency)
        N_idx = np.argmax((self.t - np.min(self.t)) >= (N_cycles*.8)/self.frequency) # Number of indices per N_cycles*0.8 (legend hides 20% of plot)
        N_idx_full = np.argmax((self.t - np.min(self.t)) >= N_cycles/self.frequency) # Number of indices per N_cycles (to know the full plot length)
        offset = int(samples_per_period/7)
        # Train set:
        MSEs = [self.MSE(self.fit_reservoir[i:i+N_idx], self.target_train[i:i+N_idx]) for i in range(self.target_train.size - N_idx_full)]
        best_start_idx = np.argmin(MSEs[offset::samples_per_period])*samples_per_period + offset
        train_slice = (best_start_idx, best_start_idx + N_idx_full)
        # Test set:
        MSEs = [self.MSE(self.prediction_reservoir[i:i+N_idx], self.target_test[i:i+N_idx]) for i in range(self.target_test.size - N_idx_full)]
        best_start_idx = np.argmin(MSEs[offset::samples_per_period])*samples_per_period + offset + self.target_train.size
        test_slice = (best_start_idx, best_start_idx + N_idx_full)
        
        ## Draw the plots
        if ax is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, fig_height))
            fig.suptitle(f"{self.signal.name} $\\rightarrow$ {self.target.name}", x=0.01, ha='left', fontsize=fontsize_header)
        t_rescaled, t_unit = hotspice.utils.appropriate_SIprefix(self.t)
        t_scale = 10**hotspice.utils.SIprefix_to_magnitude[t_unit]
        
        if ax is None: ax1: Axes = axes[0]
        else: ax1 = ax
        ax1.set_title(f"Training set", fontsize=fontsize_header)
        ax1.set_xlabel(f"Time ({t_unit}s)", fontsize=fontsize_axes)
        ax1.set_ylim([-0.1, 1.1])
        t = self.get_train(t_rescaled)
        if N_cycles is not None:
            ax1.set_xlim(self.t[train_slice[0]]/t_scale, self.t[train_slice[1]]/t_scale)
            # ax1.set_xlim([np.min(t), min(np.max(t), np.min(t) + N_cycles/self.frequency/t_scale)])
        ax1.plot(t, self.get_train(self.target_values), "k--")
        ax1.plot(t, self.fit_rawinput, "r", alpha=0.5, label=f"{1/self.MSE(self.fit_rawinput, self.target_train):.2f}")
        ax1.plot(t, self.fit_reservoir, "dodgerblue", label=f"{1/self.MSE(self.fit_reservoir, self.target_train):.2f}")
        if show_signal: ax1.plot(t, self.signal_train, "k:")
        ax1.legend(title=r"1/MSE", fontsize=fontsize_legend, loc='lower right', ncol=1, title_fontproperties={'weight':'bold', 'size': fontsize_legend-2})
        ax1.tick_params(axis='both', labelsize=fontsize_axes)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        if ax is not None: return
        
        ax2: Axes = axes[1]
        ax2.set_title(f"Test set", fontsize=fontsize_header)
        ax2.set_xlabel(f"Time ({t_unit}s)", fontsize=fontsize_axes)
        t = self.get_test(t_rescaled)
        if N_cycles is not None:
            ax2.set_xlim(self.t[test_slice[0]]/t_scale, self.t[test_slice[1]]/t_scale)
            # ax2.set_xlim([np.min(t), min(np.max(t), np.min(t) + N_cycles/self.frequency/t_scale)])
        ax2.set_ylim([-0.1, 1.1])
        line_target, = ax2.plot(t, self.get_test(self.target_values), "k--", alpha=1.0)
        line_input_pred, = ax2.plot(t, self.prediction_rawinput, "r", alpha=0.5, label=f"{1/self.MSE_rawinput:.2f}")
        line_reservoir, = ax2.plot(t, self.prediction_reservoir, "dodgerblue", label=f"{1/self.MSE_reservoir:.2f}")
        if show_signal: line_input, = ax2.plot(t, self.signal_test, "k:", alpha=1.0)
        ax2.legend(title=r"1/MSE", fontsize=fontsize_legend, loc='lower right', ncol=1, title_fontproperties={'weight':'bold', 'size': fontsize_legend-2})
        ax2.tick_params(axis='both', labelsize=fontsize_axes)
        ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
        
        fig.legend(handles=[line_input, line_target, line_input_pred, line_reservoir], labels=["Input signal", "Target", "Prediction with input only", "Prediction with ASI"],
                   loc='upper center', ncol=4, fontsize=fontsize_header)
        fig.tight_layout()
        # fig.subplots_adjust(left=0.1, right=0.99)
        if show:
            fig.show()
            plt.show()
        return fig


if __name__ == "__main__":
    from thermally_active_ASI import get_thermal_mm

    def main_single(show=True, MG: float = 0.):
        """ Sine-to-saw tf if <MG> is zero, otherwise Mackey-Glass with offset <MG>.
            Raw input has worst performance for MG=-0.7, best below that at MG=-1.35.
            Performance of ASI usually monotonously increases, indicating lack of memory.
        """
        mm_kwargs = {'E_B_ratio': 20, 'DD_ratio': 2.5, 'E_B_std': 0.05, 'gradient': 0.1, 'pattern': 'AFM', 'size': 44, 'DD_exponent': -3}
        mm = get_thermal_mm(**mm_kwargs)
        magnitude, frequency = 0.00034, 8e2 # Optimal for mm_kwargs = {'E_B_ratio': 20, 'DD_ratio': 2.5, 'E_B_std': 0.05, 'gradient': 0.1, 'pattern': 'AFM', 'size': 20, 'DD_exponent': -3}
        if MG != 0:
            signaltf = SignalTransformationExperiment(mm, Signals.MACKEYGLASS, Signals.MACKEYGLASS.offset(MG), magnitude=magnitude, frequency=frequency)
        else:
            signaltf = SignalTransformationExperiment(mm, Signals.SINE, Signals.SAW, magnitude=magnitude, frequency=frequency)
            # signaltf = SignalTransformationExperiment(mm, Signals.SAW, Signals.ABSSINE, magnitude=magnitude, frequency=frequency)
            # signaltf = SignalTransformationExperiment(mm, Signals.SINE, Signals.SINESQUARED, magnitude=magnitude, frequency=frequency)
        signaltf.run(num_periods=100, samples_per_period=20)
        signaltf.train_estimator(use_constant=True)
        if show: signaltf.plot()
        return signaltf

    main_single()
    
    # TODO:
    # - Dependence on <size>? (signal-agnostic, performance should increase with increasing size, so don't optimize this for a given task. The question is how much better it gets with larger sizes.)
    #   NOTE: does it depend on <size> or on <res_x>? Interesting distinction to make. Better performance might just be due to the larger number of readouts, but on the other hand adding more vertical readouts makes the results overfit more. Certainly worth researching the dependence on <res_x> for bigger systems to see if it is <size> or <res_x> that improves the results (or both).
    # - Dependence on <gradient>? (apparently not signal-agnostic, but perhaps there is a general trend? Is there an optimum value that depends on the transformation?)
    # - Dependence on <DD_ratio>?
    # - Dependence on <offset>? For MG this is obviously important, but also for Sin-Saw this can be interesting because SIN.offset(0.5) gives significantly worse results by the looks of it.
    #                           Perhaps there is an interesting co-dependence between <offset> and <frequency>?
    # DONE:
    # - Dependence on <frequency> and <magnitude> (TODO: do a wider range once the parallel calculations are implemented)
