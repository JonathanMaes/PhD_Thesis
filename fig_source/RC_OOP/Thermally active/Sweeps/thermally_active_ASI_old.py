"""
Here we try to make a thermally active OOP ASI, inspired by the system that Alex fabricated.
This file contains methods and classes for easily creating thermal ASI and corresponding Inputters.
"""
import os
os.environ['HOTSPICE_USE_GPU'] = 'False'
import hotspice

import numpy as np


def get_thermal_mm(E_B_ratio: float = 1, DD_ratio: float = 1, J_ratio: float = 0, E_B_std: float = 0.05, gradient: float = 0, size: int = 20, size_y: int = None, magnet_size_ratio: float = 0, DD_exponent: float = -3, T_factor: float = 1, pattern: str = 'AFM'):
    """ Creates a thermally active ASI with
            E_B = normal distribution (mean: E_B_ratio*kBT, standard deviation: E_B_std*E_B)
            NN DD = DD_ratio*kBT
        The temperature is always assumed to be 300 K.
        @param E_B_ratio [float] (1): the mean energy barrier will be this multiple of kBT.
        @param DD_ratio [float] (1): the DD interaction energy between nearest-neighbors as a multiple of kBT.
        @param J_ratio [float] (1): the exchange interaction energy between neighbors as a multiple of kBT.
        @param E_B_std [float] (0.05): the standard deviation (as a relative fraction) on the energy barrier. Default 5%.
        @param gradient [float] (0): if nonzero, a horizontal gradient is applied to some parameter(s) (multiplicative 1±<gradient>).
            Currently, this gradient is only applied to the magnetic moment.
        @param size [int] (20): the number of magnets along a side of the system, so there are `size`x`size` magnets.
            Can also use `size_y` to specify the y-size separately (then `size` is used as the x-size).
        @param DD_exponent [float] (-3): the decay of the DD interaction as a function of distance.
            Can be used for simulating additional permalloy striplines, in combination with `limit_DD_orthogonal()`.
        @param T_factor [float] (1): <{E_B|DD|J}_ratio> are all relative to 300K. T_factor determines the actual temperature.
            This functionality was added to allow changing the temperature while keeping the physical system identical.
            This way, we can show that changing the temperature can shift the optimal frequency of a signal transformation.
        @param pattern [str] ('AFM'): the initial state of the system. Default: the ground state (AFM).
    """
    kBT = hotspice.kB*300 # [K]
    T = 300*T_factor
    if size_y is None: size_y = size
    dipolar_energy = hotspice.DipolarEnergy(prefactor=1.75, decay_exponent=DD_exponent) # prefactor gets overwritten by set_NN_interaction, so does not matter (1.75 is MuMax-inspired correction factor due to finite size of magnets)
    gradient_profile = np.tile(np.logspace(np.log10(1-gradient), np.log10(1+gradient), size), (size_y,1))
    mm = hotspice.ASI.OOP_Square(a=(a := (d := 170e-9) + (sep := 30e-9)),
                                 n=size, ny=size_y,
                                 moment=gradient_profile*(Msat := 1063242)*(t_lay := 1.4e-9)*(n_lay := 7)*np.pi*(d/2)**2,
                                 E_B=gradient_profile*E_B_ratio*kBT*np.random.normal(1, E_B_std, size=(size_y, size)), T=T,
                                 energies=(hotspice.ZeemanEnergy(magnitude=0), dipolar_energy),
                                 major_axis=magnet_size_ratio*a,
                                 params=hotspice.SimParams(UPDATE_SCHEME=hotspice.Scheme.NEEL),
                                 pattern=pattern)
    dipolar_energy.set_NN_interaction(DD_ratio*kBT)
    if J_ratio != 0: mm.add_energy(hotspice.ExchangeEnergy(J=J_ratio*kBT))
    return mm
    

def limit_DD_orthogonal(energy: hotspice.DipolarEnergy, horizontal=True, vertical=True):
    """ This removes all the dipole interactions between magnets which do not have the same x- or y- coordinates.
        This mimics the permalloy striplines that can be added to the system to reduce the DD decay with distance.
        @param <horizontal> [bool] (True): If False, the purely horizontal interactions are removed as well.
        @param <vertical> [bool] (True): If False, the purely vertical interactions are removed as well.
        So basically, these booleans correspond to whether or not there are Py striplines along that axis.
    """
    for kernel in energy.kernel_unitcell:
        ny, nx = kernel.shape
        ny, nx = ny//2, nx//2
        kernel[:ny,:nx] = kernel[ny+1:,:nx] = kernel[:ny,nx+1:] = kernel[ny+1:,nx+1:] = 0
        if not vertical: kernel[:ny,:] = kernel[ny+1:,:] = 0
        if not horizontal: kernel[:,:nx] = kernel[:,nx+1:] = 0

## CUSTOM INPUTTER
# Most basic inputter: just a uniform external field
class UniformInputter(hotspice.io.Inputter):
    def __init__(self, datastream: hotspice.io.Datastream = None, 
                 magnitude: float = 1, dutycycle: float = 1, frequency: float = 1, n: float = 2):
        """ Applies an external field for a duration of <dutycycle/frequency> seconds.
            Time between the start of successive pulses is <(1-dutycycle)/frequency> seconds.
            If any of <magnitude>, <duty_cycle> or <frequency> are a tuple of two scalars (only one of these can be
            a tuple at once), then the <datastream> will modulate that parameter between the two values in the tuple
            (The first item of the tuple corresponds to <value=0>, the second to <value=1>.)
            If none of these parameters is explicitly passed as a tuple, the <magnitude> is used as modulated parameter
            with minimal value of 0.
            In the case of <frequency>, the two values are interpolated logarithmically by the <datastream>.
                
            @param n [float] (2): The maximum number of Monte Carlo steps performed during each phase of input_single().
            NOTE: If one only considers the state at the end of an input bit, then changing duty cycle is similar to changing frequency.
        """
        if datastream is None: datastream = hotspice.io.RandomScalarDatastream()
        super().__init__(datastream)
        ## Determine which of the parameters should be modulated by the datastream
        total_tuples = isinstance(magnitude, tuple) + isinstance(dutycycle, tuple) + isinstance(frequency, tuple)
        if total_tuples == 0: magnitude = (0, magnitude) # No parameter was explicitly passed as tuple, so by default <magnitude> is treated as the modulated parameter with 0 as lower limit.
        self.set_magnitude(magnitude)
        self.set_dutycycle(dutycycle)
        self.set_frequency(frequency)
        self.n = n
        
    def set_magnitude(self, magnitude: float|tuple[float,float]):
        self.magnitude_min, self.magnitude_max = magnitude if isinstance(magnitude, tuple) else (magnitude, magnitude)
    def set_dutycycle(self, dutycycle: float|tuple[float,float]):
        self.dutycycle_min, self.dutycycle_max = dutycycle if isinstance(dutycycle, tuple) else (dutycycle, dutycycle)
        if not (0 <= self.dutycycle_min <= 1) or not (0 <= self.dutycycle_max <= 1):
            raise ValueError("Dutycycle must lay within range [0, 1].")
    def set_frequency(self, frequency: float|tuple[float,float]):
        self.frequency_min, self.frequency_max = frequency if isinstance(frequency, tuple) else (frequency, frequency)
        if self.frequency_max < 0 or self.frequency_min < 0:
            raise ValueError("Frequency must be positive.")

    def input_single(self, mm: hotspice.Magnets, value: float|int, stepwise: bool = False):
        if mm.params.UPDATE_SCHEME != hotspice.Scheme.NEEL: raise RuntimeError("Can only apply this Inputter with a Néel update scheme.")
        Zeeman: hotspice.ZeemanEnergy = mm.get_energy('Zeeman')
        if Zeeman is None: mm.add_energy(Zeeman := hotspice.ZeemanEnergy(0, 0))
        magnitude = self.magnitude_min + value*(self.magnitude_max - self.magnitude_min) # Linear interpolation
        dutycycle = self.dutycycle_min + value*(self.dutycycle_max - self.dutycycle_min) # Linear interpolation
        frequency = self.frequency_min*10**(value*np.log10(self.frequency_max/self.frequency_min)) # Exponential interpolation
        Zeeman.set_field(magnitude=magnitude)
        x = mm.progress(t_max=dutycycle/frequency, MCsteps_max=self.n, stepwise=stepwise)
        if stepwise: yield from x
        Zeeman.set_field(magnitude=0)
        x = mm.progress(t_max=(1 - dutycycle)/frequency, MCsteps_max=self.n, stepwise=stepwise)
        if stepwise: yield from x

#! WHAT IS THE BEST INPUTTER? MANY VARIATIONS OF 'WIGGLING' BETWEEN STATES EXIST, LIKE:
#   - Just applying a uniform field, that's the easiest one but likely won't work very well because it does not really use the spatial memory, but it is easiest to fabricate
#     -> The problem with this is that this effectively only uses the logarithmic decay or growth of the average magnetization, so memory can be present but will be quite minimal. Nothing that can't be done with something else than an ASI.
#   - Only applying one half-checkerboard at once, and the direction of that field on that part of the checkerboard can be varied to promote states, but then we might get closer to binary input again
# TODO: make a list of all the various possible 'wiggling' procedures, their advantages, disadvantages, practicality, whether they can easily be done scalar instead of binary, ...


if __name__ == "__main__":
    # TODO: Next quest: see how the decay time changes as the 1/r^3 DD interaction is changed to extend further.

    ## DD INTERACTION AND DISTANCE-DEPENDENCE
    # 1) Is the DD interaction 1/r³ or relatively constant?
    # 2) If it is 1/r³, then NN has an extra factor of 1.75, NNN 1.25, NNNN 1.2... due to the nonzero size of the magnets

    ## CREATE HOTSPICE OBJECTS
    mm = get_thermal_mm(E_B_ratio=20, DD_ratio=2.5, E_B_std=0.0, size=20, gradient=0.1, pattern='uniform')
    inputter = UniformInputter(hotspice.io.RandomScalarDatastream(), magnitude=(0, 3e-4), n=2, frequency=1e4)
    # limit_DD_orthogonal(mm.get_energy('dipolar'))

    ## SHOW GUI
    gui = hotspice.gui.GUI(mm, inputter=inputter, editable=True)
    gui.show()
