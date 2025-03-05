import hotspice

import numpy as np


def get_nonvolatile_mm(E_EA_ratio: float = 1, E_MC_ratio: float = 1, J_ratio: float = 0, E_B_std: float = 0.05, gradient: float = 0, size: int = 20, size_y: int = None, magnet_size_ratio: float = 0, MS_exponent: float = -3, T_factor: float = 1, pattern: str = 'AFM'):
    """ Creates a non-volatile ASI with
            E_B = normal distribution (mean: E_EA_ratio*kBT, standard deviation: E_B_std*E_B)
            NN MS = E_MC_ratio*kBT
        The temperature is always assumed to be 300 K.
        @param E_EA_ratio [float] (1): the mean energy barrier will be this multiple of kBT.
        @param E_MC_ratio [float] (1): the MS interaction energy between nearest-neighbors as a multiple of kBT.
        @param J_ratio [float] (1): the exchange interaction energy between neighbors as a multiple of kBT.
        @param E_B_std [float] (0.05): the standard deviation (as a relative fraction) on the energy barrier. Default 5%.
        @param gradient [float] (0): if nonzero, a horizontal gradient is applied to some parameter(s) (multiplicative 1±<gradient>).
            Currently, this gradient is only applied to the magnetic moment.
        @param size [int] (20): the number of magnets along a side of the system, so there are `size`x`size` magnets.
            Can also use `size_y` to specify the y-size separately (then `size` is used as the x-size).
        @param MS_exponent [float] (-3): the decay of the MS interaction as a function of distance.
            Can be used for simulating additional permalloy striplines, in combination with `limit_MS_orthogonal()`.
        @param T_factor [float] (1): <{E_B|MS|J}_ratio> are all relative to 300K. T_factor determines the actual temperature.
            This functionality was added to allow changing the temperature while keeping the physical system identical.
            This way, we can show that changing the temperature can shift the optimal frequency of a signal transformation.
        @param pattern [str] ('AFM'): the initial state of the system. Default: the ground state (AFM).
    """
    kBT = hotspice.kB*300 # [K]
    T = 300*T_factor
    if size_y is None: size_y = size
    dipolar_energy = hotspice.DipolarEnergy(prefactor=1.75, decay_exponent=MS_exponent) # prefactor gets overwritten by set_NN_interaction, so does not matter (1.75 is MuMax-inspired correction factor due to finite size of magnets)
    gradient_profile = np.tile(np.logspace(np.log10(1-gradient), np.log10(1+gradient), size), (size_y,1))
    mm = hotspice.ASI.OOP_Square(a=(a := (d := 170e-9) + (sep := 30e-9)),
                                 n=size, ny=size_y,
                                 moment=gradient_profile*(Msat := 1063242)*(t_lay := 1.4e-9)*(n_lay := 7)*np.pi*(d/2)**2,
                                 E_B=gradient_profile*E_EA_ratio*kBT*np.random.normal(1, E_B_std, size=(size_y, size)), T=T,
                                 energies=(hotspice.ZeemanEnergy(magnitude=0), dipolar_energy),
                                 major_axis=magnet_size_ratio*a,
                                 params=hotspice.SimParams(UPDATE_SCHEME=hotspice.Scheme.NEEL),
                                 pattern=pattern)
    dipolar_energy.set_NN_interaction(E_MC_ratio*kBT)
    if J_ratio != 0: mm.add_energy(hotspice.ExchangeEnergy(J=J_ratio*kBT))
    return mm