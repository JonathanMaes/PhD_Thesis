""" This script implements a clocking scheme in OOP Square ASI. """
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from matplotlib import colormaps, colors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Ellipse
from scipy.signal import convolve2d

import hotspice
import thesis_utils

from nonvolatile_ASI import get_nonvolatile_mm


def run(N: int = 13, E_B_std: float = 0.05, E_EA_ratio: float = 100, E_MC_ratio: float = 25, repeats: int = 1,
        magnitude: float = 0.00185, vacancy_fraction: float = 0., size: int = 50, finite: bool = True, pattern: str = "AFM"):
    magnet_size_ratio = 170/(170+30)*finite # S_ASI = 30nm, D_NM = 170nm
    mm = get_nonvolatile_mm(E_EA_ratio=E_EA_ratio, E_MC_ratio=E_MC_ratio, E_B_std=E_B_std,
                            size=size, magnet_size_ratio=magnet_size_ratio)
    vacancies = int(mm.n*vacancy_fraction)
    mm.occupation[np.random.randint(mm.ny, size=vacancies), np.random.randint(mm.nx, size=vacancies)] = 0
    inputter = hotspice.io.OOPSquareChessStepsInputter(hotspice.io.RandomBinaryDatastream(), magnitude=magnitude)
    
    states, domains = [], []
    values = []
    match pattern:
        case "seed":
            mm.initialize_m("AFM", angle=np.pi)
            mm.m[size//2, size//2] *= -1
            mm.update_energy()
        case _:
            mm.initialize_m(pattern, angle=np.pi)
    # hotspice.gui.show(mm, inputter=inputter)
    for i in range(N):
        if i == 0: values.append(None)
        else: values.append(inputter.input(mm, values=[(i < (N//2 + 1))]*repeats)[0])
        states.append(np.where(mm.occupation == 0, np.nan, mm.m))
        domains.append(np.where(mm.occupation == 0, np.nan, mm.get_domains()))
        # hotspice.gui.show(mm)
    
        
    ## Save
    hotspice.utils.save_results(parameters={"magnitude": magnitude, "E_MC_ratio": E_MC_ratio, "E_EA_ratio": E_EA_ratio, "E_B_std": E_B_std,
                                            "vacancy_fraction": vacancy_fraction, "vacancies": vacancies, "a": mm.a, "d": mm.a*magnet_size_ratio,
                                            "T": mm.T_avg, "PBC": mm.PBC, "scheme": mm.params.UPDATE_SCHEME.name, "size": size, "moment": mm.moment_avg,
                                            "N": N, "ASI_type": "OOP_Square", "repeats": repeats},
                                data={"values": values, "states": states, "domains": domains})
    plot()


def plot(data_dir=None, label_domains=None, label_moments=None, highlight_leaking=False):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    if params is None and data is None: return
    N = params.get('N', len(data['values'])) if params is not None else len(data['values']) # Number of panels
    repeats = params.get('repeats', 1) # Number of repeated A- or B-cycles between state snapshots
    
    ## Plot
    thesis_utils.init_style(style="default")
    figs = {}
    for show_domains in (True, False):
        figsize=(thesis_utils.page_width, thesis_utils.page_width/2)
        aspect = figsize[0]/figsize[1]
        fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=figsize)
        fig.subplots_adjust(top=1, right=1, bottom=0, left=0)
        ax: plt.Axes = axes[0,0]
        ax.set_aspect('equal')
        ax.set_axis_off()
        fs = min(thesis_utils.fs_large + 1, 180/N)

        ## Setup plot variables
        OOP = params['ASI_type'] == "OOP_Square"
        if OOP:
            if show_domains:
                OOPcmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
                OOPcmap.set_bad(color='red')
            else:
                cmap = colormaps['inferno']
                r0, g0, b0 = colors.to_rgb("C0") # Value at angle 'pi' (-1)
                r1, g1, b1, _ = cmap(0.85) # Value at angle '0' (1)
                cdict = {
                    'red':   [[0.0, r0,  r0], # x, value_left, value_right
                            [0.5, 0.0, 0.0],
                            [1.0, r1,  r1]],
                    'green': [[0.0, g0,  g0],
                            [0.5, 0.0, 0.0],
                            [1.0, g1,  g1]],
                    'blue':  [[0.0, b0,  b0],
                            [0.5, 0.0, 0.0],
                            [1.0, b1,  b1]]
                }
                OOPcmap = colors.LinearSegmentedColormap('OOP_cmap', segmentdata=cdict, N=256)
                OOPcmap.set_bad(color='k')
        else:
            if show_domains:
                avg = hotspice.plottools.Average.SQUAREFOUR
            else:
                avg = hotspice.plottools.Average.POINT

        imfrac = 0.8
        r = N/4
        a, b = r, r/aspect
        dangle = 2*np.pi/N
        color = (30/255,100/255,200/255) #"C0"
        
        def get_xy(i):
            from scipy.special import ellipeinc, ellipe
            from scipy.optimize import root_scalar
            e2 = 1 - (b**2 / a**2)
            def inverse_ellipeinc(target, m):
                res = root_scalar(lambda phi: ellipeinc(phi, m) - target, bracket=[0, 2*np.pi], method='brentq')
                return res.root if res.converged else None
            phi = inverse_ellipeinc(i/N*ellipeinc(2*np.pi, e2), e2)
            theta = phi + np.arctan((a-b)*np.tan(phi)/(b + a*np.tan(phi)**2))
            return -a*np.cos(theta), b*np.sin(theta)
        
        ax.set_xlim([-imfrac/2, imfrac/2])
        ax.set_ylim([-imfrac/2, imfrac/2])
        ax.add_patch(Ellipse((0,0), width=2*r, height=2*r/aspect, fill=False, color=color, zorder=-1))
        ## Plot all states
        for i, state in enumerate(data['states']):
            # Draw arrow from previous
            arrow_kwargs = dict(arrowstyle='-|>', mutation_scale=12, color=color, lw=1.5, zorder=-1)
            # ax.arrow(*get_xy(i+0.49), *get_xy(i+0.51), shape='full', lw=0, length_includes_head=False, head_width=1)
            # ax.add_patch(FancyArrowPatch(get_xy(i), get_xy(i + .5), connectionstyle=f"arc3,rad={dangle/2}", **arrow_kwargs))
            # ax.add_patch(FancyArrowPatch(get_xy(i + .5), get_xy(i + 1), **arrow_kwargs))
            ax.add_patch(FancyArrowPatch(get_xy(i+0.5), get_xy(i+0.6), **arrow_kwargs))
            
            x, y = get_xy(i)
            ax.set_xlim(min(x-.5, ax.get_xlim()[0]), max(x+.5, ax.get_xlim()[1]))
            ax.set_ylim(min(y-.5, ax.get_ylim()[0]), max(y+.5, ax.get_ylim()[1]))
            if OOP:
                image = (1 - data['domains'][i]) if show_domains else (state + 1)/2
            else:
                mm = hotspice.ASI.IP_Pinwheel(a=1, n=params['size'])
                image = hotspice.plottools.get_rgb(mm, m=state, avg=avg, fill=True)
            ax.imshow(image, extent=[x-imfrac/2, x+imfrac/2, y-imfrac/2, y+imfrac/2],
                    vmin=0, vmax=1, cmap=OOPcmap if OOP else 'hsv', origin='lower')
            ax.add_patch(Rectangle((x-imfrac/2, y-imfrac/2), imfrac, imfrac, fill=False, color=color, linewidth=1))
            if highlight_leaking and show_domains and i > 0:
                y_leaks, x_leaks = get_leaked_magnets(data['domains'][i], data['domains'][i-1], cycles=repeats).nonzero()
                ny, nx = image.shape
                ax.scatter(x + imfrac*((.5 + x_leaks)/nx - .5), y + imfrac*((.5 + y_leaks)/ny - .5),
                           color="red", s=(100/ax.get_figure().dpi)**2)
        figs['Clocking_OOP' + ('domains' if show_domains else 'moments')] = fig
    
    ## Save the result
    hotspice.utils.save_results(figures=figs, outdir=data_dir, copy_script=False)
    plt.close()


def get_leaked_magnets(domains, prev_domains, cycles=1): # TODO: find a way to not include nucleation
    switched = domains != prev_domains # Leaking obviously only occurs at switched magnets
    spread = np.copy(2*prev_domains-1)
    mask = [[0,0,1,0,0], [0,1,1,1,0], [1,1,1,1,1], [0,1,1,1,0], [0,0,1,0,0]]
    total = convolve2d(np.ones_like(spread), mask, mode='same')
    for _ in range(cycles): spread = convolve2d(spread, mask, mode='same')/total
    leaked = np.logical_and(switched, np.isclose(np.abs(spread), 1))
    injection_points = [[0,0], [-1,-1], [-1,0], [-1,1], [-2,0], [0,-1], [1,-1], [0,-2]] # Never mark these corners as 'leaked'
    for x, y in injection_points: leaked[y, x] = False
    return leaked


if __name__ == "__main__":
    thesis_utils.replot_all(plot, subdir="massive_forking")
    # thesis_utils.replot_all(plot, recursive=True)
    
    ## OLD SYSTEMS
    # run(E_B_std=0.1, E_EA=hotspice.utils.eV_to_J(1), moment=1.6e-16, magnitude=0.003, vacancy_fraction=0.01) # Here, vacancies are important.
    # run(E_B_std=0.01, E_EA=hotspice.utils.eV_to_J(60), moment=2.37e-16, magnitude=0.0615, vacancy_fraction=0.01)
    # run(E_B_std=0.05, E_EA=hotspice.utils.eV_to_J(60), moment=2.37e-16, magnitude=0.048)
    
    ## THESIS SYSTEMS (preliminary)
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=25, magnitude=0.00185) # Region II (magnitude range 0.0018-0.0019 works)
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=10, magnitude=0.00145) # Region I (magnitude range 0.00145-0.0015 works decently, up to 0.002 stuff happens)
    
    ## Clocking_clearly_EBstd0.pdf IS GENERATED WITH:
    # run(E_B_std=0.00, E_EA_ratio=200, E_MC_ratio=40, magnitude=0.00378, size=14, N=15)
    
    ## Clocking_clearly_EBstd5.pdf IS GENERATED WITH:
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=200, magnitude=0.0055, size=20, N=11) # Region III
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=200, magnitude=0.0055, size=64, N=41) # Region III but ridiculous
    # thesis_utils.replot_all(plot, subdir="20250306090630", highlight_leaking=True)
    
    ## Clocking_clearly_EBstd5_highEMC.pdf and _lowEMC.pdf ARE GENERATED WITH:
    # run(E_B_std=0.05, E_EA_ratio=200, E_MC_ratio=400, magnitude=0.0112, size=20, N=13)
    # thesis_utils.replot_all(plot, subdir="EBstd=5%/EMC_high", label_domains=0)
    # run(E_B_std=0.05, E_EA_ratio=200, E_MC_ratio=40, magnitude=0.00378, size=20, N=13)
    # thesis_utils.replot_all(plot, subdir="EBstd=5%/EMC_low", label_domains=1)
    
    ## Clocking_binary_KQ_states.pdf IS GENERATED WITH:
    # run(E_B_std=0.05, E_EA_ratio=455, E_MC_ratio=110, magnitude=0.01, finite=False, size=20, N=9)
    
    ## Clocking_massive_seeded.pdf IS GENERATED WITH:
    # run(E_B_std=0.00, E_EA_ratio=200, E_MC_ratio=40, magnitude=0.00378, size=141, N=13, repeats=6, pattern='seed')
