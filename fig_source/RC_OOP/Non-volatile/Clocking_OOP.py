""" This script implements a clocking scheme in OOP Square ASI. """
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np

from matplotlib import colormaps, colors
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import hotspice
import thesis_utils

from nonvolatile_ASI import get_nonvolatile_mm


def run(N: int = 13, E_B_std: float = 0.05, E_EA_ratio: float = 100, E_MC_ratio: float = 25,
        magnitude: float = 0.00185, vacancy_fraction: float = 0., size: int = 50):
    magnet_size_ratio = 170/(170+30) # S_ASI = 30nm, D_NM = 170nm
    mm = get_nonvolatile_mm(E_EA_ratio=E_EA_ratio, E_MC_ratio=E_MC_ratio, E_B_std=E_B_std,
                            size=size, magnet_size_ratio=magnet_size_ratio)
    vacancies = int(mm.n*vacancy_fraction)
    mm.occupation[np.random.randint(mm.ny, size=vacancies), np.random.randint(mm.nx, size=vacancies)] = 0
    inputter = hotspice.io.OOPSquareChessStepsInputter(hotspice.io.RandomBinaryDatastream(), magnitude=magnitude)
    
    states, domains = [], []
    values = []
    mm.initialize_m('AFM', angle=np.pi)
    for i in range(N):
        if i == 0: values.append(None)
        else: values.append(inputter.input(mm, values=(i < (N//2 + 1)))[0])
        states.append(np.where(mm.occupation == 0, np.nan, mm.m))
        domains.append(np.where(mm.occupation == 0, np.nan, mm.get_domains()))
        # hotspice.gui.show(mm)
    
        
    ## Save
    hotspice.utils.save_results(parameters={"magnitude": magnitude, "E_MC_ratio": E_MC_ratio, "E_EA_ratio": E_EA_ratio, "E_B_std": E_B_std,
                                            "vacancy_fraction": vacancy_fraction, "vacancies": vacancies, "a": mm.a, "d": mm.a*magnet_size_ratio,
                                            "T": mm.T_avg, "PBC": mm.PBC, "scheme": mm.params.UPDATE_SCHEME, "size": size, "moment": mm.moment_avg,
                                            "N": N, "ASI_type": "OOP_Square"},
                                data={"values": values, "states": states, "domains": domains})
    plot()


def plot(data_dir=None, show_domains=True):
    """ (Re)plots the figures in the `data_dir` folder.
        If `data_dir` is not specified, the most recently created directory in <filename>.out is used.
    """
    ## Load data
    if data_dir is None: data_dir = thesis_utils.get_last_outdir()
    params, data = hotspice.utils.load_results(data_dir)
    
    ## Plot
    thesis_utils.init_style()
    fig, axes = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(thesis_utils.page_width, thesis_utils.page_width/7*2))
    fig.subplots_adjust(top=1, bottom=0, left=0, right=1)
    ax: plt.Axes = axes[0,0]
    ax.set_aspect('equal')
    ax.set_axis_off()
    fs = thesis_utils.fs_large + 1

    ## Setup plot variables
    OOP = params['ASI_type'] == "OOP_Square"
    if OOP:
        if show_domains:
            OOPcmap = mpl.colormaps.get_cmap('gray')  # viridis is the default colormap for imshow
            OOPcmap.set_bad(color='red')
        else:
            cmap = colormaps['hsv']
            r0, g0, b0, _ = cmap(.5) # Value at angle 'pi' (-1)
            r1, g1, b1, _ = cmap(0) # Value at angle '0' (1)
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

    imfrac = 0.8 # How wide the imshows are w.r.t. their spacing
    x, y = -1, 0
    ax.set_xlim([-imfrac/2, imfrac/2])
    ax.set_ylim([-imfrac/2, imfrac/2])
    ## Plot all states
    for i, state in enumerate(data['states']):
        nextrow = (data['values'][i] != data['values'][i-1]) if i > 1 else False
        dy = -1 if nextrow else 0
        dx = 0 if nextrow else -(y % 2)*2 + 1
        x, y = x + dx, y + dy
        ax.set_xlim(min(x-.5, ax.get_xlim()[0]), max(x+.5, ax.get_xlim()[1]))
        ax.set_ylim(min(y-.5, ax.get_ylim()[0]), max(y+.5, ax.get_ylim()[1]))
        if OOP:
            image = (1 - data['domains'][i]) if show_domains else (state + 1)/2
        else:
            mm = hotspice.ASI.IP_Pinwheel(a=1, n=params['size'])
            image = hotspice.plottools.get_rgb(mm, m=state, avg=avg, fill=True)
        ax.imshow(image, extent=[x-imfrac/2, x+imfrac/2, y-imfrac/2, y+imfrac/2],
                  vmin=0, vmax=1, cmap=OOPcmap if OOP else 'hsv', origin='lower')
        ax.add_patch(Rectangle((x-imfrac/2, y-imfrac/2), imfrac, imfrac, fill=False, color="gray", linewidth=1))
        # Draw arrow from previous
        if i == 0: continue # No arrow drawn before the first image
        text = '$A$' if data['values'][i] else '$B$'
        if nextrow: annotate_connection(ax, text, x, y+1-imfrac/2, x, y+imfrac/2, opposite_side=y%2, text_pad=3, text_size=fs)
        else: annotate_connection(ax, text, x+(-1+imfrac/2)*(1 if dx > 0 else -1), y, x+(-imfrac/2)*(1 if dx > 0 else -1), y, text_size=fs)
    
    ## Draw "legend" on the axes
    lw, color = 1, "k"
    x, y = 0, -1 # The index of the empty spot in the figure, DON'T CHANGE
    ys = [y + 0.15, y - 0.25]
    x_square  = x - 0.25
    wh_square = 0.2 # Width/height of the black/white square
    r_magnets = 0.08 # Radius of circles indicating magnets
    x_equals  = x_square + wh_square
    x_magnets = x_equals + wh_square/2 + 2*r_magnets
    
    ax.text(x_magnets, ys[0] + r_magnets*2, r"$\boldsymbol{\mu}$", va="bottom", ha="center", color=color, fontdict=dict(size=fs)) # Bold mu symbol for tiny states
    for i in range(2): # Black (i=0) and white (i=1) states
        # The square
        ax.add_patch(Rectangle((x_square - wh_square/2, ys[i] - wh_square/2), wh_square, wh_square, fill=None if i else "k", color=color, linewidth=lw))
        
        # Equals sign
        ax.text(x_equals, ys[i], r"=", va="center", ha="center", color=color, fontdict=dict(size=fs))
        
        # The four magnet icons
        for ix in range(2):
            for iy in range(2):
                pos = (x_magnets + (2*ix-1)*r_magnets, ys[i] + (2*iy-1)*r_magnets)
                draw_Zvector(ax, (ix + iy + i) % 2, pos[0], pos[1], r_magnets*0.9)
    
    ## Save the result
    hotspice.utils.save_results(figures={'OOP_Square_clocking': fig}, outdir=data_dir, copy_script=False)


def draw_Zvector(ax: plt.Axes, up: bool, x: float, y: float, r: float, lw: float = 1, color='k'):
    """ If 'up' is True, then a dot is drawn, otherwise a cross, in line with the +z/-z symbols for vectors. """
    ax.add_patch(Circle((x, y), r, fill=False, lw=lw, color=color))
    if up: # Draw cross
        xarr = [x - r/np.sqrt(2), x + r/np.sqrt(2)]
        yarr = [y - r/np.sqrt(2), y + r/np.sqrt(2)]
        ax.plot(xarr, yarr, lw=lw, color=color, solid_capstyle="butt")
        ax.plot(xarr, yarr[::-1], lw=lw, color=color, solid_capstyle="butt")
    else: # Draw dot
        ax.scatter([x], [y], s=((lw*200)/ax.get_figure().dpi)**2, edgecolors="none", color=color)

def annotate_connection(ax: plt.Axes, text, x1, y1, x2, y2, color='k', opposite_side: bool = False, text_pad=3, text_size=None):
    fancyarrowkwargs = dict(posA=(x1, y1), posB=(x2, y2), color=color, lw=1, linestyle='-', zorder=1, shrinkA=0, shrinkB=0)
    ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='->', mutation_scale=8))
    
    ln_slope = 1 + int(np.sign((x2 - x1)*(y2 - y1))) # Determine slope direction of line to set text anchors correctly
    ha = ["right", "center", "left"][::(-1 if opposite_side else 1)][ln_slope]
    va = "baseline" if opposite_side else "top"
    if x1 == x2: ha, va = "left" if opposite_side else "right", "center_baseline" # Special case: vertical line
    offset_x = [text_pad, 0, -text_pad][::(-1 if opposite_side else 1)][ln_slope] if x1 != x2 else text_pad*(1 if opposite_side else -1)
    offset_y = text_pad*(1 if opposite_side else -1)*(va != "center_baseline")
    text_offset = transforms.offset_copy(ax.transData, x=offset_x, y=offset_y, units="points", fig=ax.get_figure())
    ax.text(x=np.mean((x1, x2)), y=np.mean((y1, y2)), s=text, color=color, ha=ha, va=va, transform=text_offset, fontdict=dict(size=text_size))


if __name__ == "__main__":
    ## OLD SYSTEMS
    # run(E_B_std=0.1, E_EA=hotspice.utils.eV_to_J(1), moment=1.6e-16, magnitude=0.003, vacancy_fraction=0.01) # Here, vacancies are important.
    # run(E_B_std=0.01, E_EA=hotspice.utils.eV_to_J(60), moment=2.37e-16, magnitude=0.0615, vacancy_fraction=0.01)
    # run(E_B_std=0.05, E_EA=hotspice.utils.eV_to_J(60), moment=2.37e-16, magnitude=0.048)
    
    ## THESIS SYSTEMS (preliminary)
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=25, magnitude=0.00185) # Region II (magnitude range 0.0018-0.0019 works)
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=10, magnitude=0.00145) # Region I (magnitude range 0.00145-0.0015 works decently, up to 0.002 stuff happens)
    # run(E_B_std=0.05, E_EA_ratio=100, E_MC_ratio=200, magnitude=0.0055, size=20) # Region III
    
    thesis_utils.replot_all(plot)
    # plot(show_domains=True)
    