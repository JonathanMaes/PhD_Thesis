import hotspice
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import os
import pandas as pd

from enum import Enum, auto
from matplotlib.patches import Annulus, Ellipse, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import thesis_utils

mumax_data_dir = os.path.join(os.path.dirname(__file__), "Fig_MS_distance_mumax") # Directory containing OOP.mx3, IP_parallel.mx3, IP_antiparallel.mx3

class Types(Enum):
    OOP = auto()
    IP_PARALLEL = auto()
    IP_ANTIPARALLEL = auto()


def mumax_csv(scriptfile: str) -> pd.DataFrame:
    scriptfile = os.path.join(mumax_data_dir, scriptfile)
    tablefile = os.path.join(mumax_data_dir, os.path.splitext(scriptfile)[0] + ".out", "table.txt")
    if not os.path.exists(tablefile):
        os.system(f'mumax3 "{scriptfile}"')
    return pd.read_csv(tablefile, sep="\t")


def get_data_OOP(scale: bool = False, **kwargs):
    moment = (Msat := 1063242)*(t_lay := 1.4e-9)*(n_lay := 7)*np.pi*((d := 170e-9)/2)**2
    # Mumax
    data = mumax_csv("OOP.mx3")
    E_mumax = data["E_MC (J)"].to_numpy()
    distances = data["Distance (d)"].to_numpy()*d
    # Point dipole approximation
    E_dipole = 1e-7*moment**2*(-1/distances**3)
    # Point dipole approximation with second-order correction
    I_ij = ((d/2)**2 + (d/2)**2)/4
    E_dipole_finite = E_dipole + 1e-7*moment**2*(3/2*I_ij)*(-3/distances**5)
    # Dumbbell approximation
    dumbbell_d = t_lay*n_lay
    with np.errstate(divide='ignore'):
        E_dumbbell = 1e-7*moment**2/dumbbell_d**2*(-2/distances + 2/np.sqrt(distances**2+dumbbell_d**2))
    
    factor = -1e-7*moment**2 if scale else 1
    return {"distances": distances/d, "dipole": E_dipole/factor, "dipole_finite": E_dipole_finite/factor, "dumbbell": E_dumbbell/factor, "mumax": E_mumax/factor}


def get_data_IP_parallel(scale: bool = False, dumbbell_ratio: float = 1, **kwargs):
    moment = (Msat := 1063242)*(t := 10e-9)*np.pi*(l := 220e-9)*(b := 80e-9)/4
    # Mumax
    data = mumax_csv("IP_parallel.mx3")
    E_mumax = data["E_MC (J)"].to_numpy()
    distances = data["Distance (l)"].to_numpy()*l
    # Point dipole approximation
    E_dipole = 1e-7*moment**2*(-2/distances**3)
    # Point dipole approximation with second-order correction
    I_ij = ((l/2)**2 + (b/2)**2)/4
    E_dipole_finite = E_dipole + 1e-7*moment**2*(3/2*I_ij)*(-4/distances**5)
    # Dumbbell approximation
    dumbbell_d = l*dumbbell_ratio #! Use l*0.9 for a good correspondence to mumax curve.
    with np.errstate(divide='ignore'):
        E_dumbbell = 1e-7*moment**2/dumbbell_d**2*(2/distances - 1/(distances-dumbbell_d) - 1/(distances+dumbbell_d))
    
    factor = -1e-7*moment**2 if scale else -1
    return {"distances": distances/l, "dipole": E_dipole/factor, "dipole_finite": E_dipole_finite/factor, "dumbbell": E_dumbbell/factor, "mumax": E_mumax/factor}


def get_data_IP_antiparallel(scale: bool = False, dumbbell_ratio: float = 1, **kwargs):
    moment = (Msat := 1063242)*(t := 10e-9)*np.pi*(l := 220e-9)*(b := 80e-9)/4
    # Mumax
    data = mumax_csv("IP_antiparallel.mx3")
    E_mumax = data["E_MC (J)"].to_numpy()
    distances = data["Distance (w)"].to_numpy()*b
    # Point dipole approximation
    E_dipole = 1e-7*moment**2*(-1/distances**3)
    # Point dipole approximation with second-order correction
    I_ij = ((l/2)**2 + (b/2)**2)/4
    E_dipole_finite = E_dipole + 1e-7*moment**2*(3/2*I_ij)*(-1/distances**5)
    # Dumbbell approximation
    dumbbell_d = l*dumbbell_ratio #! Use l*0.9 for a good correspondence to mumax curve.
    with np.errstate(divide='ignore'):
        E_dumbbell = 1e-7*moment**2/dumbbell_d**2*(2/np.sqrt(distances**2 + dumbbell_d**2) - 2/distances)
    
    factor = -1e-7*moment**2 if scale else -1
    return {"distances": distances/b, "dipole": E_dipole/factor, "dipole_finite": E_dipole_finite/factor, "dumbbell": E_dumbbell/factor, "mumax": E_mumax/factor}


def inset_ax(ax: plt.Axes, ASI_type: Types, fig: plt.Figure = None):
    if fig is None: fig = plt.gcf()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height # Width and height of ax in inches
    inset_width = 0.95*width # Width of inset ax in inches
    inset_ax: plt.Axes = inset_axes(ax, width=inset_width, height=inset_width, loc='upper right',
                        bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, borderpad=0)
    # inset_ax.set_aspect('equal', adjustable='datalim')
    inset_ax.axis('off')
    inset_ax.patch.set_alpha(0) # Transparent background
    
    def draw_ellipse(x, y, rx, ry=None, **kwargs):
        if ry is None: ry = rx
        inset_ax.add_patch(Ellipse((x, y), width=2*rx, height=2*ry, **kwargs))
    
    def draw_charge(x, y, r, color="black"):
        draw_ellipse(x, y, r/2, facecolor=color, edgecolor="none")
        inset_ax.add_patch(Annulus((x, y), r=r, width=r/4, facecolor=color, edgecolor="none"))
    
    def annotate_distance(text, x1, y1, x2, y2, color='k', opposite_side: bool = False, text_pad=3, endlines=False):
        fancyarrowkwargs = dict(posA=(x1, y1), posB=(x2, y2), color=color, lw=1, linestyle='-', zorder=1, shrinkA=0, shrinkB=0)
        inset_ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='<|-|>', mutation_scale=8))
        if endlines: inset_ax.add_artist(FancyArrowPatch(**fancyarrowkwargs, arrowstyle='|-|', mutation_scale=2))
        
        ln_slope = 1 + int(np.sign((x2 - x1)*(y2 - y1))) # Determine slope direction of line to set text anchors correctly
        ha = ["right", "center", "left"][::(-1 if opposite_side else 1)][ln_slope]
        va = "baseline" if opposite_side else "top"
        if x1 == x2: ha, va = "left" if opposite_side else "right", "center" # Special case: vertical line
        text_offset = transforms.offset_copy(inset_ax.transData, x=[text_pad, 0, -text_pad][::(-1 if opposite_side else 1)][ln_slope],
                                             y=text_pad*(1 if opposite_side else -1)*(va != "center"), units="points", fig=fig)
        inset_ax.text(x=np.mean((x1, x2)), y=np.mean((y1, y2)), s=text, color=color, ha=ha, va=va, transform=text_offset, fontsize=thesis_utils.fs_large+1)

    r, padding = 0.19, 0.05
    d = 0.9
    wl_ratio = 4/11 # w/l ratio for ellipses
    colors = "gray", "gray" # "blue", "red"
    match ASI_type:
        case Types.OOP:
            rx = ry = r
            text = "$2r$"
        case Types.IP_PARALLEL:
            rx = r
            ry = wl_ratio*r
            text = "$l$"
        case Types.IP_ANTIPARALLEL:
            rx = wl_ratio*r
            ry = r
            text = "$w$"
    y = 1 - padding - r
    x2 = 1 - padding - rx
    x1 = x2 - rx - padding - rx
    draw_ellipse(x1, y, rx=rx, ry=ry, color=colors[0], alpha=0.9)
    draw_ellipse(x2, y, rx=rx, ry=ry, color=colors[1], alpha=0.9)
    annotate_distance("$r_{ij}$", x1, y - ry - padding/2, x2, y - ry - padding/2, endlines=True)
    match ASI_type:
        # case Types.OOP:
        #     inset_ax.text(x=x1, y=y, s=r"$\otimes$", color='k', ha="center", va="center")
        #     inset_ax.text(x=x2, y=y, s=r"$\odot$", color='k', ha="center", va="center")
        case Types.IP_PARALLEL:
            draw_charge(x1 - d*rx, y, r*0.15, color="blue")
            draw_charge(x1 + d*rx, y, r*0.15, color="red")
            draw_charge(x2 - d*rx, y, r*0.15, color="blue")
            draw_charge(x2 + d*rx, y, r*0.15, color="red")
            annotate_distance("$d=0.9l$", x2 - d*rx, y, x2 + d*rx, y, opposite_side=True, text_pad=12, color="C2")
        case Types.IP_ANTIPARALLEL:
            draw_charge(x1, y - d*ry, r*0.15, color="red")
            draw_charge(x1, y + d*ry, r*0.15, color="blue")
            draw_charge(x2, y - d*ry, r*0.15, color="blue")
            draw_charge(x2, y + d*ry, r*0.15, color="red")
            # annotate_distance("$d=0.9l$", x2 - d*rx, y, x2 + d*rx, y, opposite_side=True, text_pad=6)
    annotate_distance(text, x1 - rx, y, x1 + rx, y, opposite_side=True, text_pad=12 if ASI_type == Types.IP_PARALLEL else 3, endlines=True)
            

def show_MS_distance_fig():
    """ Shows a comparison between the MS interaction in OOP_Square as a function of distance,
        calculated using various methods:
            - The point dipole approximation
            - The point dipole approximation with second-order correction
            - The dumbbell approximation
            - mumax³
    """
    get_data_kwargs = {'scale': True, 'dumbbell_ratio': 0.9}
    data = {Types.OOP: get_data_OOP(**get_data_kwargs), Types.IP_PARALLEL: get_data_IP_parallel(**get_data_kwargs), Types.IP_ANTIPARALLEL: get_data_IP_antiparallel(**get_data_kwargs)}
    titles = {Types.OOP: "OOP", Types.IP_PARALLEL: r"IP $\rightarrow\rightarrow$", Types.IP_ANTIPARALLEL: r"IP $\uparrow\downarrow$"}
    
    thesis_utils.init_style()
    fig, axes = plt.subplots(1, len(data), figsize=(thesis_utils.page_width, 2.7))
    for i, (ASI_type, data_i) in enumerate(data.items()):
        ax: plt.Axes = axes[i]
        scale_y = 1e20
        title = titles[ASI_type]
        dumbbell_is_dipole = np.allclose(data_i['dipole'], data_i['dumbbell'], rtol=1e-2)
        # The following scale_y is very ad-hoc: scale_y = -np.nanmax(np.abs([data_i['mumax'], data_i['dipole'], data_i['dipole_finite'], data_i['dumbbell']])) # -1e-7*moment**2
        markevery, markersize = 0.1, 5
        ax.plot(data_i['distances'], data_i['dipole']/scale_y, label="Point dipole",
                marker=thesis_utils.marker_cycle[0], markevery=(0, markevery), ms=markersize)
        ax.plot(data_i['distances'], data_i['dipole_finite']/scale_y, label="Finite dipole",
                marker=thesis_utils.marker_cycle[1], markevery=(0, markevery), ms=markersize)
        ax.plot(data_i['distances'], data_i['dumbbell']/scale_y, label="Dumbbell" + (f" $d={get_data_kwargs['dumbbell_ratio']:.1f}l$" if get_data_kwargs['dumbbell_ratio'] != 1 else ""),
                marker=thesis_utils.marker_cycle[2], markevery=(markevery/2, markevery), ms=markersize,
                linestyle=(0,(3,3)) if dumbbell_is_dipole else '-')
        ax.plot(data_i['distances'], data_i['mumax']/scale_y, label="mumax³", color='k', ls='--')
        # ax.set_title(title, fontsize=10)
        if i == 0: ax.set_ylabel(r"$|E_\mathrm{MS}|/\mu^2$ [a.u.]" if get_data_kwargs['scale'] else r"$E_{MS}$ (J)", fontdict=dict(fontsize=thesis_utils.fs_large))
        else: ax.set_yticklabels([])
        ax.set_xlabel([r"$r_{ij}/2r$", r"$r_{ij}/l$", r"$r_{ij}/w$"][i], fontdict=dict(fontsize=thesis_utils.fs_large))
        ax.axvline(1, linestyle=':', color='grey', linewidth=1)
        ax.set_xlim(right=data_i['distances'].max())
        # ax.set_ylim(bottom=0, top=data_i['mumax'].max()/scale_y*1.05)
        ax.set_ylim(bottom=0, top=5)
        inset_ax(ax, ASI_type=ASI_type)
    # fig.supxlabel("Normalized center-to-center distance", fontsize=10, x=0.52) # "Normalized" means that distance was divided by the distance at which the magnets barely touch.
    leg = fig.legend(*ax.get_legend_handles_labels(), ncol=4, loc="upper center")
    thesis_utils.move_legend(leg, ax, dx=0.04, dy=0.1)
    fig.tight_layout()
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.065, right=0.99)
    hotspice.utils.save_results(data=data, figures={"MS_distance": fig}, timestamped=False)


if __name__ == "__main__":
    show_MS_distance_fig()
    # thesis_utils.replot_all()