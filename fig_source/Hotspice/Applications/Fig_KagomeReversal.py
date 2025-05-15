import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import hotspice
from hotspice.energies import DipolarEnergy, DiMonopolarEnergy, ZeemanEnergy
from hotspice.utils import asnumpy, SIprefix_to_mul

import thesis_utils


def vertex(mm, skip):
    initial, type_I, type_II, inside, wrong, final = 0, 0, 0, 0, 0, 0
    red_x, red_y, blue_x, blue_y = [], [], [], []
    for y in range(len(mm.m)):
        if skip <= y <= len(mm.m)-skip-1:
            for x in range(len(mm.m[y])):
                if np.abs(mm.m[y,x]) == 1 and mm.angles[y,x] == np.pi/2 and skip <= x <= len(mm.m[y])-skip-1:
                    self, up_left, up_right, low_left, low_right = mm.m[y,x]*np.sign(mm.angles[y,x]), mm.m[y-1,x-1]*np.sign(mm.angles[y-1,x-1]), mm.m[y-1,x+1]*np.sign(mm.angles[y-1,x+1]), mm.m[y+1,x-1]*np.sign(mm.angles[y+1,x-1]), mm.m[y+1,x+1]*np.sign(mm.angles[y+1,x+1])
                    if (self == 1 and up_left == 1 and up_right == 1):
                        initial += 1
                    elif (self == 1 and up_left == -1 and up_right == 1) or (self == 1 and up_left == 1 and up_right == -1):
                        type_I += 1
                        red_x.append(mm.xx[y,x]*10**6)
                        red_y.append((mm.yy[y,x] - 288.6751346e-9)*10**6)
                    elif (self == -1 and up_left == 1 and up_right == 1):
                        type_II += 1
                        blue_x.append(mm.xx[y,x]*10**6)
                        blue_y.append((mm.yy[y,x] - 288.6751346e-9)*10**6)
                    elif (self == -1 and up_left == -1 and up_right == 1) or (self == -1 and up_left == 1 and up_right == -1):
                        inside += 1
                    elif (self == 1 and up_left == -1 and up_right == -1):
                        wrong += 1
                    else:
                        final += 1

                    if (self == 1 and low_left == 1 and low_right == 1):
                        initial += 1
                    elif (self == 1 and low_left == -1 and low_right == 1) or (self == 1 and low_left == 1 and low_right == -1):
                        type_I += 1
                        blue_x.append(mm.xx[y,x]*10**6)
                        blue_y.append((mm.yy[y,x] + 288.6751346e-9)*10**6)
                    elif (self == -1 and low_left == 1 and low_right == 1):
                        type_II += 1
                        red_x.append(mm.xx[y,x]*10**6)
                        red_y.append((mm.yy[y,x] + 288.6751346e-9)*10**6)
                    elif (self == -1 and low_left == -1 and low_right == 1) or (self == -1 and low_left == 1 and low_right == -1):
                        inside += 1
                    elif (self == 1 and low_left == -1 and low_right == -1):
                        wrong += 1
                    else:
                        final += 1
    total = initial + type_I + type_II + inside + wrong + final
    return initial/total*100, type_I/total*100, type_II/total*100, inside/total*100, wrong/total*100, final/total*100, red_x, red_y, blue_x, blue_y, total


def plot():
    # Best fit once was 240 and 13%, 120 and 13%
    plt.rcParams.update({'font.size': 25})

    a0 = 500e-9
    a = a0 / np.sin(30*np.pi/180)
    n = 120/4 #120
    d = 470e-9  # max 577.3502692
    T = 300
    t = 1
    moment = 1.1e-15
    np.random.seed(0)
    E_B_m = hotspice.utils.eV_to_J(120)  # Mumax JM said E_B < 250eV (first 120)
    E_B_m = E_B_m * np.random.normal(1, 5/100, (15,30))  # 67,120 (first 5%)

    np.random.seed(0)
    E_B_d = hotspice.utils.eV_to_J(120)  # Mumax JM zei E_B < 250eV
    E_B_d = E_B_d * np.random.normal(1, 5/100, (15,30))  # 67,120
    angle = np.pi/2 - 3.6*np.pi/180

    B_max_mono = 120/2000
    B_vals_mono = np.linspace(B_max_mono, 0, 1000)
    B_vals_mono = np.append(B_vals_mono, np.linspace(0, -B_max_mono, 1000))
    B_vals_mono = np.append(B_vals_mono, np.linspace(-B_max_mono, 0, 1000))
    B_vals_mono = np.append(B_vals_mono, np.linspace(0, B_max_mono, 1000))

    B_max_di = 120/2000
    B_vals_di = np.linspace(B_max_di, 0, 1000)
    B_vals_di = np.append(B_vals_di, np.linspace(0, -B_max_di, 1000))
    B_vals_di = np.append(B_vals_di, np.linspace(-B_max_di, 0, 1000))
    B_vals_di = np.append(B_vals_di, np.linspace(0, B_max_di, 1000))

    mm_mono = hotspice.ASI.IP_Kagome(a, n, moment=moment, energies=[DiMonopolarEnergy(d=d), ZeemanEnergy(magnitude=500, angle=np.pi/2)], pattern="uniform", T=T, E_B=E_B_m, m_perp_factor=0)
    mm_mono.params.UPDATE_SCHEME = hotspice.Scheme.NEEL
    mm_mono.progress(t)
    M_S = mm_mono.m_avg_y
    nonzero = mm_mono.m.nonzero()
    mm_old = asnumpy(np.multiply(mm_mono.m, mm_mono.orientation[:, :, 0])[nonzero])

    mm_di = hotspice.ASI.IP_Kagome(a, n, moment=moment, energies=[DipolarEnergy(), ZeemanEnergy(magnitude=500, angle=np.pi/2)], pattern="uniform", T=T, E_B=E_B_d, m_perp_factor=0)
    mm_di.params.UPDATE_SCHEME = hotspice.Scheme.NEEL
    mm_di.progress(t)

    E_D = mm_mono.get_energy('dimonopolar').E
    vertical = np.where(mm_mono.angles == 90 * np.pi/180)
    result = list(zip(vertical[0], vertical[1]))
    print("Vertical: {}".format(hotspice.utils.J_to_eV(np.max([E_D[row, col] for row, col in result]))))
    slope = np.where(mm_mono.angles == 30 * np.pi/180)
    result = list(zip(slope[0], slope[1]))
    print("slope: {}".format(hotspice.utils.J_to_eV(np.max([E_D[row, col] for row, col in result]))))

    E_D = mm_di.get_energy('dipolar').E
    vertical = np.where(mm_di.angles == 90 * np.pi/180)
    result = list(zip(vertical[0], vertical[1]))
    print("Vertical: {}".format(hotspice.utils.J_to_eV(np.max([E_D[row, col] for row, col in result]))))
    slope = np.where(mm_di.angles == 30 * np.pi/180)
    result = list(zip(slope[0], slope[1]))
    print("slope: {}".format(hotspice.utils.J_to_eV(np.max([E_D[row, col] for row, col in result]))))

    ZM = mm_mono.get_energy('zeeman')
    ZM.magnitude = 0
    ZM.angle = angle

    ZM = mm_di.get_energy('zeeman')
    ZM.magnitude = 0
    ZM.angle = angle

    M_y_mono = []
    M_y_di = []
    for k, B in enumerate(B_vals_mono):
        ZM = mm_mono.get_energy('zeeman')
        ZM.magnitude = B
        mm_mono.progress(t)
        M_y_mono.append(mm_mono.m_avg_y/M_S)

        ZM = mm_di.get_energy('zeeman')
        ZM.magnitude = B_vals_di[k]
        mm_di.progress(t)
        M_y_di.append(mm_di.m_avg_y/M_S)

    found_mono = False
    found_di = False

    i = 0
    while not found_mono or not found_di:
        if not found_mono:
            if (M_y_mono[i] < 0 < M_y_mono[i + 1]) or (M_y_mono[i] > 0 > M_y_mono[i + 1]):
                B_C_mono = np.abs(B_vals_mono[i] - (M_y_mono[i]*(B_vals_mono[i+1] - B_vals_mono[i]))/(M_y_mono[i+1] - M_y_mono[i]))
                found_mono = True
        if not found_di:
            if (M_y_di[i] < 0 < M_y_di[i + 1]) or (M_y_di[i] > 0 > M_y_di[i + 1]):
                B_C_di = np.abs(B_vals_di[i] - (M_y_di[i]*(B_vals_di[i+1] - B_vals_di[i]))/(M_y_di[i+1] - M_y_di[i]))
                found_di = True
        i += 1

    ZM = mm_mono.get_energy('zeeman')
    ZM.angle = angle
    ZM.magnitude = 0

    ZM = mm_di.get_energy('zeeman')
    ZM.angle = angle
    ZM.magnitude = 0
    
    ## Create plot
    thesis_utils.init_style()
    figsize = (thesis_utils.page_width, thesis_utils.page_width*0.43)
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    fig.subplots_adjust(top=0.9, bottom=0, left=0.05, right=0.98, wspace=0.15, hspace=0.12)
    fig.patch.set_alpha(0) # Transparent fig, but not axes cause I put transparent=False in save_results()

    i_vals = [79, 86, 92, 99]
    for i, B in enumerate(np.linspace(0, -1.06, 100)):
        ZM = mm_mono.get_energy('zeeman')
        ZM.magnitude = B * B_C_mono
        mm_mono.progress(t)

        ZM = mm_di.get_energy('zeeman')
        ZM.magnitude = B * B_C_di
        mm_di.progress(t)
        if i in i_vals:
            x = i_vals.index(i)
            print(B)
            unit = 'µ'
            unit_factor = SIprefix_to_mul(unit)
            s = 5

            ax1: plt.Axes = axes[1,x]
            ax1.set_aspect('equal')
            ax1.set_facecolor("gray")
            ax1.patch.set_alpha(1)
            ax1.set_xticks([])
            ax1.set_yticks([])
            for spine in ax1.spines.values(): spine.set_visible(False)
            if x == 0: ax1.set_ylabel("Dumbbell", fontsize=thesis_utils.fs_large)
            nonzero = mm_mono.m.nonzero()
            mx, my = asnumpy(np.multiply(mm_mono.m, mm_mono.orientation[:, :, 0])[nonzero]), asnumpy(np.multiply(mm_mono.m, mm_mono.orientation[:, :, 1])[nonzero])
            ax1.quiver(asnumpy(mm_mono.xx[nonzero]) / unit_factor, asnumpy(mm_mono.yy[nonzero]) / unit_factor, mx / unit_factor, my / unit_factor,
                color=np.where(mx - mm_old == 0, "black", "white"),
                pivot='mid', scale=1.1 / mm_mono._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7,
                units='xy')  # units='xy' makes arrows scale correctly when zooming
            initial, type_I, type_II, inside, wrong, final, red_x, red_y, blue_x, blue_y, _ = vertex(mm_mono, 1)
            ax1.scatter(red_x, red_y, color="red", s=s, zorder=10)
            ax1.scatter(blue_x, blue_y, color="blue", s=s, zorder=10)

            ax2: plt.Axes = axes[0,x]
            ax2.set_aspect('equal')
            ax2.set_facecolor("gray")
            ax2.patch.set_alpha(1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            for spine in ax2.spines.values(): spine.set_visible(False)
            if x == 0: ax2.set_ylabel("Dipole", fontsize=thesis_utils.fs_large)
            ax2.set_title(f"$B = {abs(B):.2f} B_c$", fontsize=thesis_utils.fs_large)
            nonzero = mm_di.m.nonzero()
            mx, my = asnumpy(np.multiply(mm_di.m, mm_di.orientation[:, :, 0])[nonzero]), asnumpy(np.multiply(mm_di.m, mm_di.orientation[:, :, 1])[nonzero])
            ax2.quiver(asnumpy(mm_di.xx[nonzero]) / unit_factor, asnumpy(mm_di.yy[nonzero]) / unit_factor, mx / unit_factor, my / unit_factor,
                color=np.where(mx - mm_old == 0, "black", "white"),
                pivot='mid', scale=1.1 / mm_di._get_closest_dist(), headlength=17, headaxislength=17, headwidth=7,
                units='xy')  # units='xy' makes arrows scale correctly when zooming
            initial, type_I, type_II, inside, wrong, final, red_x, red_y, blue_x, blue_y, _ = vertex(mm_di, 1)
            ax2.scatter(red_x, red_y, color="red", s=s, zorder=10)
            ax2.scatter(blue_x, blue_y, color="blue", s=s, zorder=10)
            
            if x != 0:
                x0 = axes[0,x-1].get_position().x1
                x1 = axes[0,x].get_position().x0
                y1 = (ax1.get_position().y0 + ax1.get_position().y1)/2
                y2 = (ax2.get_position().y0 + ax2.get_position().y1)/2
                arrow_kwargs = dict(arrowstyle='-|>', mutation_scale=14, linewidth=2, shrinkA=1, shrinkB=0, color='k')
                fig.patches.append(FancyArrowPatch((x0, y1), (x1, y1), transform=fig.transFigure, **arrow_kwargs))
                fig.patches.append(FancyArrowPatch((x0, y2), (x1, y2), transform=fig.transFigure, **arrow_kwargs))
    
    ## Save figure
    hotspice.utils.save_results(figures={f"Hysteresis_Kagome": fig}, timestamped=False, copy_script=False, transparent=False)

if __name__ == "__main__":
    plot()