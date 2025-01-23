import hotspice
import matplotlib.pyplot as plt
import numpy as np

xp = hotspice.xp

import thesis_utils


E_B = 1
    
def method1(delta_E):
    E = -delta_E/2
    return np.maximum(delta_E, E_B - E)

def method2(delta_E):
    E = -delta_E/2
    E_highest_state = np.abs(E)
    return np.where(E_B > E_highest_state, E_B - E, delta_E)

col_1 = "C0"
col_2 = "C1"
col_exact = "grey"

thetas = np.linspace(0, 180, 181) # [deg]
def E_landscape(delta_E=0, thetas=thetas): #! `thetas` in degrees! (bad design choice)
    return E_landscape_base(delta_E, thetas) + (1 - np.cos(2*np.deg2rad(thetas)))*E_B/2
def E_landscape_base(delta_E=0, thetas=thetas): #! `thetas` in degrees! (bad design choice)
    return (1 - np.cos(np.deg2rad(thetas)))*delta_E/2

def plot_EB_meanbarrier(N: int = 5): # N: number of bottom plots
    ## Create plot
    thesis_utils.init_style()
    figsize = (thesis_utils.page_width, 5)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, N, height_ratios=[3, 1])
    
    ## Main graph showing E_B_eff(Delta E)
    ax1 = fig.add_subplot(gs[0,:])
    delta_E_range = np.linspace(-E_B*4, E_B*4, 201)
    
    E_barrier_method1 = method1(delta_E_range)
    E_barrier_method2 = method2(delta_E_range)

    ax1.plot(delta_E_range, E_barrier_method1, color=col_1, label=r"Method 1: $\mathrm{max}(E_\perp, E_2) - E_1$")
    ax1.plot(delta_E_range, E_barrier_method2, color=col_2, label=r"Method 2: $E_\perp$ if $E_\perp > \mathrm{max}(E_1, E_2)$ else $E_2$")
    exact = [E_landscape(delta_E)[np.argmax(E_landscape(delta_E))] for delta_E in delta_E_range]
    # ax1.scatter(delta_E_range, exact, marker='*', color=col_exact, label=r"Real $E_\mathrm{barrier}$ for ideal sines")
    ax1.plot(delta_E_range, exact, color=col_exact, label=r"Exact $\tilde{E_\mathrm{B}}$")
    # ax1.plot(delta_E_range, E_B*(delta_E_range/E_B/4+1)**2, color=col_exact, label=r"Exact $\widetilde{E_\mathrm{B}}$")
    ax1.set_xlabel(r"$\Delta E$ between state $1 \rightarrow 2$")
    ax1.set_ylabel(r'$E_\mathrm{barrier}$')
    ax1.set_xlim([-4*E_B, 4*E_B])
    ax1.set_ylim([-4*E_B, 4*E_B])
    # ax1.set_xticks(ticks)
    # ax1.set_yticks(ticks)
    ax1.legend()
    
    ## Subplots
    for i in range(N):
        ax = fig.add_subplot(gs[1,i])
        start, end = -4*E_B, 4*E_B
        delta_E = start + (end - start)*i/(N-1)
        plot_subplot(ax, delta_E = delta_E)

    ## Adjust layout to avoid overlapping elements
    fig.tight_layout()
    # fig.subplots_adjust()

    ## Save
    hotspice.utils.save_results(figures={"EB_meanbarrier": fig}, timestamped=False)


def plot_subplot(ax: plt.Axes, delta_E: float = 0):
    ## Prepare axis
    # ax.set_axis_off()
    ax.set_frame_on(False)
    ax.tick_params(bottom=True, labelbottom=True, left=False, labelleft=False)
    ax.set_title(r"$\Delta E = " + f"{delta_E:.2g}$")    
    # ax.set_xlabel("Magnetisation angle [°]")
    margin = 0.1
    xmin, xmax = 0, 180
    ymin, ymax = -2*E_B + delta_E/2, 2*E_B + delta_E/2
    xmin, xmax = xmin - (xmax - xmin)*margin, xmax + (xmax - xmin)*margin
    ymin, ymax = ymin - (ymax - ymin)*margin, ymax + (ymax - ymin)*margin
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.tick_params(bottom=False, labelbottom=False)
    # ax.set_xticks([0, 180])
    # ax.set_xticklabels(["State\n1", "State\n2"])

    ## Basic landscape (black)
    ax.plot(thetas, E_landscape_base(delta_E=delta_E), color='black', lw=1, linestyle='--')
    
    ## Landscape with barrier
    landscape = E_landscape(delta_E)
    ax.plot(thetas, landscape, color=col_exact)
    
    ## Points of interest
    # Values at 0, 90 and 180°
    dots_x = np.array([0, 90, 180])
    ax.scatter(dots_x, E_landscape(delta_E, thetas=dots_x), color='black')
    
    # Exact top
    topindex = np.argmax(landscape)
    ax.scatter([thetas[topindex]], [landscape[topindex]], marker='*', label="Maximum", color=col_exact)
    
    ## Lines
    d = -4
    # Vertical & horizontal bars: exact top
    x = d/3
    ax.plot([x, x], [0, landscape[topindex]], color=col_exact)[0]
    ax.plot([x, 180], [landscape[topindex], landscape[topindex]], color='grey', lw=1, linestyle=':')[0]
    # Vertical & horizontal bars: method 1
    x += d
    ax.plot([x, x], [0, method1(delta_E)], color=col_1)[0]
    ax.plot([x, 180], [method1(delta_E), method1(delta_E)], color=col_1, lw=1, linestyle=':')[0]
    # Vertical & horizontal bars: method 2
    x += d
    ax.plot([x, x], [0, method2(delta_E)], color=col_2)[0]
    ax.plot([x, 180], [method2(delta_E), method2(delta_E)], color=col_2, lw=1, linestyle=':')[0]


if __name__ == "__main__":
    plot_EB_meanbarrier()
