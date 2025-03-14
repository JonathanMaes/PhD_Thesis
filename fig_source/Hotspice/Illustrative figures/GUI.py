import hotspice


mm = hotspice.ASI.IP_Pinwheel(230e-9, nx=80, ny=50, pattern="random", PBC=True,
                              E_B=hotspice.utils.eV_to_J(5), T=300)#, V=2e-21, energies=(hotspice.energies.DiMonopolarEnergy(d=200e-9, small_d=70e-9),))
mm.params.UPDATE_SCHEME = hotspice.Scheme.NEEL
mm.progress(MCsteps_max=20, t_max=2e-6)
hotspice.gui.show(mm, custom_step=lambda x: 0, custom_reset=lambda x: 0)
# Take a window screenshot after adjusting window width such that all panels are tightly packed
