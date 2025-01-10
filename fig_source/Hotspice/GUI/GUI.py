import hotspice

mm = hotspice.ASI.IP_Pinwheel(230e-9, nx=80, ny=50, E_B=0, pattern="random")
mm.progress(MCsteps_max=12)
hotspice.gui.show(mm, custom_step=lambda x: 0, custom_reset=lambda x: 0)
# Take a window screenshot after adjusting window width such that all panels are tightly packed