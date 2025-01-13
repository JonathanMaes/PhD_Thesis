import time

import os
os.environ["HOTSPICE_USE_GPU"] = "true"
import hotspice


def test_performance(N):
    t = time.time_ns()
    ASI = hotspice.ASI.OOP_Square(a=230e-9, n=N, T=300, pattern="random")
    t_init = time.time_ns() - t
    
    t = time.time_ns()
    for _ in range(5000): ASI._update_Néel()
    t_5000 = time.time_ns() - t
    
    return t_init/1e9, t_5000/1e9, hotspice.config.USE_GPU

def test_performance_range():
    for N in [50, 100, 150, 200, 400, 1000]:
        t_init = t_5000 = 0
        repetitions = 5
        for _ in range(repetitions):
            perf = test_performance(N)
            t_init += perf[0]
            t_5000 += perf[1]
            if t_init > 600 or t_5000 > 600: return
        t_init /= repetitions
        t_5000 /= repetitions
        print(f"{N}x{N} {'G' if perf[2] else 'C'}PU: init = {t_init:.3f}s, 5000 switches = {t_5000:.3f}s")


if __name__ == "__main__":
    test_performance_range()
