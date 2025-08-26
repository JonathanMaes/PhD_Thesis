import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests
import string

from ast import literal_eval
from datetime import datetime, timedelta
from matplotlib.ticker import FuncFormatter

import hotspice
import thesis_utils

def plot(download: bool = True, show_yticks: bool = False, log_scale: bool = False, start_year: int = None):
    CSVfile = f"{os.path.splitext(__file__)[0]}.out/data.csv"
    os.makedirs(os.path.dirname(CSVfile), exist_ok=True)
    
    subjects = ["Neuromorphic computing", "Reservoir computing", "In-memory computing", "von Neumann bottleneck", "Dennard scaling", "Material computation"]
    def standard_str(s): return re.sub(r'\W+', '', s)
    if download:
        if start_year is None: start_year = 1980
        end_year = 2022
        url = f"https://books.google.com/ngrams/graph?content={','.join(subjects)}&year_start={start_year}&year_end={end_year}&corpus=en&smoothing=3&case_insensitive=false"
        req = requests.get(url)
        match = re.search(r'<script id="ngrams-data" type="application/json">(.*?)</script>', req.text)
        if match:
            le = literal_eval(match.group(1))
            case_insensitive = any(["(All)" in qry['ngram'] for qry in le])
            data = {}
            for qry in le:
                if case_insensitive and "(All)" not in qry['ngram']: continue
                matches = [standard_str(qry['ngram'].strip("(All)")) == standard_str(subject) for subject in subjects]
                key = subjects[matches.index(True)] if any(matches) else qry['ngram']
                data[key] = qry['timeseries']
            df = pd.DataFrame(data, index=range(start_year, end_year+1))
            df.index.name = 'year'
            df.to_csv(CSVfile)
        else:
            raise RuntimeError("Failed to fetch Ngram data")

    # load
    df = pd.read_csv(CSVfile)
    years = [datetime(year, 1, 1) for year in df["year"].values]
    ngrams = df.columns.tolist()
    y_max = np.max([df[col].values for col in ngrams if col != "year"])
    data_vals = [100*df[col].values/y_max for col in ngrams]  # percent

    # style
    thesis_utils.init_style()
    fig, ax = plt.subplots(figsize=(thesis_utils.page_width*0.55, thesis_utils.page_width*0.4))

    # plot & inline labels
    num = len(ngrams)
    for idx, (label, series) in enumerate(zip(ngrams, data_vals)):
        if label == "year": continue
        line, = ax.plot(years, series, color=f"C{idx}")
        # label at end
        ax.text(
            years[-1] + timedelta(days=365),    # one year to the right
            series[-1],                         # at last point
            label,
            color=f"C{idx}",
            va='center',
            fontsize=thesis_utils.fs_small
        )
    
    # axes styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', direction='out', labelsize=thesis_utils.fs_small)

    # ticks
    if show_yticks: ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{int(y)}%"))
    else: ax.set_yticks([])
    ax.set_ylabel("Popularity in literature" + " [a.u.]"*show_yticks, fontsize=thesis_utils.fs_small)
    if log_scale: ax.set_yscale('log')
    else: ax.set_ylim([0, ax.get_ylim()[1]])
    ax.set_xlim(years[0] if start_year is None else datetime(start_year, 1, 1), years[-1] + timedelta(days=5*365)) # add margin for labels
    ax.xaxis.set_major_locator(mdates.YearLocator(int((years[-1].year - start_year)//2.5)))

    plt.tight_layout(pad=1)
    hotspice.utils.save_results(copy_script=False, figures={"Ngrams": fig}, timestamped=False)

if __name__ == "__main__":
    plot(download=True, show_yticks=False, start_year=2000)
