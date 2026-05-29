import pandas as pd
import numpy as np

def nse(o, m):
    return 1 - ((o -m)**2).sum(axis=0) / ((o - o.mean())**2).sum(axis=0)

def kge(o, m):
    r = o.corrwith(m)
    a = m.std() / o.std()
    b = m.mean() / o.mean()
    return 1 - np.sqrt((r - 1)**2 + (a - 1)**2 + (b - 1)**2)

def read_met(metfile):
    header = ["Year", "Month", "Day", "Hr", "Dayl", "Prcp", "Srad", "Swe", "Tmax", "Tmin", "Vp",]
    m = pd.read_csv(metfile, sep="\\s+", skiprows=4, names=header)
    m["Date"] = pd.to_datetime(dict(year=m.Year, month=m.Month, day=m.Day))
    m["Gauge"] = metfile.split("/")[-1].split("_")[0]
    m = m.set_index("Date")
    return m
