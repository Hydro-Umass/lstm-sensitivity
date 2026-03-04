import pandas as pd
import numpy as np
import subprocess
import tempfile
from pathlib import Path
from platypus import NSGAII, Problem, Real, ProcessPoolEvaluator

from lstm import camels

class VIC():

    def __init__(self, soilfile, gauge, startdate, enddate, datadir="data"):
        self.bid = gauge.gauge_id
        self.startdate = pd.to_datetime(startdate)
        self.enddate = pd.to_datetime(enddate)
        with open(soilfile) as fin:
            lines = fin.readlines()
        lats = np.array([float(line.split()[2]) for line in lines])
        lons = np.array([float(line.split()[3]) for line in lines])
        i = np.argmin(np.sqrt((lats - gauge.gauge_lat)**2 + (lons - gauge.gauge_lon)**2))
        self.soil = lines[i]
        self.lat = float(lats[i])
        self.lon = float(lons[i])
        self.area = float(gauge.area_gages2)
        self.datadir = datadir

    def write_soil(self, outfile, line):
        with open(outfile, 'w') as fout:
            fout.write(line)

    def write_global(self, outdir):
        """Write global control file for VIC."""
        with open(f"{self.datadir}/vic/global.template") as fin:
            lines = fin.readlines()
        with open("{0}/global.txt".format(outdir), 'w') as fout:
            for line in lines:
                if line.find("SOIL") == 0:
                    fout.write(line.replace("soil.txt", "{0}/soil.txt".format(outdir)))
                elif line.find("RESULT_DIR") == 0:
                    fout.write(line.replace("./output", "{0}".format(outdir)))
                elif line.find("STARTYEAR") == 0:
                    fout.write(line.replace("1980", str(self.startdate.year)))
                elif line.find("STARTMONTH") == 0:
                    fout.write(line.replace("01", str(self.startdate.month)))
                elif line.find("STARTDAY") == 0:
                    fout.write(line.replace("01", str(self.startdate.day)))
                elif line.find("ENDYEAR") == 0:
                    fout.write(line.replace("2014", str(self.enddate.year)))
                elif line.find("ENDMONTH") == 0:
                    fout.write(line.replace("12", str(self.enddate.month)))
                elif line.find("ENDDAY") == 0:
                    fout.write(line.replace("31", str(self.enddate.day)))
                elif line.find("FORCEYEAR") == 0:
                    fout.write(line.replace("1985", str(self.startdate.year)))
                elif line.find("FORCEMONTH") == 0:
                    fout.write(line.replace("10", str(self.startdate.month)))
                elif line.find("FORCEDAY") == 0:
                    fout.write(line.replace("01", str(self.startdate.day)))
                else:
                    fout.write(line)

    def forcings(self, forcing):
        """Write forcing file from CAMELS data."""
        metfile = f"{self.datadir}/{forcing}/{self.bid}_lump_forcing_leap.txt"
        met = camels.read_met(metfile).loc[self.startdate:self.enddate, :]
        outdir = Path(f"{self.datadir}/forcings")
        outdir.mkdir(exist_ok=True)
        with open(f"{outdir}/data_{self.lat:.5f}_{self.lon:.5f}", 'w') as fout:
            for i, row in met.iterrows():
                fout.write("{0:f} {1:.2f} {2:.2f} 5.00\n".format(row['Prcp'], row['Tmax'], row['Tmin']))

    def params(self, x):
        data = self.soil.split()
        data[4] = "{0:.4f}".format(x[0]) # b
        data[5] = "{0:.4f}".format(x[1]) # Ds
        data[6] = "{0:.4f}".format(x[2]) # Dsmax
        data[7] = "{0:.4f}".format(x[3]) # Ws
        data[23] = "{0:.4f}".format(x[4]) # dz2
        data[24] = "{0:.4f}".format(x[5]) # dz3
        for lyr in range(3):
            data[9+lyr] = "{0:.3f}".format(x[6] * float(data[9+lyr])) # expt
            data[12+lyr] = "{0:.1f}".format(x[7] * float(data[12+lyr])) # Ksat
            data[33+lyr] = "{0:.0f}".format(x[8]) # bulk density
        return " ".join(data) + "\n"

    def run(self, params=[]):
        """Run the VIC model and return the output."""
        with tempfile.TemporaryDirectory(dir="./", delete=True) as outdir:
            if len(params) > 0:
                soil = self.params(params)
            else:
                soil = self.soil
            self.write_soil("{0}/soil.txt".format(outdir), soil)
            self.write_global(outdir)
            _ = subprocess.run(["vicNl", "-g", "{0}/global.txt".format(outdir)], capture_output=True, text=True)
            try:
                out = pd.read_csv("{0}/flux_snow_{1:.5f}_{2:.5f}".format(outdir, self.lat, self.lon), sep='\\s+',
                                  header=None, names=['year', 'month', 'day', 'prec', 'evap', 'runoff', 'baseflow', 'sm1', 'sm2', 'sm3', 'swe', 'sensible', 'latent'])
                out.index = pd.to_datetime(out.loc[:, ['year', 'month', 'day']])
            except FileNotFoundError:
                dt = pd.date_range(self.startdate, self.enddate)
                out = pd.DataFrame(dict(runoff=pd.Series(np.zeros(len(dt)), dt), baseflow=pd.Series(np.zeros(len(dt)), dt)))
        return (out.runoff + out.baseflow) / 1000

def evaluate(bids, soilfile, forcing, startdate, enddate, datadir="data"):
    basins = pd.read_csv(f"{datadir}/camels_topo.txt", sep=';', dtype={'gauge_id': str})
    mod = {}
    obs = {}
    for bid in bids:
        gauge = basins.query("gauge_id == @bid").T.iloc[:, 0]
        model = VIC(soilfile, gauge, startdate, enddate)
        model.forcings(forcing)
        qfile =f"{datadir}/usgs/{model.bid}_streamflow_qc.txt"
        qobs = camels.read_q(qfile).loc[model.startdate:model.enddate, 'Flow'] * 0.0283168 * 86400 / (model.area * 1e6)
        q = model.run()
        mod[bid] = q
        obs[bid] = qobs
    return pd.DataFrame(mod), pd.DataFrame(obs)

class VICObjective:

    def __init__(self, model, obs):
        self.model = model
        self.obs = obs

    def __call__(self, vars):
        q = self.model.run(vars)
        nse = 1 - np.sum((self.obs - q)**2) / np.sum((self.obs - np.mean(self.obs))**2)
        return [-nse]  # Minimize negative NSE

def calibrate(bid, soilfile, forcing, startdate, enddate, datadir="data", nprocs=32):
    """Calibrate VIC model using NSGA-II optimization."""
    basins = pd.read_csv(f"{datadir}/camels_topo.txt", sep=';', dtype={'gauge_id': str})
    gauge = basins.query("gauge_id == @bid").T.iloc[:, 0]
    model = VIC(soilfile, gauge, startdate, enddate, datadir)
    model.forcings(forcing)
    qfile = f"{datadir}/usgs/{model.bid}_streamflow_qc.txt"
    obs = camels.read_q(qfile).loc[model.startdate:model.enddate, 'Flow'] * 0.0283168 * 86400 / (model.area * 1e6)

    problem = Problem(9, 1)  # 9 parameters, 1 objective
    problem.types[:] = [
        Real(0.001, 0.4),  # b
        Real(0.001, 1.0),  # Ds
        Real(0.1, 30.0),    # Dsmax
        Real(0.2, 1.0),     # Ws
        Real(0.1, 4.0),     # dz2
        Real(0.1, 10.0),    # dz3
        Real(0.5, 2.0),     # Exp
        Real(0.1, 10.0),    # Ksat
        Real(1350, 1650)    # bd
    ]
    problem.function = VICObjective(model, obs)

    with ProcessPoolEvaluator(nprocs) as evaluator:
        algorithm = NSGAII(problem, evaluator=evaluator)
        algorithm.run(1000)

    best = min(algorithm.result, key=lambda x: x.objectives[0])
    return model.params(best.variables)
