import os
import pandas as pd
import numpy as np
import subprocess
import tempfile
import shutil
from pathlib import Path
from platypus import NSGAII, Problem, Real, ProcessPoolEvaluator

from lstm import camels

import jax.numpy as jnp
import jax.random as jrn

class VIC:
    """VIC hydrologic model wrapper."""

    def __init__(
        self, soilfile, gauge, startdate, enddate, datadir="data", vic_exec="vicNl"
    ):
        """Initialize VIC model for a specific basin."""
        self.bid = gauge.gauge_id
        self.startdate = pd.to_datetime(startdate)
        self.enddate = pd.to_datetime(enddate)
        with open(soilfile) as fin:
            lines = fin.readlines()
        lats = np.array([float(line.split()[2]) for line in lines])
        lons = np.array([float(line.split()[3]) for line in lines])
        i = np.argmin(
            np.sqrt((lats - gauge.gauge_lat) ** 2 + (lons - gauge.gauge_lon) ** 2)
        )
        self.soil = lines[i]
        self.lat = float(lats[i])
        self.lon = float(lons[i])
        self.area = float(gauge.area_gages2)
        self.datadir = datadir
        self.vic_exec = vic_exec
        self.forcingpath = f"{datadir}/forcings"

    def write_soil(self, outfile, line):
        """Write soil parameter file."""
        with open(outfile, "w") as fout:
            fout.write(line)

    def write_global(self, outdir):
        """Write global control file for VIC."""
        datadir = self.datadir

        with open(f"{datadir}/vic/global.template") as fin:
            lines = fin.readlines()

        with open(f"{outdir}/global.txt", "w") as fout:
            for line in lines:
                line = line.strip()

                if not line:
                    fout.write("\n")
                    continue

                parts = line.split(None, 1)
                if len(parts) < 2:
                    fout.write(line + "\n")
                    continue

                param_name, param_value = parts[0], parts[1]

                if param_name == "FORCING1":
                    fout.write(f"FORCING1        {self.forcingpath}/data_\n")
                elif param_name == "SOIL":
                    fout.write(f"SOIL            {outdir}/soil.txt\n")
                elif param_name == "RESULT_DIR":
                    fout.write(f"RESULT_DIR      {outdir}\n")
                elif param_name == "STARTYEAR":
                    fout.write(f"STARTYEAR\t{self.startdate.year}\t# year model simulation starts\n")
                elif param_name == "STARTMONTH":
                    fout.write(f"STARTMONTH\t{self.startdate.month:02d}\t# month model simulation starts\n")
                elif param_name == "STARTDAY":
                    fout.write(f"STARTDAY\t{self.startdate.day:02d}\t# day model simulation starts\n")
                elif param_name == "ENDYEAR":
                    fout.write(f"ENDYEAR\t{self.enddate.year}\t# year model simulation ends\n")
                elif param_name == "ENDMONTH":
                    fout.write(f"ENDMONTH\t{self.enddate.month:02d}\t# month model simulation ends\n")
                elif param_name == "ENDDAY":
                    fout.write(f"ENDDAY\t{self.enddate.day:02d}\t# day model simulation ends\n")
                elif param_name == "FORCEYEAR":
                    fout.write(f"FORCEYEAR\t{self.startdate.year}\t# Year of first forcing record\n")
                elif param_name == "FORCEMONTH":
                    fout.write(f"FORCEMONTH\t{self.startdate.month:02d}\t# Month of first forcing record\n")
                elif param_name == "FORCEDAY":
                    fout.write(f"FORCEDAY\t{self.startdate.day:02d}\t# Day of first forcing record\n")
                elif param_name == "VEGLIB":
                    fout.write(f"VEGLIB          {datadir}/vic/LDAS_veg_lib\n")
                elif param_name == "VEGPARAM":
                    fout.write(f"VEGPARAM        {datadir}/vic/vic.nldas.mexico.veg.txt\n")
                elif param_name == "SNOW_BAND":
                    snow_parts = param_value.split()
                    if len(snow_parts) > 1:
                        snow_path = snow_parts[1]
                        if snow_path.startswith("data/"):
                            new_snow_path = f"{datadir}/vic/vic.nldas.mexico.snow.txt.L13"
                            fout.write(f"SNOW_BAND       {snow_parts[0]} {new_snow_path}\n")
                        else:
                            fout.write(line + "\n")
                    else:
                        fout.write(line + "\n")
                elif param_name in ("LAKES",):
                    if param_value.startswith("data/"):
                        fout.write(f"LAKES           {datadir}/vic/lake_param.txt\n")
                    else:
                        fout.write(line + "\n")
                else:
                    # Keep other lines as-is
                    fout.write(line + "\n")

    def forcings(self, forcing, perturbation=None, seed=None):
        """Write forcing file from CAMELS data."""
        metfile = f"{self.datadir}/{forcing}/{self.bid}_lump_forcing_leap.txt"
        met = camels.read_met(metfile).loc[self.startdate : self.enddate, :]

        if perturbation is not None:
            data_cols = ["Prcp", "Tmax", "Tmin", "Srad", "Vp"]
            x = jnp.asarray(met.loc[:, data_cols].values)
            x = x[None, :, :] # add dummy batch dimension
            key = jrn.PRNGKey(0) if seed is None else jrn.PRNGKey(seed)
            x, _ = perturbation(x, key=key)
            met.loc[:, data_cols] = np.asarray(x[0])

        with tempfile.TemporaryDirectory(dir=f"{self.datadir}/forcings", delete=False) as outdir:
            self.forcingpath = outdir
            with open(f"{outdir}/data_{self.lat:.5f}_{self.lon:.5f}", "w") as fout:
                for i, row in met.iterrows():
                    prcp = row["Prcp"]
                    fout.write(
                        "{0:f} {1:.2f} {2:.2f} 5.00\n".format(
                            prcp, row["Tmax"], row["Tmin"]
                        )
                    )

    def params(self, x):
        """Apply calibration parameters to soil file line."""
        data = self.soil.split()
        data[4] = "{0:.4f}".format(x[0])  # b
        data[5] = "{0:.4f}".format(x[1])  # Ds
        data[6] = "{0:.4f}".format(x[2])  # Dsmax
        data[7] = "{0:.4f}".format(x[3])  # Ws
        data[23] = "{0:.4f}".format(x[4])  # dz2
        data[24] = "{0:.4f}".format(x[5])  # dz3
        for lyr in range(3):
            data[9 + lyr] = "{0:.3f}".format(x[6] * float(data[9 + lyr]))  # expt
            data[12 + lyr] = "{0:.1f}".format(x[7] * float(data[12 + lyr]))  # Ksat
            data[33 + lyr] = "{0:.0f}".format(x[8])  # bulk density
        return " ".join(data) + "\n"

    def run(self, params=[]):
        """Run VIC model and return streamflow in m/day."""
        with tempfile.TemporaryDirectory(dir="./", delete=True) as outdir:
            if len(params) > 0:
                soil = self.params(params)
            else:
                soil = self.soil
            self.write_soil("{0}/soil.txt".format(outdir), soil)
            self.write_global(outdir)
            _ = subprocess.run(
                [self.vic_exec, "-g", "{0}/global.txt".format(outdir)],
                capture_output=True,
                text=True,
            )
            try:
                out = pd.read_csv(
                    "{0}/flux_snow_{1:.5f}_{2:.5f}".format(outdir, self.lat, self.lon),
                    sep="\\s+",
                    header=None,
                    names=[
                        "year",
                        "month",
                        "day",
                        "prec",
                        "evap",
                        "runoff",
                        "baseflow",
                        "sm1",
                        "sm2",
                        "sm3",
                        "swe",
                        "sensible",
                        "latent",
                    ],
                )
                out.index = pd.to_datetime(out.loc[:, ["year", "month", "day"]])
            except FileNotFoundError:
                dt = pd.date_range(self.startdate, self.enddate)
                out = pd.DataFrame(
                    dict(
                        runoff=pd.Series(np.zeros(len(dt)), dt),
                        baseflow=pd.Series(np.zeros(len(dt)), dt),
                    )
                )
        return (out.runoff + out.baseflow) / 1000


def evaluate(bids, soilfile, forcing, startdate, enddate, datadir="data", vic_exec="vicNl", perturbation=None, seed=None):
    """Run VIC for multiple basins and return simulated vs observed streamflow."""
    basins = pd.read_csv(f"{datadir}/camels_topo.txt", sep=";", dtype={"gauge_id": str})
    mod = {}
    obs = {}
    for bid in bids:
        gauge = basins.query("gauge_id == @bid").T.iloc[:, 0]
        model = VIC(soilfile, gauge, startdate, enddate, datadir=datadir, vic_exec=vic_exec)
        model.forcings(forcing, perturbation=perturbation, seed=seed)
        qfile = f"{datadir}/usgs/{model.bid}_streamflow_qc.txt"
        qobs = (
            camels.read_q(qfile).loc[model.startdate : model.enddate, "Flow"]
            * 0.0283168
            * 86400
            / (model.area * 1e6)
        )
        q = model.run()
        mod[bid] = q
        obs[bid] = qobs
    return pd.DataFrame(mod), pd.DataFrame(obs)


class VICObjective:
    """Objective function for VIC calibration (minimizes negative NSE)."""

    def __init__(self, model, obs):
        self.model = model
        self.obs = obs

    def __call__(self, vars):
        """Evaluate NSE for given parameter set."""
        q = self.model.run(vars)
        nse = 1 - np.sum((self.obs - q) ** 2) / np.sum(
            (self.obs - np.mean(self.obs)) ** 2
        )
        return [-nse]  # Minimize negative NSE


def calibrate(
    bid,
    soilfile,
    forcing,
    startdate,
    enddate,
    datadir="data",
    nprocs=None,
    vic_exec="vicNl",
    perturbation=None,
    seed=None,
):
    """Calibrate VIC model using NSGA-II optimization."""
    if nprocs is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        if slurm_cpus is not None:
            nprocs = int(slurm_cpus)
        else:
            nprocs = os.cpu_count()
    basins = pd.read_csv(f"{datadir}/camels_topo.txt", sep=";", dtype={"gauge_id": str})
    gauge = basins.query("gauge_id == @bid").T.iloc[:, 0]
    model = VIC(soilfile, gauge, startdate, enddate, datadir, vic_exec=vic_exec)
    model.forcings(forcing, perturbation=perturbation, seed=seed)
    qfile = f"{datadir}/usgs/{model.bid}_streamflow_qc.txt"
    obs = (
        camels.read_q(qfile).loc[model.startdate : model.enddate, "Flow"]
        * 0.0283168
        * 86400
        / (model.area * 1e6)
    )
    problem = Problem(9, 1)  # 9 parameters, 1 objective
    problem.types[:] = [
        Real(0.001, 0.4),  # b
        Real(0.001, 1.0),  # Ds
        Real(0.1, 30.0),  # Dsmax
        Real(0.2, 1.0),  # Ws
        Real(0.1, 4.0),  # dz2
        Real(0.1, 10.0),  # dz3
        Real(0.5, 2.0),  # Exp
        Real(0.1, 10.0),  # Ksat
        Real(1350, 1650),  # bd
    ]
    # wrap model.run to include datadir
    problem.function = VICObjective(model, obs)

    with ProcessPoolEvaluator(nprocs) as evaluator:
        algorithm = NSGAII(problem, evaluator=evaluator)
        algorithm.run(1000)

    # delete temporary forcing directory
    shutil.rmtree(model.forcingpath)

    best = min(algorithm.result, key=lambda x: x.objectives[0])
    return model.params(best.variables)
