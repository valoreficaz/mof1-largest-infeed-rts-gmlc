# Curtailed for security: code and hourly results (RTS-GMLC)

Code and hourly results for the article

> N. Moreno-Alfonso, *Curtailed for security: Inertia rules when the largest loss is a renewable plant*, The Electricity Journal (submitted).

The model is a full-year, day-ahead unit commitment of the public [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) test system in which the loss of a whole wind plant is a credible event. Each case adds a frequency-security rule (none, a minimum number of large units, a cap on non-synchronous penetration, a fixed kinetic-energy floor, or a limit on the initial rate of change of frequency, RoCoF, indexed to the largest infeed of each hour) and records, hour by hour, the schedule, the synchronous kinetic energy and the initial RoCoF that the worst credible loss of that hour would cause.

Code comments, console messages and some file names are in Spanish.

## Contents

| Path | What it is |
|---|---|
| `uc_mof1_red.py` | Unit commitment with DC network, hydro reservoirs and the largest infeed of each hour as dimensioning event; one case per call, written to `resultados_red/<case>.csv` |
| `uc_mof1.py` | Data loading, rule definitions and kinetic energy of each unit, used by `uc_mof1_red.py` |
| `ramas_semilla.txt` | Branches monitored from the first day |
| `run_cases.sh` | Re-runs every published case with the exact options used |
| `setup.sh` | Fetches the third-party data at the commits used |
| `results/` | Hourly results of the 27 cases used in the article (`<case>.csv.gz`) and `resumen_v5.csv`, the annual summary of the cases used for the numerical resolution |
| `article/` | `cifras.py` computes every number of the article from `results/` and writes `cifras.tex`; `figuras.py` draws Figures 1 to 3 |
| `inertia_data/` | Generator parameters of the published dynamic data set and kinetic energy per unit, as published and as intended by its generating script |

## Case names

`<rule>_<options>`: `base` no rule; `muon8` at least eight units of 155 MW or more; `snsp75` non-synchronous penetration cap of 75 %; `ekin13781` and `ekin11661` fixed floors calibrated on the largest synchronous unit with and without the nuclear unit; `ekin25410` fixed floor calibrated on the largest wind plant; `rocof1` RoCoF limit of 1 Hz/s indexed to the largest infeed of each hour; `sinc2485` and `ekin12584` matched levels used for the numerical resolution. Options: `_dyn` intended inertia data, `_dynsabot` inertia as published, `_mp` largest infeed of each hour as event, `_sinnuclear` without the nuclear unit, `_nuclibre` nuclear unit free to be decommitted, `_gap0.1` solved to a 0.1 % gap, `_hmin` hydro minimum output of 15 MW.

## Requirements

Linux with bash and git; Python 3.12 with the versions in `requirements.txt` (SciPy 1.11.4, whose `scipy.optimize.milp` wraps HiGHS, NumPy 1.26.4, pandas 2.1.4, and matplotlib for the figures). About 560 MB for the third-party data.

## Usage

```bash
git clone https://github.com/valoreficaz/mof1-largest-infeed-rts-gmlc
cd mof1-largest-infeed-rts-gmlc
bash setup.sh --venv              # third-party data at the pinned commits and a virtual environment
export PY=$PWD/.venv/bin/python
```

Numbers and figures of the article from the published results, in seconds:

```bash
cd article
MOF1_RESULTADOS=../results $PY cifras.py     # writes cifras.tex, identical to the one in the repository
MOF1_RESULTADOS=../results $PY figuras.py    # writes figs/fig1_reglas.pdf, fig2_reparto.pdf, fig3_datos.pdf
```

Re-running the unit commitment, from about one hour to several hours per case:

```bash
PY=$PY bash run_cases.sh                  # all cases, two at a time
PY=$PY bash run_cases.sh rocof1_dyn_mp    # one case
```

`setup.sh` clones [GridMod/RTS-GMLC](https://github.com/GridMod/RTS-GMLC) at commit `3ece0d3` and [FredericSabot/PDSA-RTS-GMLC](https://github.com/FredericSabot/PDSA-RTS-GMLC) at commit `9edac7e`; these data sets are not redistributed here.

## Numerical reproducibility

Each daily mixed-integer problem is solved to a relative gap of 0.5 % or 120 s. On another machine, or with another build of HiGHS, the solver can return a different solution inside the gap, so re-runs are not expected to be identical bit for bit. The cases with `_gap0.1` re-solve the main rules with a 0.1 % gap and 300 s to bound the effect of the tolerance; the article reports it.

## Inertia data

The kinetic energies used are those intended by the script that generates the dynamic data set of PDSA-RTS-GMLC. In the published files the units of 350, 355 and 400 MW carry the parameters intended for the 155 MW units; the discrepancy has been reported to the maintainers (https://github.com/FredericSabot/PDSA-RTS-GMLC/issues/1). `inertia_data/kinetic_energy_published_vs_intended.csv` gives both values for every unit.

## Licence

Code: MIT (`LICENSE`). Results and derived data (`results/`, `article/cifras.tex`, `inertia_data/`): Creative Commons Attribution 4.0 International (`LICENSE-data`). RTS-GMLC and PDSA-RTS-GMLC keep their own licences.

## Contact

Narciso Moreno-Alfonso, Departamento de Ingeniería Eléctrica, Escuela Politécnica Superior, Universidad de Sevilla, narciso-ma@us.es
