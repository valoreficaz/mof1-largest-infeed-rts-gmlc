#!/usr/bin/env python3
"""Cifras del articulo «Curtailed for security» (The Electricity Journal).

Calcula, a partir de los ficheros horarios de results/ y de los datos del banco, todas las cifras que
cita el articulo, y escribe:
  cifras.tex   una macro \\V<nombre> por cifra, ya formateada;
  cifras.json  el valor sin redondear de cada macro, su texto y de donde sale.

Uso, desde la carpeta article/:   MOF1_RESULTADOS=../results python3 cifras.py
Lee:
  results/<caso>.csv.gz                   resultados horarios de los casos del articulo
  results/resumen_v5.csv                  resumen anual de los casos de la resolucion numerica
  inertia_data/tabla_RTS_par_publicado.csv  constantes de inercia publicadas por grupo
  RTS-GMLC/RTS_Data/SourceData/{gen,bus,branch}.csv
  uc_mof1_red.py                          constantes del modelo (F0, TH, GAP, TLIM)
"""
import json
import math
import os
import re

import numpy as np
import pandas as pd

AQUI = os.path.dirname(os.path.abspath(__file__))
# Rutas configurables con variables de entorno (por omision, las de la carpeta MOF1 o de la raiz del
# repositorio publicado): MOF1_RAIZ (carpeta con uc_mof1_red.py), MOF1_RESULTADOS (ficheros horarios
# <caso>.csv o <caso>.csv.gz), MOF1_RTS_GMLC (copia de RTS-GMLC) y MOF1_SALIDA (salidas).
RAIZ = os.environ.get('MOF1_RAIZ', os.path.dirname(AQUI))
RR = os.environ.get('MOF1_RESULTADOS', os.path.join(RAIZ, 'resultados_red'))
RTS = os.environ.get('MOF1_RTS_GMLC', os.path.join(RAIZ, 'RTS-GMLC'))
SALIDA = os.environ.get('MOF1_SALIDA', AQUI)


def ruta_csv(p):
    """p si existe; si no, p + '.gz' cuando el fichero esta comprimido."""
    return p + '.gz' if not os.path.exists(p) and os.path.exists(p + '.gz') else p


SRC = os.path.join(RTS, 'RTS_Data', 'SourceData')
TOL = 1e-6                      # tolerancia de la tasa (Hz/s)
GRANDES = ['303_WIND_1', '317_WIND_1', '122_WIND_1']
EOLICAS = GRANDES + ['309_WIND_1']
# maquinas de referencia de Anderson y Fouad (apendice D): F13 384 MVA y 1006.5 MW s; N3 500 MVA y 1990 MW s
H_F13 = 1006.5 / 384.0
H_N3 = 1990.0 / 500.0

V = {}       # nombre -> dict(valor, texto, fuente)


def pon(nombre, valor, fmt, fuente):
    if fmt == 'int':
        txt = '{:,}'.format(int(round(valor)))
    elif fmt == 'signed1':
        txt = ('%+.1f' % valor).replace('-', '$-$')
    else:
        txt = format(valor, fmt)
        if 'f' in fmt and abs(valor) >= 1000:
            txt = '{:,.{p}f}'.format(valor, p=int(fmt.split('.')[1][0]))
    V[nombre] = dict(valor=float(valor), texto=txt, fuente=fuente)


def leer(nombre):
    d = pd.read_csv(ruta_csv(os.path.join(RR, nombre + '.csv')))
    assert len(d) == 8784, nombre
    return d


def coste(d):                       # M$ en el ano (coste_dia se repite en las 24 horas del dia)
    return d.coste_dia.groupby(d.hora // 24).first().sum() / 1e6


def horas_sobre(d):                 # % de horas con tasa > 1 Hz/s ante el peor candidato
    return 100.0 * (d.rocof_peor > 1.0 + TOL).mean()


def recorte_planta(d, p):           # GWh
    return (d['wd_' + p] - d['w_' + p]).sum() / 1e3


# ------------------------------------------------------------------ constantes del modelo
cod = open(os.path.join(RAIZ, 'uc_mof1_red.py')).read()
cte = {k: float(re.search(r'^%s\s*=\s*([0-9.eE+-]+)' % k, cod, re.M).group(1)) for k in ('F0', 'TH', 'GAP', 'TLIM')}
pon('Fzero', cte['F0'], 'int', 'uc_mof1_red.py F0')
pon('Horizon', cte['TH'], 'int', 'uc_mof1_red.py TH')
pon('GapPct', 100 * cte['GAP'], '.1f', 'uc_mof1_red.py GAP')
pon('Tlim', cte['TLIM'], 'int', 'uc_mof1_red.py TLIM')
F0 = cte['F0']

bus = pd.read_csv(os.path.join(SRC, 'bus.csv'))
br = pd.read_csv(os.path.join(SRC, 'branch.csv'))
gen = pd.read_csv(os.path.join(SRC, 'gen.csv'))
pon('Buses', len(bus), 'int', 'bus.csv')
pon('Branches', len(br), 'int', 'branch.csv')
term = gen[gen['Unit Type'].isin(['CC', 'CT', 'STEAM', 'NUCLEAR'])]
pon('ThermalUnits', len(term), 'int', 'gen.csv, unidades CC, CT, STEAM y NUCLEAR')
pon('ThermalMW', term['PMax MW'].sum(), 'int', 'gen.csv')
zona = term['GEN UID'].str[0]
pon('Clusters', len(set(zip(zona, term['Unit Group']))), 'int', 'gen.csv, pares (zona, grupo)')
hid = gen[gen['Unit Group'] == 'U50']
pon('HydroUnits', len(hid), 'int', 'gen.csv U50')
pon('HydroMW', hid['PMax MW'].sum(), 'int', 'gen.csv U50')
pon('SyncCond', (gen['Unit Group'] == 'Sync_Cond').sum(), 'int', 'gen.csv Sync_Cond')
eo = gen.set_index('GEN UID').loc[EOLICAS, 'PMax MW']
for k, p in zip(['WindA', 'WindB', 'WindC', 'WindD'], EOLICAS):
    pon(k, eo[p], '.1f', 'gen.csv PMax de ' + p)
nuc = gen[gen['Unit Group'] == 'U400'].iloc[0]
pon('NucMW', nuc['PMax MW'], 'int', 'gen.csv U400')

# ------------------------------------------------------------------ energia cinetica por unidad y de la flota
par_ = [os.path.join(RAIZ, 'inertia_data', 'tabla_RTS_par_publicado.csv'),
        os.path.join(RAIZ, 'demostracion_inercia', 'salida', 'tabla_RTS_par_publicado.csv')]
par = pd.read_csv([p for p in par_ if os.path.exists(p)][0])
H_pub = par.groupby('grupo').generator_H.mean().to_dict()
H_int = dict(H_pub)
H_int.update({'U350': H_F13, 'U355': H_F13, 'U400': H_N3})
S = np.hypot(gen['PMax MW'], gen['QMax MVAR'])
g = gen['Unit Group']
e_pub = g.map(H_pub).fillna(0) * S
e_int = g.map(H_int).fillna(0) * S
# como en uc_mof1.energia_dinamica: S = |Pmax + j Qmax| del gen.csv oficial para todas las unidades,
# condensadores incluidos (200 MVA)
pon('FleetEkInt', e_int.sum(), 'int', 'H previsto x |Pmax+jQmax|, flota sincrona completa')
pon('FleetEkPub', e_pub.sum(), 'int', 'H publicado x |Pmax+jQmax|')
pon('FleetEkPct', 100 * (e_pub.sum() / e_int.sum() - 1), '.1f', 'cociente publicado/previsto')
pon('HydroEk', e_int[g == 'U50'].sum(), 'int', 'hidraulicas U50')
for grp, nm in [('U350', 'Tres'), ('U355', 'TresC'), ('U400', 'Nuc'), ('U155', 'Uno')]:
    i = g[g == grp].index[0]
    pon('Ek%sInt' % nm, e_int[i], 'int', 'unidad ' + grp + ', previsto')
    pon('Ek%sPub' % nm, e_pub[i], 'int', 'unidad ' + grp + ', publicado')
i350 = g[g == 'U355'].index[0]
pon('EkBigPct', 100 * (e_pub[i350] / e_int[i350] - 1), '.0f', 'U355 publicado/previsto')
inuc = g[g == 'U400'].index[0]
pon('EkNucPct', 100 * (e_pub[inuc] / e_int[inuc] - 1), '.1f', 'U400 publicado/previsto')
af = g.isin(['U350', 'U355', 'U400'])
pon('AffUnits', af.sum(), 'int', 'unidades U350, U355 y U400')
pon('AffMW', gen.loc[af, 'PMax MW'].sum(), 'int', 'MW de esas unidades')
pon('HFeight', 634.0 / 160.0, '.2f', 'F8: 634 MW s / 160 MVA')
pon('HFthirteen', H_F13, '.2f', 'F13')
pon('HNthree', H_N3, '.2f', 'N3')
# niveles de los suelos fijos
e_nuc = e_int[inuc]
NIVEL_NUC = 'ekin13781'      # caso resuelto: e_nuc + f0 400/2 = 13 779,9 MW s, redondeado hacia arriba
assert 0 <= int(NIVEL_NUC[4:]) - (e_nuc + F0 * nuc['PMax MW'] / 2) < 2
pon('EkNucNeed', e_nuc + F0 * nuc['PMax MW'] / 2, 'int', 'e_nuc + f0 400/2, energia que exige la perdida de la nuclear')
pon('EkNucFloor', int(NIVEL_NUC[4:]), 'int', 'nivel del caso resuelto ekin13781')
pon('ThreshExtra', 2 * 1.0 * e_nuc / F0, 'int', '2 R e_nuc / f0: margen sobre la perdida de calibracion')
pon('WindThreshold', nuc['PMax MW'] + 2 * e_nuc / F0, 'int', 'produccion eolica que exige mas energia que la perdida de la nuclear')
pon('EkMaxFloor', F0 * eo['303_WIND_1'] / 2, 'int', 'f0 847/2')
pon('EkMaxFloorPct', 100 * F0 * eo['303_WIND_1'] / 2 / e_int.sum(), '.0f', 'suelo / energia de toda la flota')
i355 = i350
pon('EkThermFloor', math.ceil(e_int[i355] + F0 * gen.loc[i355, 'PMax MW'] / 2 - 1e-9), 'int', 'U355: e + f0 P/2')

# ------------------------------------------------------------------ suelos reales (datos de los documentos citados)
# Irlanda: 23 GW s, 1 Hz/s, 50 Hz (eirgrid2025roadmap). Perdida sin inercia cubierta: 2 R E / f0
pon('IrlCoverMW', 2 * 1.0 * 23000.0 / 50.0, 'int', 'Irlanda: 2 x 1 Hz/s x 23 000 MW s / 50 Hz')
# Gran Bretana: 1800 MW a 0,5 Hz/s y 50 Hz exigen f0 P / (2 R) (nesoFrequencyRiskControl2025)
pon('GBNeedGVAs', 50.0 * 1800.0 / (2 * 0.5) / 1000.0, 'int', 'GB: 50 x 1800 / (2 x 0,5) / 1000')

# ------------------------------------------------------------------ configuracion principal (con la nuclear)
R = {r: leer(r + '_dyn_mp') for r in ['base', 'rocof1', 'ekin25410', 'ekin13781', 'muon8', 'snsp75']}
b = R['base']
pon('Days', b.hora.max() // 24 + 1, 'int', 'horas/24')
pon('Hours', len(b), 'int', 'filas')
pon('DemandGWh', b.demanda.sum() / 1e3, 'int', 'suma de demanda')
pon('PeakMW', b.demanda.max(), 'int', 'maximo de demanda')
pon('VreGWh', b.vre_disp.sum() / 1e3, 'int', 'eolica y solar disponible')
pon('VrePct', 100 * b.vre_disp.sum() / b.demanda.sum(), '.1f', 'disponible/demanda')
wmax = b[['w_' + p for p in EOLICAS]].max(axis=1)
pon('WindLargestPct', 100 * (wmax > b.p_U400 + TOL).mean(), '.1f', 'horas con la mayor eolica servida > nuclear, base')
pon('WindWorstPct', 100 * b.suceso_peor.str.contains('WIND').mean(), '.1f', 'horas con peor candidato eolico, base')
pon('MaxInfeed', b.dP_MW.max(), '.1f', 'mayor perdida del ano, base')
cb = coste(b)
pon('CostBase', cb, '.1f', 'coste anual base')
rec_b = b.recorte.sum() / 1e3
for r, nm in [('base', 'Base'), ('rocof1', 'Idx'), ('ekin25410', 'Fix'), ('ekin13781', 'Nuc'), ('muon8', 'Count'), ('snsp75', 'Snsp')]:
    d = R[r]
    pon('Hrs' + nm, horas_sobre(d), '.1f', r + ': % horas > 1 Hz/s')
    if r == 'base':
        continue
    dc = coste(d) - cb
    pon('DCost' + nm, dc, '.2f', r + ': coste incremental M$')
    pon('DCostPct' + nm, 100 * dc / cb, '.2f' if nm == 'Idx' else '.1f', r + ': % del coste base')
    pon('DCurt' + nm, d.recorte.sum() / 1e3 - rec_b, 'int', r + ': recorte incremental GWh')
    pon('DCurtPct' + nm, 100 * (d.recorte.sum() / 1e3 - rec_b) / (b.vre_disp.sum() / 1e3), '.1f', r + ': puntos del recurso')
pon('DCostFixR', V['DCostFix']['valor'], '.1f', 'ekin25410 coste incremental, una cifra decimal')
pon('DCostCountAbs', abs(V['DCostCount']['valor']), '.2f', 'muon8 |coste incremental|')
pon('Ratio', V['DCostFix']['valor'] / V['DCostIdx']['valor'], '.1f', 'cociente ekin25410/rocof1')
pon('SaveIdx', V['DCostFix']['valor'] - V['DCostIdx']['valor'], '.0f', 'ahorro del limite indexado frente al suelo fijo, M$/ano')
pon('SavePctIdx', 100 * (V['DCostFix']['valor'] - V['DCostIdx']['valor']) / cb, '.0f', 'id., % del coste base')
# horas en que el limite indexado esta activo: tasa del peor suceso a 1 Hz/s con 1 mHz/s de tolerancia
act = R['rocof1'].rocof_peor >= 1 - 1e-3
pon('BindIdxPct', 100 * act.mean(), '.1f', 'rocof1: % horas con el limite activo (tasa >= 0,999 Hz/s)')
pon('BindIdxWindPct', 100 * R['rocof1'][act].suceso_peor.str.contains('WIND').mean(), '.1f', 'rocof1: % de esas horas con peor suceso eolico')
cur = np.zeros(len(R['rocof1']), bool)
for p in EOLICAS:
    cur |= (R['rocof1']['wd_' + p] - R['rocof1']['w_' + p]).values > 1.0
wa = act.values & R['rocof1'].suceso_peor.str.contains('WIND').values
dd = R['rocof1']
setter = [(dd['wd_' + dd.suceso_peor.iloc[i]].iloc[i] - dd['w_' + dd.suceso_peor.iloc[i]].iloc[i]) > 1.0 for i in np.where(wa)[0]]
pon('BindSetterCurtPct', 100 * np.mean(setter), '.0f', 'rocof1: % de las horas activas con peor suceso eolico en que esa planta esta recortada mas de 1 MW')
pon('BindCurtPct', 100 * (act.values & cur).sum() / act.sum(), '.0f', 'rocof1: % de las horas activas con alguna eolica recortada mas de 1 MW')
pon('BindBaseExceedPct', 100 * (b.rocof_peor[act] > 1 + TOL).mean(), '.0f', 'base: % de esas mismas horas por encima de 1 Hz/s sin regla')
pon('BindCount', 100 * (b.grandes_on < 8 - TOL).mean(), '.1f', 'base: horas con menos de ocho unidades grandes')
pon('BindSnsp', 100 * (b.snsp > 0.75 + TOL).mean(), '.1f', 'base: horas con SNSP > 0,75')
pon('BindNuc', 100 * (b.E_corr_MWs < V['EkNucFloor']['valor'] - TOL).mean(), '.1f', 'base: horas bajo el suelo sobre la nuclear')
fx = R['ekin25410']
falta = V['EkMaxFloor']['valor'] - fx.E_MWs
pon('FixSlackHrs', (falta > TOL).sum(), 'int', 'ekin25410: horas con holgura del suelo')
pon('FixSlackMax', falta.max(), 'int', 'ekin25410: holgura maxima MW s')
pon('FixRocofMax', fx.rocof_peor.max(), '.3f', 'ekin25410: tasa maxima Hz/s')
pon('DollarIdx', 1e3 * V['DCostIdx']['valor'] / V['DCurtIdx']['valor'], '.0f', '$ por MWh recortado, rocof1')
pon('DollarFix', 1e3 * V['DCostFix']['valor'] / V['DCurtFix']['valor'], '.0f', '$ por MWh recortado, ekin25410')
for r, nm in [('rocof1', 'Idx'), ('ekin25410', 'Fix')]:
    d = R[r]
    tot = d.recorte.sum() / 1e3 - rec_b
    per = {p: recorte_planta(d, p) - recorte_planta(b, p) for p in EOLICAS}
    pon('Share' + nm, 100 * sum(per[p] for p in GRANDES) / tot, '.0f', r + ': % del recorte incremental en las tres grandes')
    pon('Rest' + nm, tot - sum(per.values()), 'int', r + ': recorte incremental fuera de las eolicas (solar)')
    pon('DEk' + nm, (d.E_corr_MWs - b.E_corr_MWs).mean(), 'int', r + ': energia cinetica adicional media MW s')
    pon('DUnits' + nm, (d.unid_on - b.unid_on).mean(), '.1f', r + ': unidades adicionales acopladas, media')
    pon('DTherm' + nm, (d.termica.sum() - b.termica.sum()) / 1e3, 'int', r + ': termica adicional GWh')
    pon('Dp' + nm, d.dP_MW.mean(), 'int', r + ': mayor perdida media MW')
pon('NotShareFix', 100 - V['ShareFix']['valor'], '.0f', 'ekin25410: % del recorte incremental fuera de las plantas que fijan el suceso')
pon('DpBase', b.dP_MW.mean(), 'int', 'base: mayor perdida media MW')
pon('CountHrsShare', 100 * V['HrsCount']['valor'] / V['HrsBase']['valor'], '.0f', 'muon8/base horas')

# ------------------------------------------------------------------ nuclear no obligada
nl = leer('base_dyn_mp_nuclibre')
pon('NucOffHrs', (nl.n_U400 < 0.5).sum(), 'int', 'horas con la nuclear desacoplada, nuclibre')
pon('NucStarts', ((nl.n_U400.diff() > 0.5)).sum(), 'int', 'arranques de la nuclear, nuclibre')

# ------------------------------------------------------------------ sistema sin nuclear
Q = {r: leer(r + '_dyn_mp_sinnuclear') for r in ['base', 'rocof1', 'ekin25410', 'ekin11661', 'muon8', 'snsp75']}
qb = Q['base']
assert (qb.n_U400 == 0).all()
cqb = coste(qb)
rec_qb = qb.recorte.sum() / 1e3
pon('SnWindWorstPct', 100 * qb.suceso_peor.str.contains('WIND').mean(), '.1f', 'sin nuclear: peor candidato eolico')
for r, nm in [('base', 'Base'), ('rocof1', 'Idx'), ('ekin25410', 'Fix'), ('ekin11661', 'Therm'), ('muon8', 'Count'), ('snsp75', 'Snsp')]:
    d = Q[r]
    pon('SnHrs' + nm, horas_sobre(d), '.1f', 'sin nuclear ' + r + ': % horas > 1 Hz/s')
    if r == 'base':
        continue
    pon('SnDCost' + nm, coste(d) - cqb, '.2f', 'sin nuclear ' + r + ': coste incremental M$')
    pon('SnDCurt' + nm, d.recorte.sum() / 1e3 - rec_qb, 'int', 'sin nuclear ' + r + ': recorte incremental GWh')
pon('SnDCostCountAbs', abs(V['SnDCostCount']['valor']), '.2f', 'sin nuclear muon8 |coste incremental|')
pon('SnRatio', V['SnDCostFix']['valor'] / V['SnDCostIdx']['valor'], '.1f', 'sin nuclear: cociente')
for r, nm in [('rocof1', 'Idx'), ('ekin25410', 'Fix')]:
    d = Q[r]
    tot = d.recorte.sum() / 1e3 - rec_qb
    per = {p: recorte_planta(d, p) - recorte_planta(qb, p) for p in EOLICAS}
    pon('SnShare' + nm, 100 * sum(per[p] for p in GRANDES) / tot, '.0f', 'sin nuclear ' + r + ': % en las tres grandes')
    pon('SnRest' + nm, tot - sum(per.values()), 'signed1', 'sin nuclear ' + r + ': recorte incremental fuera de las eolicas')
floor_t = V['EkThermFloor']['valor']
pon('SnThermActs', 100 * (qb.E_corr_MWs < floor_t - TOL).mean(), '.1f', 'sin nuclear: horas de base bajo el suelo de la mayor termica')
pon('SnEkMin', qb.E_corr_MWs.min(), 'int', 'sin nuclear: energia cinetica minima de base')

# ------------------------------------------------------------------ error de los datos de inercia
x = leer('rocof1_dynsabot_mp')
m = x.rocof_peor > 1.0 + TOL
pon('ErrHrs', m.sum(), 'int', 'rocof1 resuelto con datos publicados: horas > 1 Hz/s con los previstos')
pon('ErrHrsPct', 100 * m.mean(), '.1f', 'id. en %')
pon('ErrMean', (x.rocof_peor[m] - 1).mean(), '.3f', 'exceso medio Hz/s')
pon('ErrMax', (x.rocof_peor[m] - 1).max(), '.3f', 'exceso maximo Hz/s')
pon('ErrDCost', coste(x) - cb, '.2f', 'coste incremental con datos publicados')
pon('ErrDCurt', x.recorte.sum() / 1e3 - rec_b, 'int', 'recorte incremental con datos publicados')
pon('ErrCostGap', V['DCostIdx']['valor'] - (coste(x) - cb), '.2f', 'diferencia de coste entre datos')

# ------------------------------------------------------------------ resolucion numerica
rs_ = [os.path.join(RR, 'resumen_v5.csv'), os.path.join(RAIZ, 'analisis_v5_20261013', 'resumen_v5.csv')]
rs = pd.read_csv([p for p in rs_ if os.path.exists(p)][0]).set_index('caso')
dif_c, dif_r = [], []
for r in ['base', 'muon8', 'sinc2485', 'ekin12584']:
    a, z = rs.loc[r + '_dyn_mp'], rs.loc[r + '_dyn_gap0.1_mp']
    dif_c.append(abs(a.coste_MUSD - z.coste_MUSD))
    dif_r.append(abs(a.recorte_GWh - z.recorte_GWh))
pon('ResCost', max(dif_c), '.1f', 'max |dif. de coste| hueco 0,5 % frente a 0,1 %, cuatro casos')
pon('ResCurt', max(dif_r), 'int', 'max |dif. de recorte| GWh, id.')
pon('GapFine', 0.1, '.1f', 'hueco de la comprobacion')
pon('HminMinMW', 15, 'int', 'minimo hidraulico de la sensibilidad (caso _hmin)')
pon('LostIdx', rs.loc['rocof1_dyn_mp', 'delta_renov_pp'], '.2f', 'rocof1: renovable perdida, puntos (resumen_v5)')
pon('LostIdxHmin', rs.loc['rocof1_dyn_hmin_mp', 'delta_renov_pp'], '.2f', 'rocof1 con minimo hidraulico: puntos (resumen_v5)')
# reglas que aseguran todas las horas resueltas con hueco del 0,1 %, con y sin nuclear: se declaran si existen
g01 = {s: [ruta_csv(os.path.join(RR, n + '_dyn_gap0.1_mp' + s + '.csv')) for n in ('base', 'rocof1', 'ekin25410')]
       for s in ('', '_sinnuclear')}
pre = {'': 'D', '_sinnuclear': 'SnD'}
if all(os.path.exists(p) for s in g01 for p in g01[s]):
    dif, cocientes = [], []
    for s, ps in g01.items():
        bg, ig, fg = (pd.read_csv(p) for p in ps)
        ci, cf = coste(ig) - coste(bg), coste(fg) - coste(bg)
        dif += [abs(ci - V[pre[s] + 'CostIdx']['valor']), abs(cf - V[pre[s] + 'CostFix']['valor'])]
        cocientes.append(cf / ci)
    pon('ResIdxFix', max(dif), '.1f', 'rocof1 y ekin25410 con hueco 0,1 %, con y sin nuclear: max |dif. de coste incremental|')
    pon('ResRatioMin', min(cocientes), '.1f', 'cociente ekin25410/rocof1 con hueco 0,1 %, minimo de con y sin nuclear')
    pon('ResRatioMax', max(cocientes), '.1f', 'id., maximo')
    V['GapRuleSentence'] = dict(valor=float('nan'), texto=('Re-solved with the narrower gap, with and without the nuclear unit, the '
                                'additional cost of the two rules that secure every hour changes by at most %s~M\\$ and the ratio '
                                'of their costs is %s to %s.' % (V['ResIdxFix']['texto'], V['ResRatioMin']['texto'],
                                                                 V['ResRatioMax']['texto'])), fuente='texto')
else:
    V['GapRuleSentence'] = dict(valor=float('nan'), texto=('The two rules that secure every hour have not been re-solved with the '
                                'narrower gap, so their cost ratio is stated only approximately.'), fuente='texto')
TODOS = list(R.values()) + list(Q.values()) + [nl, x]
dias_cortados = {k: int((d.groupby(d.hora // 24).gap.first() > cte['GAP'] + 1e-9).sum()) for k, d in
                 list(R.items()) + [(k + '_sn', d) for k, d in Q.items()]}
pon('CutDaysMax', max(dias_cortados.values()), 'int', 'dias con hueco final > 0,5 %% (limite de tiempo), maximo por caso: %s' % dias_cortados)
pon('CutDaysBase', dias_cortados['base'], 'int', 'base: dias cortados por tiempo')
pon('CutDaysIdx', dias_cortados['rocof1'] + dias_cortados['ekin25410'], 'int', 'rocof1 y ekin25410: dias cortados')
pon('CutDaysFixSn', dias_cortados['ekin25410_sn'], 'int', 'ekin25410 sin nuclear: dias cortados')
pon('HydroOnMin', min(d.hidro_on.min() for d in TODOS), 'int', 'hidraulicas acopladas, minimo en todas las horas de todos los casos')

pon('ShedMWh', max(d.deslastre.sum() for d in TODOS), 'int', 'deslastre maximo de todos los casos')
pon('ResShortMWh', max(d.reserva_falta.sum() for d in TODOS), 'int', 'falta de reserva maxima, MWh/ano')

# ------------------------------------------------------------------ escritura
with open(os.path.join(SALIDA, 'cifras.tex'), 'w') as f:
    f.write('%% Generado por cifras.py; no editar a mano.\n')
    for k in sorted(V):
        f.write('\\newcommand{\\V%s}{%s}%% %s\n' % (k, V[k]['texto'], V[k]['fuente']))
json.dump(V, open(os.path.join(SALIDA, 'cifras.json'), 'w'), indent=1, ensure_ascii=False)
print('%d cifras escritas en cifras.tex y cifras.json' % len(V))
