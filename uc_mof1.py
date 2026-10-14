#!/usr/bin/env python3
"""Prueba de viabilidad de MOF1 sobre el RTS-GMLC.

Compromiso de unidades diario por agrupaciones de grupos identicos, sin red
(nudo unico), con reservas, hidraulica programada, almacenamiento y la nuclear.
Mide el recorte anual de renovable variable bajo cada criterio indirecto de
seguridad del operador frente al caso sin criterio. Independiente del codigo
MATLAB del proyecto.

Uso:  python3 uc_mof1.py <config> [dia_inicial dia_final] [--dyn | --dynsabot]
Las configuraciones estan en CONFIGS; config() admite ademas sincNNNN y ekinNNNN (v5, 12/10/2026). Escribe resultados/<config>.csv por hora.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

D = 'RTS-GMLC/RTS_Data/'
PDSA = 'PDSA-RTS-GMLC/'
F0 = 60.0
T = 24
HIST = 48
TH = 36
AGRUPAR = True
GAP = 5e-3
TLIM = 30
NUC_FIJA = True
HMIN = 0.0

CONFIGS = {
    'base': {},
    'snsp75': {'snsp': 0.75},
    'snsp90': {'snsp': 0.90},
    'snsp95': {'snsp': 0.95},
    'rocof1': {'rocof': 1.0},
    'rocof05': {'rocof': 0.5},
    'muon4': {'muon': 4},
    'muon6': {'muon': 6},
    'muon8': {'muon': 8},
    'muon8any': {'muon': 8, 'muon_any': True},
    'paquete': {'snsp': 0.75, 'rocof': 1.0, 'muon': 8},
    # Campana v4 (08/10/2026): el minimo de generacion sincrona escrito de tres formas. Las tres
    # admiten exactamente el mismo conjunto minimo en la flota publicada: la nuclear de 400 MW y
    # las siete unidades de 155 MW, que es el conjunto mas pequeno con ocho unidades grandes.
    #   muon8      ocho unidades de 155 MW o mas (forma de EirGrid y AEMO)
    #   sinc8      potencia sincrona acoplada >= la de ese conjunto, 1485 MW (forma en MW)
    #   ekin8      energia cinetica acoplada, condensadores incluidos, >= la de ese conjunto mas la
    #              de los condensadores, fija e independiente de la mayor perdida (forma en MW s,
    #              como el suelo de inercia de EirGrid)
    # Los niveles de sinc8 y ekin8 se calculan sobre la flota publicada con el mismo conjunto de
    # inercia del caso y no cambian con la flota (uc_mof1_red.niveles_conjunto8).
    'sinc8': {'sinc8': True},
    'ekin8': {'ekin8': True},
    # Campana v4b (09/10/2026): las formas en MW y en MW s a varios niveles, incluidos los que
    # garantizan en cada hora lo que el recuento de ocho unidades garantiza de hecho: la potencia
    # sincrona y la energia cinetica minimas que acopla muon8 en todo el ano, 2485 MW y 12585 MW s
    # en la flota publicada y 3017 MW y 14581 MW s en la de 355 MW (resultados_v4, inercia
    # corregida, condensadores incluidos). ekin13781 es el suelo de RoCoF de 1 Hz/s escrito solo
    # para la perdida de la nuclear, e_nuc + f0 P / (2 R), y sirve de control frente a rocof1.
    'sinc2000': {'sinc_mw': 2000.0}, 'sinc2485': {'sinc_mw': 2485.0}, 'sinc3000': {'sinc_mw': 3000.0},
    'sinc3017': {'sinc_mw': 3017.0},
    'ekin10000': {'ekin_mws': 10000.0}, 'ekin12585': {'ekin_mws': 12585.0},
    'ekin13781': {'ekin_mws': 13781.0}, 'ekin14581': {'ekin_mws': 14581.0},
    # Campana v5 (12/10/2026): suelo fijo de energia cinetica que garantiza 1 Hz/s ante la mayor
    # perdida del ano, la 303_WIND_1 a 847 MW, que no aporta inercia: f0 * 847 / (2 * 1 Hz/s) =
    # 25 410 MW s, con los condensadores en el lado izquierdo. Es la forma del suelo fijo de EirGrid
    # calibrado sobre la mayor perdida, y el control de rocof1 con la mayor perdida de cada hora.
    'ekin25410': {'ekin_mws': 25410.0},
}


def config(nombre):
    """Configuracion de un caso por su nombre (campana v5, 12/10/2026).

    Devuelve CONFIGS[nombre] si existe. Si no, admite los niveles fijos escritos en el nombre,
    sincNNNN (potencia sincrona acoplada >= NNNN MW) y ekinNNNN (energia cinetica acoplada >= NNNN
    MW s, condensadores incluidos), para los niveles emparejados que niveles_emparejados.py lee
    de los casos muon8 de la campana v5. Los nombres ya definidos dan la misma configuracion que antes.
    """
    import re
    if nombre in CONFIGS:
        return CONFIGS[nombre]
    m = re.fullmatch(r'(sinc|ekin)(\d+)', nombre)
    if m is None:
        raise KeyError('configuracion desconocida: %s' % nombre)
    return {'sinc_mw': float(m.group(2))} if m.group(1) == 'sinc' else {'ekin_mws': float(m.group(2))}


def serie(f):
    d = pd.read_csv(D + 'timeseries_data_files/' + f)
    return d.drop(columns=['Year', 'Month', 'Day', 'Period'])


def energia_dinamica(g, dyn='corr'):
    """Energia cinetica nominal por unidad (MW s) con el conjunto dinamico de Sabot et al.

    E = H * S, con la constante de inercia H que el conjunto dinamico (3-DynData/RTS.par) asigna
    al grupo de la unidad y la potencia aparente nominal S de la red oficial, el modulo del limite
    de capacidad |Pmax + j Qmax| del gen.csv oficial (capitulo de bancos de la tesis). No se toma
    la generator_SNom del .par porque es la de la variante de red de Sabot et al., que cambia los
    limites de reactiva de treinta y dos generadores y redefine los condensadores a 100 MVA.

    Con dyn='sabot' se usan las H tal como estan publicadas. Con dyn='corr' se corrigen los grupos
    de 350, 355 y 400 MW: en 3-DynData/add_dyn_data.py la condicion
    `elif unit_group == 'Coal_155' or 'CC_150':` es siempre cierta y esos grupos reciben los
    parametros de F8 (H = 634/160 = 3,9625 s), mientras que las ramas muertas del mismo fichero
    les asignan F13 (350 y 355 MW) y N3 (nuclear). Los valores corregidos son los del apendice D
    de Anderson y Fouad (2003): energia cinetica nominal dividida por la potencia aparente nominal
    de la maquina tipo, 1006,5/384 y 1990/500.
    """
    import xml.etree.ElementTree as ET
    import re
    ns = {'d': 'http://www.rte-france.com/dynawo'}
    raiz = ET.parse(PDSA + '3-DynData/RTS.par').getroot()
    par = {}
    for s in raiz.findall('d:set', ns):
        if re.match(r'^\d+_', s.get('id')):
            par[s.get('id')] = {p.get('name'): p.get('value') for p in s.findall('d:par', ns)}
    gp = pd.read_csv(PDSA + 'RTS-Data/gen.csv')
    Hg = {}
    for grupo, dd in gp.groupby('Unit Group'):
        v = {par[u]['generator_H'] for u in dd['GEN UID'] if 'generator_H' in par.get(u, {})}
        if len(v) == 1:
            Hg[grupo] = float(v.pop())
    if dyn != 'sabot':
        Hg.update(U350=1006.5 / 384.0, U355=1006.5 / 384.0, U400=1990.0 / 500.0)
    S = np.sqrt(g['PMax MW'] ** 2 + g['QMax MVAR'] ** 2)
    return np.array([Hg.get(x, 0.0) for x in g['Unit Group']]) * S.values


def cargar(dyn=False, gruesa=0):
    # dyn: False usa la columna de inercia de gen.csv; 'corr' (o True) el conjunto dinamico de
    # Sabot et al. con la correccion de los grupos de 350, 355 y 400 MW; 'sabot' el conjunto tal
    # como esta publicado.
    if dyn is True:
        dyn = 'corr'
    g = pd.read_csv(D + 'SourceData/gen.csv')
    th = g[g['Unit Type'].isin(['CC', 'CT', 'STEAM', 'NUCLEAR'])].copy()
    if dyn:
        g['E'] = energia_dinamica(g, dyn)
        th['E'] = g.loc[th.index, 'E']
        e_hidro = g[g['Unit Type'] == 'HYDRO']['E'].iloc[0]
        e_sc = g[g['Unit Type'] == 'SYNC_COND']['E'].sum()
    else:
        th['E'] = th['Inertia MJ/MW'] * th['Base MVA']
        e_hidro = 3.5 * 53.0
        e_sc = 0.0
    claves = ['Unit Group', 'Fuel', 'PMax MW', 'PMin MW', 'Min Up Time Hr', 'Min Down Time Hr',
              'Ramp Rate MW/Min', 'Fuel Price $/MMBTU', 'HR_avg_0', 'HR_incr_1', 'HR_incr_2',
              'HR_incr_3', 'Output_pct_1', 'Output_pct_2', 'Start Heat Hot MBTU',
              'Non Fuel Start Cost $', 'E']
    if AGRUPAR:
        # Un equivalente por grupo de unidad: los parametros tecnicos son comunes y los de
        # coste se promedian; se pierde solo el orden de merito dentro del grupo.
        num = ['Fuel Price $/MMBTU', 'HR_avg_0', 'HR_incr_1', 'HR_incr_2', 'HR_incr_3']
        th[num] = th.groupby('Unit Group')[num].transform('mean')
    cl = th.groupby(claves).size().reset_index(name='N')
    if gruesa:
        # Flota gruesa: cada grupo se reagrupa en unidades mayores, del orden de las mayores
        # termicas del banco: k = min(N, floor(355/Pmax)) y round(N/k) unidades resultantes, con la misma potencia, minimo tecnico, energia cinetica,
        # rampa y coste de arranque totales. El mayor inyector sigue siendo la nuclear de 400 MW.
        # Solo cambia el tamano de la unidad.
        cl = cl.groupby('Unit Group', as_index=False).agg(
            {**{k: 'first' for k in claves if k != 'Unit Group'}, 'N': 'sum'})
        k = np.clip(np.floor(355.0 / cl['PMax MW']), 1, cl['N'])
        nuevo = np.maximum(np.round(cl['N'] / k), 1).astype(int)
        f = cl['N'] / nuevo
        for k in ['PMax MW', 'PMin MW', 'Ramp Rate MW/Min', 'E', 'Start Heat Hot MBTU', 'Non Fuel Start Cost $']:
            cl[k] = cl[k] * f
        cl['N'] = nuevo
    c = pd.DataFrame({
        'grupo': cl['Unit Group'], 'N': cl['N'], 'Pmax': cl['PMax MW'], 'Pmin': cl['PMin MW'].astype(float),
        'UT': np.ceil(cl['Min Up Time Hr']).astype(int), 'DT': np.ceil(cl['Min Down Time Hr']).astype(int),
        'RU': cl['Ramp Rate MW/Min'] * 60.0, 'e': cl['E'],
    })
    pr = cl['Fuel Price $/MMBTU']
    c['c0'] = pr * cl['HR_avg_0'] * cl['PMin MW'] / 1000.0          # coste a minimo tecnico por grupo
    p1 = cl['Output_pct_1'] * cl['PMax MW']
    p2 = cl['Output_pct_2'] * cl['PMax MW']
    c['w1'] = np.maximum(p1 - cl['PMin MW'], 0.0)
    c['w2'] = np.maximum(p2 - p1, 0.0)
    c['w3'] = np.maximum(cl['PMax MW'] - p2, 0.0)
    c['m1'] = pr * cl['HR_incr_1'] / 1000.0
    c['m2'] = pr * cl['HR_incr_2'] / 1000.0
    c['m3'] = pr * cl['HR_incr_3'] / 1000.0
    c['cs'] = pr * cl['Start Heat Hot MBTU'] + cl['Non Fuel Start Cost $']
    c['grande'] = c['Pmax'] >= 155
    load = serie('Load/DAY_AHEAD_regional_Load.csv').sum(axis=1).values
    vre = (serie('WIND/DAY_AHEAD_wind.csv').sum(axis=1) + serie('PV/DAY_AHEAD_pv.csv').sum(axis=1)
           + serie('RTPV/DAY_AHEAD_rtpv.csv').sum(axis=1)).values
    hyd = serie('Hydro/DAY_AHEAD_hydro.csv').sum(axis=1).values
    res = sum(serie('Reserves/DAY_AHEAD_regional_%s.csv' % k).sum(axis=1).values
              for k in ('Spin_Up_R1', 'Spin_Up_R2', 'Spin_Up_R3'))
    # La regulacion viene en formato dia por fila con 24 columnas horarias.
    rg = pd.read_csv(D + 'timeseries_data_files/Reserves/DAY_AHEAD_regional_Reg_Up.csv')
    res = res + rg.drop(columns=['Year', 'Month', 'Day']).values.reshape(-1)
    return c, dict(load=load, vre=vre, hyd=hyd, res=res, e_hidro=e_hidro, e_sc=e_sc, nh=20)


class Modelo:
    def __init__(self):
        self.lb, self.ub, self.cost, self.integ = [], [], [], []
        self.rows, self.cols, self.vals, self.rlb, self.rub = [], [], [], [], []
        self.nr = 0

    def var(self, n, lb=0.0, ub=np.inf, cost=0.0, integer=False):
        i0 = len(self.lb)
        self.lb += list(np.broadcast_to(lb, n)); self.ub += list(np.broadcast_to(ub, n))
        self.cost += list(np.broadcast_to(cost, n)); self.integ += [1 if integer else 0] * n
        return np.arange(i0, i0 + n)

    def fila(self, idx, coef, lo, hi):
        self.rows += [self.nr] * len(idx); self.cols += list(idx); self.vals += list(coef)
        self.rlb.append(lo); self.rub.append(hi); self.nr += 1

    def resolver(self, gap, tlim):
        A = sp.csr_matrix((self.vals, (self.rows, self.cols)), shape=(self.nr, len(self.lb)))
        r = milp(np.array(self.cost), integrality=np.array(self.integ),
                 bounds=Bounds(np.array(self.lb), np.array(self.ub)),
                 constraints=LinearConstraint(A, np.array(self.rlb), np.array(self.rub)),
                 options={'mip_rel_gap': gap, 'time_limit': tlim, 'disp': False})
        return r


def dia(c, dat, cfg, d0, estado, gap=None, tlim=None):
    gap = GAP if gap is None else gap
    tlim = TLIM if tlim is None else tlim
    T = TH                       # horizonte con anticipacion; se aplican las 24 primeras horas
    h = np.minimum(np.arange(d0 * 24, d0 * 24 + T), len(dat['load']) - 1)
    L, V, Hy, R = dat['load'][h], dat['vre'][h], dat['hyd'][h], dat['res'][h]
    nC = len(c)
    m = Modelo()
    n = [m.var(T, c.N[k] if (NUC_FIJA and c.grupo[k] == 'U400') else 0, c.N[k], integer=True)
         for k in range(nC)]
    s = [m.var(T, 0, c.N[k], cost=c.cs[k], integer=True) for k in range(nC)]
    dd = [m.var(T, 0, c.N[k], integer=True) for k in range(nC)]
    for k in range(nC):
        for t in range(T):
            m.cost[n[k][t]] = c.c0[k]
    sg = [[m.var(T, cost=c['m%d' % j][k]) for j in (1, 2, 3)] for k in range(nC)]
    r = [m.var(T) for k in range(nC)]
    # En la sensibilidad HMIN el numero de hidraulicas acopladas se relaja a continuo: su inercia
    # queda acotada por la produccion (nh <= ph/HMIN) sin anadir enteros al programa.
    nh = m.var(T, 0, dat['nh'], integer=(HMIN == 0))
    ph = m.var(T, 0, Hy, cost=-0.1)
    w = m.var(T, 0, V, cost=-0.5)
    ch = m.var(T, 0, 50.0); dis = m.var(T, 0, 50.0); soc = m.var(T, 0, 150.0)
    shed = m.var(T, cost=1e4); rsh = m.var(T, cost=1e3)
    # balance
    for t in range(T):
        idx, cf = [], []
        for k in range(nC):
            idx += [n[k][t], sg[k][0][t], sg[k][1][t], sg[k][2][t]]; cf += [c.Pmin[k], 1, 1, 1]
        idx += [ph[t], w[t], dis[t], ch[t], shed[t]]; cf += [1, 1, 1, -1, 1]
        m.fila(idx, cf, L[t], L[t])
    P = lambda k, t: ([n[k][t], sg[k][0][t], sg[k][1][t], sg[k][2][t]], [c.Pmin[k], 1, 1, 1])
    for k in range(nC):
        for t in range(T):
            for j in range(3):
                m.fila([sg[k][j][t], n[k][t]], [1, -c['w%d' % (j + 1)][k]], -np.inf, 0)
            i, f = P(k, t)
            m.fila(i + [r[k][t], n[k][t]], f + [1, -c.Pmax[k]], -np.inf, 0)
            # logica de estados
            if t == 0:
                m.fila([n[k][0], s[k][0], dd[k][0]], [1, -1, 1], estado['n'][k], estado['n'][k])
            else:
                m.fila([n[k][t], n[k][t - 1], s[k][t], dd[k][t]], [1, -1, -1, 1], 0, 0)
            # minimos de funcionamiento y parada, con historia
            UT, DT = c.UT[k], c.DT[k]
            hs = estado['hs'][k][HIST - max(UT - 1 - t, 0):] if UT - 1 - t > 0 else []
            m.fila(list(s[k][max(0, t - UT + 1):t + 1]) + [n[k][t]],
                   [1] * (t + 1 - max(0, t - UT + 1)) + [-1], -np.inf, -float(np.sum(hs)))
            hd = estado['hd'][k][HIST - max(DT - 1 - t, 0):] if DT - 1 - t > 0 else []
            m.fila(list(dd[k][max(0, t - DT + 1):t + 1]) + [n[k][t]],
                   [1] * (t + 1 - max(0, t - DT + 1)) + [1], -np.inf, c.N[k] - float(np.sum(hd)))
            # rampas
            i, f = P(k, t)
            if t == 0:
                m.fila(i + [n[k][t], s[k][t]], f + [-c.RU[k], c.RU[k] - c.Pmax[k]], -np.inf, estado['p'][k])
                m.fila(i + [n[k][t], dd[k][t]], f + [c.RU[k], c.Pmax[k]], estado['p'][k], np.inf)
            else:
                i0, f0 = P(k, t - 1)
                m.fila(i + i0 + [n[k][t], s[k][t]], f + [-x for x in f0] + [-c.RU[k], c.RU[k] - c.Pmax[k]], -np.inf, 0)
                m.fila(i + i0 + [n[k][t], dd[k][t]], f + [-x for x in f0] + [c.RU[k], c.Pmax[k]], 0, np.inf)
    for t in range(T):
        m.fila(list(r[k][t] for k in range(nC)) + [rsh[t]], [1] * nC + [1], R[t], np.inf)
        m.fila([ph[t], nh[t]], [1, -50.0], -np.inf, 0)
        if HMIN > 0:
            # Sensibilidad: una hidraulica acoplada produce al menos HMIN MW, de modo que no
            # puede aportar inercia en vacio.
            m.fila([ph[t], nh[t]], [1, -HMIN], 0, np.inf)
        prev = [soc[t - 1]] if t > 0 else []
        m.fila([soc[t]] + prev + [ch[t], dis[t]], [1] + ([-1] if t > 0 else []) + [-0.922, 1 / 0.922],
               estado['soc'] if t == 0 else 0, estado['soc'] if t == 0 else 0)
    m.fila([soc[T - 1]], [1], 75.0, np.inf)
    # criterios indirectos
    if 'snsp' in cfg:
        a = cfg['snsp']; sh = m.var(T, cost=1e4)
        for t in range(T):
            m.fila([w[t], dis[t], ch[t], sh[t]], [1, 1, -a, -1], -np.inf, a * L[t])
    if 'muon' in cfg:
        sh = m.var(T, cost=1e5)
        sel = [k for k in range(nC) if (cfg.get('muon_any') or c.grande[k])]
        for t in range(T):
            idx = [n[k][t] for k in sel]; cf = [1] * len(sel)
            if cfg.get('muon_any'):
                idx.append(nh[t]); cf.append(1)
            m.fila(idx + [sh[t]], cf + [1], cfg['muon'], np.inf)
    if 'rocof' in cfg:
        Rr = cfg['rocof']; sh = m.var(T, cost=100.0)
        y = [m.var(T, 0, 1, integer=True) for k in range(nC)]
        for t in range(T):
            Eidx = [n[k][t] for k in range(nC)] + [nh[t]]
            Ecf = [c.e[k] for k in range(nC)] + [dat['e_hidro']]
            for k in range(nC):
                m.fila([y[k][t], n[k][t]], [c.N[k], -1], 0, np.inf)
                m.fila(Eidx + [y[k][t], sh[t]], Ecf + [-(c.e[k] + F0 * c.Pmax[k] / (2 * Rr)), 1],
                       -dat['e_sc'], np.inf)
    t1 = time.time()
    res = m.resolver(gap, tlim)
    if res.x is None:
        raise RuntimeError('dia %d sin solucion: %s' % (d0, res.message))
    x = res.x
    A = 24
    nv = np.array([[x[n[k][t]] for t in range(T)] for k in range(nC)]).round()
    pv = np.array([[c.Pmin[k] * nv[k, t] + sum(x[sg[k][j][t]] for j in range(3)) for t in range(T)] for k in range(nC)])
    sv = np.array([[x[s[k][t]] for t in range(A)] for k in range(nC)]).round()
    dv = np.array([[x[dd[k][t]] for t in range(A)] for k in range(nC)]).round()
    nuevo = {'n': nv[:, A - 1], 'p': pv[:, A - 1], 'soc': x[soc[A - 1]],
             'hs': [np.concatenate([estado['hs'][k], sv[k]])[-HIST:] for k in range(nC)],
             'hd': [np.concatenate([estado['hd'][k], dv[k]])[-HIST:] for k in range(nC)]}
    xs = lambda v: x[v][:A]
    nv, pv = nv[:, :A], pv[:, :A]
    E = (c.e.values[:, None] * nv).sum(0) + dat['e_hidro'] * xs(nh) + dat['e_sc']
    grandes = nv[c.grande.values].sum(0)
    # coste de explotacion de las 24 horas aplicadas, sin los terminos de desempate
    coste = (c.c0.values[:, None] * nv).sum() + (c.cs.values[:, None] * sv).sum() + sum(
        c['m%d' % (j + 1)][k] * x[sg[k][j][:A]].sum() for k in range(nC) for j in range(3))
    L, V, Hy = L[:A], V[:A], Hy[:A]
    out = pd.DataFrame({
        'hora': np.arange(d0 * 24, d0 * 24 + A), 'demanda': L, 'vre_disp': V, 'vre': xs(w),
        'recorte': V - xs(w), 'hidro_disp': Hy, 'hidro': xs(ph), 'carga_alm': xs(ch), 'desc_alm': xs(dis),
        'termica': pv.sum(0), 'nuclear_on': nv[c.grupo.values == 'U400'].sum(0),
        'unid_on': nv.sum(0), 'grandes_on': grandes, 'hidro_on': xs(nh).round(), 'E_MWs': E,
        'snsp': (xs(w) + xs(dis)) / (L + xs(ch)), 'deslastre': xs(shed), 'reserva_falta': xs(rsh),
        'coste_dia': coste, 'gap': getattr(res, 'mip_gap', np.nan), 'seg': time.time() - t1,
    })
    return out, nuevo


def main():
    nombre = sys.argv[1]
    dyn = 'sabot' if '--dynsabot' in sys.argv else ('corr' if '--dyn' in sys.argv else False)
    gruesa = 4 if '--gruesa' in sys.argv else 0
    global HMIN
    if '--hmin' in sys.argv:
        HMIN = 15.0
    args = [a for a in sys.argv[2:] if not a.startswith('--')]
    a, b = (int(args[0]), int(args[1])) if len(args) >= 2 else (0, 366)
    c, dat = cargar(dyn, gruesa)
    cfg = CONFIGS[nombre]
    estado = {'n': c.N.values.astype(float), 'p': (c.Pmin * c.N).values, 'soc': 75.0,
              'hs': [np.zeros(HIST) for _ in range(len(c))], 'hd': [np.zeros(HIST) for _ in range(len(c))]}
    os.makedirs('resultados', exist_ok=True)
    partes = []
    for d0 in range(a, b):
        o, estado = dia(c, dat, cfg, d0, estado)
        partes.append(o)
        if d0 % 30 == 0:
            print(nombre, d0, '%.1fs' % o.seg.iloc[0], flush=True)
    sufijo = {False: '', 'corr': '_dyn', 'sabot': '_dynsabot'}[dyn] + ('_gruesa' if gruesa else '') + ('_hmin' if HMIN > 0 else '')
    pd.concat(partes).to_csv('resultados/%s%s.csv' % (nombre, sufijo), index=False)


if __name__ == '__main__':
    main()
