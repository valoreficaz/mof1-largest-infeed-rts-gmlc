#!/usr/bin/env python3
"""MOF1 con red en continua y embalses hidraulicos sobre el RTS-GMLC.

Extiende uc_mof1.py (nudo unico, hidraulica de pasada) con:
  - red en continua de 73 nudos y 120 ramas, limites termicos 'Cont Rating', mas el enlace de
    continua DC1 (113-316) como flujo controlable de +-100 MW; factores de distribucion (PTDF)
    con restricciones de rama anadidas de forma perezosa: se resuelve, se calculan los flujos de
    todas las ramas y horas, se anaden las ramas que se pasan de su limite y se vuelve a resolver;
    las ramas que han atado alguna vez se vigilan ya desde el principio en los dias siguientes;
  - embalses de las hidraulicas del fichero storage.csv (1 GWh por unidad, nivel inicial 0,5 GWh,
    aportacion = serie DAY_AHEAD_hydro_inflow), agregados por nudo; vertido solo cuando el embalse
    esta lleno; la unidad 201_HYDRO_4 (de pasada) sin embalse; el nivel al final del horizonte no
    puede quedar por debajo del nivel al principio del dia;
  - grupos termicos agregados por zona del banco y grupo de unidad (19 equivalentes, cuya
    inyeccion se reparte entre sus nudos en proporcion a la potencia instalada; con
    POR_NUDO = True, por nudo y grupo, 41 equivalentes), con el coste promediado dentro del grupo
    de unidad, como en uc_mof1.py;
  - reserva rodante N-1: el margen de subida termico e hidraulico cubre al menos la perdida de la
    nuclear de 400 MW (con --suceso=mayor, la mayor perdida de cada hora) y nunca menos que el
    requisito del banco (--sinN1 para la version anterior);
  - salida por grupo (n_*, p_*) para las comprobaciones a posteriori.
Los criterios C1, C2 y C3 y la verificacion de frecuencia son los de uc_mof1.py.

Campana v5 (12/10/2026), rediseno con la mayor perdida de cada hora (plan_MOF1.md, apartado 00 bis),
tomado del prototipo uc_mof1_red_mayorperdida.py de la revision independiente del 12/10/2026:
  --suceso=mayor  el suceso dimensionante es la mayor perdida de cada hora entre la nuclear, por su
                  produccion programada, y cada planta eolica, por su produccion servida:
                  * reserva N-1 >= dP_t, con dP_t >= cada candidato (variable continua, sin binaria);
                  * rocof: una fila por candidato con su propia produccion y su propia inercia, y
                    una binaria por equivalente termico (no por grupo; ver la nota de ks[0]);
                  * activa ademas la agregacion corregida de la energia cinetica (ESUMA, abajo);
                  * sufijo _mp en el fichero de resultados.
  --esuma / --sinesuma  fuerzan la agregacion corregida o la de v4 con independencia del suceso
                  (solo para pruebas; sufijos _esuma y _e1 cuando difieren de lo que implica el suceso).
Sin estas opciones el programa es exactamente el de v4: mismas filas, mismas variables y mismos
resultados (comprobado en los dias 0 a 3 de base_dyn). En todos los modos la salida anade columnas
de diagnostico que no entran en el programa: produccion servida y disponible de cada planta eolica
(w_<UID>, wd_<UID>), la mayor perdida de la hora (dP_MW), la energia cinetica con el conjunto
corregido (E_corr_MWs), y la tasa inicial ante el peor candidato con ese conjunto y quien es
(rocof_peor, suceso_peor). Al terminar imprime 'escrito <fichero>'.

Uso:  python3 uc_mof1_red.py <config> [dia_inicial dia_final] [--dyn | --dynsabot] [--gruesa | --flota=MW] [--hmin]
                              [--suceso=mayor [--nuclibre]] [--gap=G] [--tlim=S]
Escribe resultados_red/<config>[_dyn|_dynsabot][_gruesa][_fMW][_hmin][_gapG][_mp][_nuclibre].csv por hora.
--nuclibre (13/10/2026): la nuclear deja de estar obligada; ver la nota de NUCLIBRE.
--sinnuclear (13/10/2026): la nuclear no esta disponible en todo el ano; ver la nota de SINNUCLEAR.
                              [--suceso=mayor [--nuclibre | --sinnuclear]]
<config> es una clave de uc_mof1.CONFIGS o un nivel fijo sincNNNN o ekinNNNN (uc_mof1.config).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import uc_mof1 as U
from uc_mof1 import Modelo, serie, CONFIGS

D = U.D
F0 = 60.0
T = 24
TH = 36
HIST = 48
GAP = 5e-3
TLIM = 120        # 06/10/2026: 30 s cortaba antes del hueco del 0,5 % en parte de los dias (--tlim N)
HMIN = 0.0
MAX_ITER = 8
POR_NUDO = False
RES_N1 = True     # reserva rodante >= mayor inyector (--sinN1 para la version anterior)
P_N1 = 400.0
# Campana v5 (12/10/2026)
SUCESO = 'nuclear'   # 'nuclear': comportamiento de la campana v4; 'mayor': mayor perdida de cada hora
# Agregacion de la energia cinetica de cada equivalente de zona y grupo. False (v4): la de la primera
# unidad del equivalente (dd.iloc[0]), que omite los 265 MW s de mas de la 315_CT_8 (S = 81,4 MVA
# frente a 58,2 MVA del resto de U55 del area 3): la flota acoplada entera suma 47 973 MW s en lugar
# de 48 238. True (v5): la media de las unidades del equivalente, de modo que N * e es la suma de la
# energia de cada unidad. Cambia el resultado, por eso solo se activa con --suceso=mayor (o --esuma).
# La comprobacion de la perdida de una unidad termica usa esa media; la 315_CT_8 tiene 55 MW y su
# fila queda dominada por la de la nuclear, de modo que no cambia ninguna tasa que importe.
ESUMA = False
# Campana nuclibre (13/10/2026): la nuclear deja de estar obligada (--nuclibre, solo con --suceso=mayor).
# Por omision (False) el programa es exactamente el de v5: mismas filas, mismas variables y mismos
# resultados (comprobado en los dias 0 a 2 de base_dyn_mp y rocof1_dyn_mp). Con --nuclibre:
#   * la nuclear es una variable entera 0-1 como las demas, con sus datos del banco: minimo tecnico
#     396 MW, 24 h de funcionamiento minimo, 48 h de parada minima y 20 MW/min de rampa;
#   * coste de arranque: el banco no da uno utilizable para la nuclear. Sus tiempos de arranque son
#     9999 h (marca de unidad que no se arranca), el calor de arranque en caliente, que es el que usa
#     este programa para las demas unidades, es 9999 MBTU (la misma marca) y el templado es 0. Se toma
#     el unico dato fisico que da, el calor de arranque en frio, 78 978 MBTU a 0,81035 $/MMBTU,
#     64 000 $ por arranque (unos 160 $/MW). Es arranque en frio porque tras 48 h de parada minima la
#     unidad esta fria. La cifra es baja frente a la de una central nuclear real; es la del banco, y el
#     efecto de un arranque mas caro es acoplar la nuclear mas horas, es decir, acercarse a v5;
#   * la nuclear solo es candidata a la mayor perdida cuando esta acoplada: su fila de RoCoF es
#     E_t - e_nuc n_nuc,t >= a p_nuc,t, con n_nuc,t la propia variable 0-1 (desacoplada, p_nuc,t = 0 y
#     E_t no incluye su energia), y la reserva sigue exigiendo dP_t >= p_nuc,t;
#   * las demas termicas pasan a ser candidatas, porque con la nuclear parada ya no las domina: por
#     escalon de potencia maxima L (unidades iguales de distintas zonas), una binaria z_L,t con
#     N_L z_L,t >= suma de sus unidades acopladas; reserva dP_t >= L z_L,t y RoCoF
#     E_t - e_L z_L,t >= a L z_L,t, con e_L la mayor energia por unidad del escalon. Se omiten las filas
#     que no pueden atar: la de reserva en las horas en que L no pasa del requisito del banco, y la de
#     RoCoF cuando la energia de los condensadores ya cubre a L;
#   * las fotovoltaicas (la mayor tiene 188 MW), el enlace de continua (100 MW), la bateria (50 MW) y
#     las hidraulicas (50 MW) siguen sin ser candidatas; la salida anade la mayor fotovoltaica disponible
#     de cada hora (pv_mayor_disp_MW) para comprobar a posteriori que no supera la mayor perdida;
#   * la salida anade dP_prog_MW (la variable dP del programa) y huella (huella_codigo()), y el
#     fichero lleva el sufijo _nuclibre.
NUCLIBRE = False
# Campana sinnuclear (13/10/2026): la nuclear no esta disponible en todo el ano (--sinnuclear, solo con
# --suceso=mayor): sistema sin unidad sincrona dominante. Usa toda la maquinaria de --nuclibre (las
# demas termicas candidatas por escalon, la fila de la nuclear con su binaria) y ademas fija a cero el
# acoplamiento de la nuclear en todas las horas y en el estado inicial: no produce, no aporta energia
# cinetica y nunca es candidata. Sufijo _sinnuclear (en lugar de _nuclibre).
SINNUCLEAR = False


def cargar(dyn=False, gruesa=False, flota=0):
    g = pd.read_csv(D + 'SourceData/gen.csv')
    th = g[g['Unit Type'].isin(['CC', 'CT', 'STEAM', 'NUCLEAR'])].copy()
    # energia cinetica por unidad, igual que uc_mof1.cargar
    if dyn:
        # Energia por unidad leida de RTS.par (H de cada grupo y S de cada maquina). Nota del
        # 12/10/2026: la columna es por unidad, pero la agregacion de abajo tomaba en v4 la de la
        # primera unidad de cada equivalente (ver ESUMA); dentro de U55 la 315_CT_8 tiene mayor
        # potencia aparente que las demas.
        g['E'] = U.energia_dinamica(g, dyn)
        th['E'] = g.loc[th.index, 'E']
        e_hidro = g[g['Unit Type'] == 'HYDRO']['E'].iloc[0]
        e_sc = g[g['Unit Type'] == 'SYNC_COND']['E'].sum()
    else:
        th['E'] = th['Inertia MJ/MW'] * th['Base MVA']
        e_hidro, e_sc = 3.5 * 53.0, 0.0
    # v5: energia con el conjunto corregido, solo para la comprobacion a posteriori (no entra en el
    # programa), de modo que la tasa ante el peor candidato se evalua igual en todos los casos
    g['Ecorr'] = U.energia_dinamica(g, 'corr')
    th['Ecorr'] = g.loc[th.index, 'Ecorr']
    e_hidro_corr = g[g['Unit Type'] == 'HYDRO']['Ecorr'].iloc[0]
    e_sc_corr = g[g['Unit Type'] == 'SYNC_COND']['Ecorr'].sum()
    num = ['Fuel Price $/MMBTU', 'HR_avg_0', 'HR_incr_1', 'HR_incr_2', 'HR_incr_3']
    th[num] = th.groupby('Unit Group')[num].transform('mean')
    tecn = ['Unit Group', 'PMax MW', 'PMin MW', 'Min Up Time Hr', 'Min Down Time Hr', 'Ramp Rate MW/Min',
            'Output_pct_1', 'Output_pct_2', 'Start Heat Hot MBTU', 'Non Fuel Start Cost $', 'E'] + num
    # Agrupacion termica: por zona (area del banco) y grupo de unidad; la inyeccion de cada
    # equivalente se reparte entre sus nudos en proporcion a la potencia instalada en cada uno.
    area = pd.read_csv(D + 'SourceData/bus.csv').set_index('Bus ID')['Area']
    th['Area'] = th['Bus ID'].map(area)
    clave = ['Unit Group'] if gruesa else (['Bus ID', 'Unit Group'] if POR_NUDO else ['Area', 'Unit Group'])
    filas = []
    for k, dd in th.groupby(clave):
        r = dd.iloc[0][tecn].to_dict()
        if ESUMA:
            # v5: energia media del equivalente, N * e = suma de la de cada unidad (ver ESUMA)
            r['E'] = float(dd['E'].mean())
            r['Ecorr'] = float(dd['Ecorr'].mean())
        else:
            r['Ecorr'] = float(dd.iloc[0]['Ecorr'])
        N = len(dd)
        reparto = (dd.groupby('Bus ID')['PMax MW'].sum() / dd['PMax MW'].sum()).to_dict()
        f = 1.0
        if flota and not gruesa:
            # Campana v4: flota reagrupada a un tamano de unidad objetivo FLOTA MW dentro de cada
            # equivalente de zona y grupo, de modo que la agregacion espacial es la misma que la de la
            # flota publicada y solo cambia el tamano de la unidad: k = min(N, floor(FLOTA/Pmax)) y
            # round(N/k) unidades, con potencia, minimo tecnico, energia cinetica, rampa y coste de
            # arranque totales conservados. La nuclear (una unidad) no cambia.
            kk = int(min(N, max(1, np.floor(float(flota) / r['PMax MW']))))
            nuevo = max(int(round(N / kk)), 1)
            f = N / nuevo
            N = nuevo
        if gruesa:
            kk = int(min(N, max(1, np.floor(355.0 / r['PMax MW']))))
            nuevo = max(int(round(N / kk)), 1)
            f = N / nuevo
            N = nuevo
        filas.append(dict(r, N=N, f=f, reparto=reparto))
    cl = pd.DataFrame(filas)
    for k in ['PMax MW', 'PMin MW', 'Ramp Rate MW/Min', 'E', 'Ecorr', 'Start Heat Hot MBTU', 'Non Fuel Start Cost $']:
        cl[k] = cl[k] * cl['f']
    pr = cl['Fuel Price $/MMBTU']
    p1 = cl['Output_pct_1'] * cl['PMax MW']
    p2 = cl['Output_pct_2'] * cl['PMax MW']
    c = pd.DataFrame({
        'grupo': cl['Unit Group'], 'N': cl['N'].astype(int), 'Pmax': cl['PMax MW'], 'Pmin': cl['PMin MW'],
        'UT': np.ceil(cl['Min Up Time Hr']).astype(int), 'DT': np.ceil(cl['Min Down Time Hr']).astype(int),
        'RU': cl['Ramp Rate MW/Min'] * 60.0, 'e': cl['E'],
        'c0': pr * cl['HR_avg_0'] * cl['PMin MW'] / 1000.0,
        'w1': np.maximum(p1 - cl['PMin MW'], 0), 'w2': np.maximum(p2 - p1, 0), 'w3': np.maximum(cl['PMax MW'] - p2, 0),
        'm1': pr * cl['HR_incr_1'] / 1000.0, 'm2': pr * cl['HR_incr_2'] / 1000.0, 'm3': pr * cl['HR_incr_3'] / 1000.0,
        'cs': pr * cl['Start Heat Hot MBTU'] + cl['Non Fuel Start Cost $'],
        'reparto': cl['reparto'], 'e_corr': cl['Ecorr'],
    })
    c['grande'] = c['Pmax'] >= 155
    if NUCLIBRE:
        # coste de arranque de la nuclear con el calor de arranque en frio del banco (ver NUCLIBRE)
        gn = th[th['Unit Group'] == 'U400'].iloc[0]
        c.loc[c.grupo == 'U400', 'cs'] = float(gn['Fuel Price $/MMBTU'] * gn['Start Heat Cold MBTU'] + gn['Non Fuel Start Cost $'])
    # ---- red
    bus = pd.read_csv(D + 'SourceData/bus.csv')
    br = pd.read_csv(D + 'SourceData/branch.csv')
    nb = bus['Bus ID'].tolist()
    ib = {b: i for i, b in enumerate(nb)}
    A = np.zeros((len(br), len(nb)))
    for l, r in br.iterrows():
        A[l, ib[r['From Bus']]] = 1.0
        A[l, ib[r['To Bus']]] = -1.0
    bx = 1.0 / br['X'].values
    B = A.T @ (bx[:, None] * A)
    s = ib[113]
    keep = [i for i in range(len(nb)) if i != s]
    Binv = np.zeros((len(nb), len(nb)))
    Binv[np.ix_(keep, keep)] = np.linalg.inv(B[np.ix_(keep, keep)])
    PTDF = (bx[:, None] * A) @ Binv
    rating = br['Cont Rating'].values.astype(float)
    # ---- series nodales
    regional = serie('Load/DAY_AHEAD_regional_Load.csv')
    area_tot = bus.groupby('Area')['MW Load'].sum()
    Lb = np.zeros((len(regional), len(nb)))
    for i, r in bus.iterrows():
        Lb[:, i] = regional[str(int(r['Area']))].values * r['MW Load'] / area_tot[r['Area']]
    vre_cols = {}
    for f, tipo in (('WIND/DAY_AHEAD_wind.csv', 'WIND'), ('PV/DAY_AHEAD_pv.csv', 'PV'), ('RTPV/DAY_AHEAD_rtpv.csv', 'RTPV')):
        dfv = serie(f)
        for uid in dfv.columns:
            b = int(uid.split('_')[0])
            vre_cols.setdefault(b, 0)
            vre_cols[b] = vre_cols[b] + dfv[uid].values
    vre_bus = sorted(vre_cols)
    Vb = np.column_stack([vre_cols[b] for b in vre_bus])
    hyd = serie('Hydro/DAY_AHEAD_hydro.csv')
    infl = serie('Hydro/DAY_AHEAD_hydro_inflow.csv')
    sto = pd.read_csv(D + 'SourceData/storage.csv')
    hb = {}
    for uid in hyd.columns:
        b = int(uid.split('_')[0])
        col = uid + '_RESERVOIR_head'
        emb = sto[sto['GEN UID'] == uid]
        h = hb.setdefault(b, {'n': 0, 'inflow': 0, 'vmax': 0.0, 'v0': 0.0})
        h['n'] += 1
        if col in infl.columns and len(emb):
            h['inflow'] = h['inflow'] + infl[col].values
            h['vmax'] += emb['Max Volume GWh'].iloc[0] * 1000.0
            h['v0'] += emb['Initial Volume GWh'].iloc[0] * 1000.0
        else:
            h['inflow'] = h['inflow'] + hyd[uid].values
    hyd_bus = sorted(hb)
    # v5: plantas eolicas como sucesos candidatos. Cada una esta sola en su nudo entre las
    # renovables variables, de modo que la variable de renovable servida de ese nudo es su produccion.
    uids_por_nudo = {}
    for f in ('WIND/DAY_AHEAD_wind.csv', 'PV/DAY_AHEAD_pv.csv', 'RTPV/DAY_AHEAD_rtpv.csv'):
        for uid in serie(f).columns:
            uids_por_nudo.setdefault(int(uid.split('_')[0]), []).append(uid)
    eolicas = []
    for uid in serie('WIND/DAY_AHEAD_wind.csv').columns:
        b = int(uid.split('_')[0])
        assert uids_por_nudo[b] == [uid], ('el nudo de %s tiene otras renovables' % uid, uids_por_nudo[b])
        eolicas.append((vre_bus.index(b), uid))
    pv_mayor = serie('PV/DAY_AHEAD_pv.csv').max(axis=1).values
    res = sum(serie('Reserves/DAY_AHEAD_regional_%s.csv' % k).sum(axis=1).values for k in ('Spin_Up_R1', 'Spin_Up_R2', 'Spin_Up_R3'))
    rg = pd.read_csv(D + 'timeseries_data_files/Reserves/DAY_AHEAD_regional_Reg_Up.csv')
    res = res + rg.drop(columns=['Year', 'Month', 'Day']).values.reshape(-1)
    return c, dict(nb=nb, ib=ib, PTDF=PTDF, rating=rating, br=br['UID'].tolist(), Lb=Lb, Lsys=Lb.sum(1),
                   vre_bus=vre_bus, Vb=Vb, hb=hb, eolicas=eolicas, hyd_bus=hyd_bus, res=res, e_hidro=e_hidro, e_sc=e_sc,
                   e_hidro_corr=e_hidro_corr, e_sc_corr=e_sc_corr,
                   dc=(ib[113], ib[316], 100.0), sto_bus=ib[313], pv_mayor=pv_mayor)


def niveles_conjunto8(dyn):
    """Niveles de sinc8 y ekin8: potencia y energia cinetica del conjunto minimo con ocho unidades
    grandes en la flota publicada (la nuclear y las siete unidades de 155 MW), con el conjunto de
    inercia dyn. La energia incluye la de los condensadores sincronos, siempre acoplados, porque la
    fila de ekin8 los cuenta en el lado izquierdo."""
    c, dat = cargar(dyn, False, 0)
    nuc = c[c.grupo == 'U400']
    u155 = c[c.grupo == 'U155']
    n155 = int(u155.N.sum())
    assert n155 == 7 and len(nuc) == 1 and int(nuc.N.iloc[0]) == 1
    P = float(nuc.Pmax.iloc[0] + 7 * u155.Pmax.iloc[0])
    E = float(nuc.e.iloc[0] + 7 * u155.e.iloc[0] + dat['e_sc'])
    return P, E


def nivel_mayor_termica(dyn='corr', Rr=1.0):
    """Campana sinnuclear: suelo fijo de energia cinetica calibrado sobre la mayor contingencia sincrona
    sin la nuclear, como ekin13781 lo esta sobre la nuclear: max_k (e_k + f0 Pmax_k / (2 R)) entre las
    demas termicas de la flota publicada, con la energia media de cada equivalente (ESUMA) y el conjunto
    dyn, redondeado hacia arriba al MW s. Devuelve (nombre del caso 'ekinNNNNN', valor exacto, grupo)."""
    import math
    global ESUMA
    viejo = ESUMA
    ESUMA = True
    try:
        c, _ = cargar(dyn, False, 0)
    finally:
        ESUMA = viejo
    o = c[c.grupo != 'U400']
    v = o.e + F0 * o.Pmax / (2 * Rr)
    k = v.idxmax()
    return 'ekin%d' % math.ceil(float(v[k]) - 1e-9), float(v[k]), c.grupo[k]


def dia(c, dat, cfg, d0, estado, vigiladas):
    T_ = TH
    nT = len(dat['Lsys'])
    h = np.minimum(np.arange(d0 * 24, d0 * 24 + T_), nT - 1)
    Lb, Ls, Vb, R = dat['Lb'][h], dat['Lsys'][h], dat['Vb'][h], dat['res'][h]
    PT = dat['PTDF']
    nC = len(c)
    m = Modelo()
    nuc = [k for k in range(nC) if c.grupo[k] == 'U400']
    n = [m.var(T_, c.N[k] if (k in nuc and not NUCLIBRE) else 0, 0 if (k in nuc and SINNUCLEAR) else c.N[k], integer=True)
         for k in range(nC)]
    # Arranques y paradas continuos: con n entero, n_t - n_t-1 = s - d obliga a que el optimo los
    # tome enteros, y relajarlos no abre ninguna ventaja en los tiempos minimos (solo los aprieta).
    s = [m.var(T_, 0, c.N[k], cost=c.cs[k]) for k in range(nC)]
    dd = [m.var(T_, 0, c.N[k]) for k in range(nC)]
    for k in range(nC):
        for t in range(T_):
            m.cost[n[k][t]] = c.c0[k]
    sg = [[m.var(T_, cost=c['m%d' % j][k]) for j in (1, 2, 3)] for k in range(nC)]
    r = [m.var(T_) for k in range(nC)]
    w = [m.var(T_, 0, Vb[:, j], cost=-0.5) for j in range(len(dat['vre_bus']))]
    H = {}
    for b in dat['hyd_bus']:
        hh = dat['hb'][b]
        inflow = hh['inflow'][h]
        nh = m.var(T_, 0, hh['n'])   # hidraulicas acopladas, relajado a continuo
        ph = m.var(T_, 0, 50.0 * hh['n'])
        sp = m.var(T_, 0, np.inf, cost=0.1)       # vertido de agua: penalizacion pequena
        v = m.var(T_, 0, hh['vmax']) if hh['vmax'] > 0 else None
        H[b] = (nh, ph, sp, v, inflow)
    ch = m.var(T_, 0, 50.0); dis = m.var(T_, 0, 50.0); soc = m.var(T_, 0, 150.0)
    # Modo de la bateria (06/10/2026): una binaria por hora impide cargar y descargar a la vez.
    # Sin ella, el premio a la renovable servida permitia quemar excedente en las perdidas de
    # ida y vuelta de la bateria y contarlo como renovable integrada.
    ub_ = m.var(T_, 0, 1, integer=True)
    for t in range(T_):
        m.fila([ch[t], ub_[t]], [1, -50.0], -np.inf, 0)
        m.fila([dis[t], ub_[t]], [1, 50.0], -np.inf, 50.0)
    fdc = m.var(T_, -dat['dc'][2], dat['dc'][2])
    shed = m.var(T_, cost=1e4); rsh = m.var(T_, cost=1e3)
    P = lambda k, t: ([n[k][t], sg[k][0][t], sg[k][1][t], sg[k][2][t]], [c.Pmin[k], 1, 1, 1])
    # balance del sistema
    for t in range(T_):
        idx, cf = [], []
        for k in range(nC):
            i, f = P(k, t); idx += i; cf += f
        for j in range(len(w)):
            idx.append(w[j][t]); cf.append(1)
        for b in H:
            idx.append(H[b][1][t]); cf.append(1)
        idx += [dis[t], ch[t], shed[t]]; cf += [1, -1, 1]
        m.fila(idx, cf, Ls[t], Ls[t])
    for k in range(nC):
        for t in range(T_):
            for j in range(3):
                m.fila([sg[k][j][t], n[k][t]], [1, -c['w%d' % (j + 1)][k]], -np.inf, 0)
            i, f = P(k, t)
            m.fila(i + [r[k][t], n[k][t]], f + [1, -c.Pmax[k]], -np.inf, 0)
            if t == 0:
                m.fila([n[k][0], s[k][0], dd[k][0]], [1, -1, 1], estado['n'][k], estado['n'][k])
            else:
                m.fila([n[k][t], n[k][t - 1], s[k][t], dd[k][t]], [1, -1, -1, 1], 0, 0)
            UT, DT = c.UT[k], c.DT[k]
            hs = estado['hs'][k][HIST - max(UT - 1 - t, 0):] if UT - 1 - t > 0 else []
            m.fila(list(s[k][max(0, t - UT + 1):t + 1]) + [n[k][t]], [1] * (t + 1 - max(0, t - UT + 1)) + [-1], -np.inf, -float(np.sum(hs)))
            hd = estado['hd'][k][HIST - max(DT - 1 - t, 0):] if DT - 1 - t > 0 else []
            m.fila(list(dd[k][max(0, t - DT + 1):t + 1]) + [n[k][t]], [1] * (t + 1 - max(0, t - DT + 1)) + [1], -np.inf, c.N[k] - float(np.sum(hd)))
            if t == 0:
                m.fila(i + [n[k][t], s[k][t]], f + [-c.RU[k], c.RU[k] - c.Pmax[k]], -np.inf, estado['p'][k])
                m.fila(i + [n[k][t], dd[k][t]], f + [c.RU[k], c.Pmax[k]], estado['p'][k], np.inf)
            else:
                i0, f0 = P(k, t - 1)
                m.fila(i + i0 + [n[k][t], s[k][t]], f + [-x for x in f0] + [-c.RU[k], c.RU[k] - c.Pmax[k]], -np.inf, 0)
                m.fila(i + i0 + [n[k][t], dd[k][t]], f + [-x for x in f0] + [c.RU[k], c.Pmax[k]], 0, np.inf)
    # v5, suceso dimensionante endogeno: dP_t >= produccion de cada candidato. dP solo aparece en el
    # lado exigente de la reserva, de modo que en el optimo es el maximo de los candidatos (epigrafe
    # exacto, sin binaria). Candidatos: la nuclear por su produccion programada y cada planta eolica
    # por la suya. Las demas termicas tienen Pmax por debajo del minimo tecnico de la nuclear, siempre
    # acoplada, y no pueden fijar dP (se comprueba con la asercion).
    if SUCESO == 'mayor':
        k_nuc = nuc[0]
        if not NUCLIBRE:
            assert c.Pmax[[k for k in range(nC) if k != k_nuc]].max() <= c.Pmin[k_nuc] + 1e-9
        dP = m.var(T_, 0, np.inf)
        for t in range(T_):
            i, f = P(k_nuc, t)
            m.fila([dP[t]] + i, [1] + [-x for x in f], 0, np.inf)
            for j, _ in dat['eolicas']:
                m.fila([dP[t], w[j][t]], [1, -1], 0, np.inf)
        if NUCLIBRE:
            # nuclibre: las demas termicas como candidatas, por escalon de potencia maxima (ver NUCLIBRE)
            esc = {}
            for k in range(nC):
                if k != k_nuc:
                    esc.setdefault(round(float(c.Pmax[k]), 6), []).append(k)
            z = {L: m.var(T_, 0, 1, integer=True) for L in esc}
            for L, ks in esc.items():
                Ntot = sum(c.N[k] for k in ks)
                for t in range(T_):
                    m.fila([z[L][t]] + [n[k][t] for k in ks], [Ntot] + [-1] * len(ks), 0, np.inf)
                    if L > R[t]:
                        m.fila([dP[t], z[L][t]], [1, -L], 0, np.inf)
    for t in range(T_):
        if RES_N1 and SUCESO == 'mayor':
            # v5: dos filas, el requisito del banco y la mayor perdida de la hora
            idx = list(r[k][t] for k in range(nC)) + [rsh[t]]
            cf = [1] * nC + [1]
            for b in H:
                idx += [H[b][0][t], H[b][1][t]]
                cf += [50.0, -1.0]
            m.fila(idx, cf, R[t], np.inf)
            m.fila(idx + [dP[t]], cf + [-1.0], 0, np.inf)
        elif RES_N1:
            # Reserva rodante con el criterio N-1: el margen de subida de las termicas y de las
            # hidraulicas acopladas cubre al menos la perdida del mayor inyector (la nuclear de
            # 400 MW, siempre acoplada) y nunca menos que el requisito del banco. Sin esto, el
            # requisito del banco (116 a 365 MW) no cubre la perdida contra la que se dimensiona
            # el criterio de RoCoF.
            idx = list(r[k][t] for k in range(nC)) + [rsh[t]]
            cf = [1] * nC + [1]
            for b in H:
                idx += [H[b][0][t], H[b][1][t]]
                cf += [50.0, -1.0]
            m.fila(idx, cf, max(R[t], P_N1), np.inf)
        else:
            m.fila(list(r[k][t] for k in range(nC)) + [rsh[t]], [1] * nC + [1], R[t], np.inf)
        prev = [soc[t - 1]] if t > 0 else []
        m.fila([soc[t]] + prev + [ch[t], dis[t]], [1] + ([-1] if t > 0 else []) + [-0.922, 1 / 0.922],
               estado['soc'] if t == 0 else 0, estado['soc'] if t == 0 else 0)
    m.fila([soc[T_ - 1]], [1], 75.0, np.inf)
    # hidraulica con embalse
    for b, (nh, ph, sp, v, inflow) in H.items():
        nunid = dat['hb'][b]['n']
        for t in range(T_):
            m.fila([ph[t], nh[t]], [1, -50.0], -np.inf, 0)
            if HMIN > 0:
                m.fila([ph[t], nh[t]], [1, -HMIN], 0, np.inf)
            if v is None:   # de pasada: lo que no se turbina se vierte
                m.fila([ph[t], sp[t]], [1, 1], inflow[t], inflow[t])
            else:
                prev = [v[t - 1]] if t > 0 else []
                rhs = inflow[t] + (estado['v'][b] if t == 0 else 0)
                m.fila([v[t]] + prev + [ph[t], sp[t]], [1] + ([-1] if t > 0 else []) + [1, 1], rhs, rhs)
        if v is not None:
            m.fila([v[T_ - 1]], [1], estado['v'][b], np.inf)
    # criterios
    Ew = lambda t: ([n[k][t] for k in range(nC)] + [H[b][0][t] for b in H],
                    [c.e[k] for k in range(nC)] + [dat['e_hidro']] * len(H))
    if 'snsp' in cfg:
        a = cfg['snsp']; sh = m.var(T_, cost=1e4)
        for t in range(T_):
            m.fila([w[j][t] for j in range(len(w))] + [dis[t], ch[t], sh[t]], [1] * len(w) + [1, -a, -1], -np.inf, a * Ls[t])
    if 'muon' in cfg:
        sh = m.var(T_, cost=1e5)
        sel = [k for k in range(nC) if (cfg.get('muon_any') or c.grande[k])]
        for t in range(T_):
            idx = [n[k][t] for k in sel]; cf = [1] * len(sel)
            if cfg.get('muon_any'):
                idx += [H[b][0][t] for b in H]; cf += [1] * len(H)
            m.fila(idx + [sh[t]], cf + [1], cfg['muon'], np.inf)
    if 'sinc_mw' in cfg:
        dat['nivel_P'] = cfg['sinc_mw']
    if 'ekin_mws' in cfg:
        dat['nivel_E'] = cfg['ekin_mws']
    if cfg.get('sinc8') or 'sinc_mw' in cfg:
        # potencia sincrona acoplada: termicas por su Pmax e hidraulicas acopladas por 50 MW
        sh = m.var(T_, cost=1e3)
        for t in range(T_):
            m.fila([n[k][t] for k in range(nC)] + [H[b][0][t] for b in H] + [sh[t]],
                   [c.Pmax[k] for k in range(nC)] + [50.0] * len(H) + [1], dat['nivel_P'], np.inf)
    if cfg.get('ekin8') or 'ekin_mws' in cfg:
        # energia cinetica acoplada fija, independiente de la mayor perdida
        sh = m.var(T_, cost=100.0)
        for t in range(T_):
            Ei, Ec = Ew(t)
            m.fila(Ei + [sh[t]], Ec + [1], dat['nivel_E'] - dat['e_sc'], np.inf)
    if 'rocof' in cfg and SUCESO == 'mayor' and NUCLIBRE:
        # nuclibre: nuclear candidata solo acoplada (n_nuc es su binaria) y una fila por escalon termico
        #   nuclear:     E_t - e_nuc n_nuc,t >= f0 p_nuc,t / (2 R)
        #   eolica j:    E_t                 >= f0 w_j,t / (2 R)
        #   escalon L:   E_t - e_L z_L,t     >= f0 L z_L,t / (2 R)
        Rr = cfg['rocof']; sh = m.var(T_, cost=100.0)
        a_ = F0 / (2 * Rr)
        eL = {L: float(max(c.e[k] for k in ks)) for L, ks in esc.items()}
        for t in range(T_):
            Ei, Ec = Ew(t)
            i, f = P(k_nuc, t)
            m.fila(Ei + i + [n[k_nuc][t], sh[t]], Ec + [-a_ * x for x in f] + [-c.e[k_nuc], 1], -dat['e_sc'], np.inf)
            for j, _ in dat['eolicas']:
                m.fila(Ei + [w[j][t], sh[t]], Ec + [-a_, 1], -dat['e_sc'], np.inf)
            for L in esc:
                if a_ * L > dat['e_sc']:
                    m.fila(Ei + [z[L][t], sh[t]], Ec + [-(eL[L] + a_ * L), 1], -dat['e_sc'], np.inf)
    elif 'rocof' in cfg and SUCESO == 'mayor':
        # v5: una fila por candidato con SU produccion y SU inercia, no una fila con un dP comun:
        # emparejar la mayor perdida (eolica, sin inercia) con la inercia de la nuclear exigiria e_nuc
        # de mas.
        #   nuclear:   E_t - e_nuc >= f0 p_nuc,t / (2 R)          (acoplada siempre, sin binaria)
        #   eolica j:  E_t        >= f0 w_j,t / (2 R)             (lineal, sin binaria)
        #   termica k: E_t - e_k y_k,t >= f0 Pmax_k y_k,t / (2 R) (por equivalente, no por grupo)
        Rr = cfg['rocof']; sh = m.var(T_, cost=100.0)
        a_ = F0 / (2 * Rr)
        # solo los equivalentes cuya fila no queda dominada por la de la nuclear, siempre acoplada
        # (en la flota publicada y la de 155 MW ninguno; en la de 355 MW los U55 fusionados)
        piso_nuc = c.e[k_nuc] + a_ * c.Pmin[k_nuc] * c.N[k_nuc]
        y = {k: m.var(T_, 0, 1, integer=True) for k in range(nC)
             if k != k_nuc and c.e[k] + a_ * c.Pmax[k] > piso_nuc}
        for t in range(T_):
            Ei, Ec = Ew(t)
            i, f = P(k_nuc, t)
            m.fila(Ei + i + [sh[t]], Ec + [-a_ * x for x in f] + [1], c.e[k_nuc] - dat['e_sc'], np.inf)
            for j, _ in dat['eolicas']:
                m.fila(Ei + [w[j][t], sh[t]], Ec + [-a_, 1], -dat['e_sc'], np.inf)
            for k, yk in y.items():
                m.fila([yk[t], n[k][t]], [c.N[k], -1], 0, np.inf)
                m.fila(Ei + [yk[t], sh[t]], Ec + [-(c.e[k] + a_ * c.Pmax[k]), 1], -dat['e_sc'], np.inf)
    elif 'rocof' in cfg:
        # Formulacion de v4 (suceso de la unidad termica mayor de cada grupo acoplado). Fallo latente
        # que se conserva para reproducir v4: la fila de cada grupo usa ks[0], el primer equivalente
        # del grupo, con su e y su Pmax; en flotas reagrupadas el tamano cambia entre zonas.
        Rr = cfg['rocof']; sh = m.var(T_, cost=100.0)
        grupos = sorted(set(c.grupo))
        y = {gname: m.var(T_, 0, 1, integer=True) for gname in grupos}
        for t in range(T_):
            Ei, Ec = Ew(t)
            for gname in grupos:
                ks = [k for k in range(nC) if c.grupo[k] == gname]
                Ntot = sum(c.N[k] for k in ks)
                m.fila([y[gname][t]] + [n[k][t] for k in ks], [Ntot] + [-1] * len(ks), 0, np.inf)
                k0 = ks[0]
                m.fila(Ei + [y[gname][t], sh[t]], Ec + [-(c.e[k0] + F0 * c.Pmax[k0] / (2 * Rr)), 1], -dat['e_sc'], np.inf)
    # inyecciones nodales para los flujos: lista de (indice de variable, nudo, coeficiente)
    iny = []
    for k in range(nC):
        for b, sh_ in c.reparto[k].items():
            iny.append(('th', k, dat['ib'][b], sh_))
    def fila_flujo(l, t):
        coef = {}
        def add(var, val):
            coef[var] = coef.get(var, 0.0) + val
        for _, k, ibus, sh_ in iny:
            i, f = P(k, t)
            for vv, ff in zip(i, f):
                add(vv, PT[l, ibus] * sh_ * ff)
        for j, b in enumerate(dat['vre_bus']):
            add(w[j][t], PT[l, dat['ib'][b]])
        for b in H:
            add(H[b][1][t], PT[l, dat['ib'][b]])
        add(dis[t], PT[l, dat['sto_bus']]); add(ch[t], -PT[l, dat['sto_bus']])
        a_, b_, _ = dat['dc']
        add(fdc[t], PT[l, b_] - PT[l, a_])
        add(shed[t], PT[l, dat['ib'][113]])
        carga = float(PT[l] @ Lb[t])
        idx = [k_ for k_, v_ in coef.items() if abs(v_) > 1e-9]
        return idx, [coef[k_] for k_ in idx], -dat['rating'][l] + carga, dat['rating'][l] + carga
    def flujos(x):
        F = np.zeros((len(dat['rating']), T_))
        for t in range(T_):
            inj = -Lb[t].copy()
            for _, k, ibus, sh_ in iny:
                i, f = P(k, t)
                inj[ibus] += sh_ * sum(x[vv] * ff for vv, ff in zip(i, f))
            for j, b in enumerate(dat['vre_bus']):
                inj[dat['ib'][b]] += x[w[j][t]]
            for b in H:
                inj[dat['ib'][b]] += x[H[b][1][t]]
            inj[dat['sto_bus']] += x[dis[t]] - x[ch[t]]
            a_, b_, _ = dat['dc']
            inj[a_] -= x[fdc[t]]; inj[b_] += x[fdc[t]]
            inj[dat['ib'][113]] += x[shed[t]]
            F[:, t] = PT @ inj
        return F
    for l in sorted(vigiladas):
        for t in range(T_):
            i, f, lo, hi = fila_flujo(l, t); m.fila(i, f, lo, hi)
    t1 = time.time()
    nuevas_total = 0
    for it in range(MAX_ITER):
        res = m.resolver(GAP, TLIM)
        if res.x is None:
            raise RuntimeError('dia %d sin solucion: %s' % (d0, res.message))
        F = flujos(res.x)
        viol = np.abs(F) > dat['rating'][:, None] * 1.001
        nuevas = sorted(set(np.where(viol.any(1))[0]) - vigiladas)
        if not nuevas:
            break
        nuevas_total += len(nuevas)
        for l in nuevas:
            vigiladas.add(l)
            for t in range(T_):
                i, f, lo, hi = fila_flujo(l, t); m.fila(i, f, lo, hi)
    # Pulido: con el acoplamiento fijado, el despacho se resuelve como programa lineal exacto, que es
    # optimo para ese acoplamiento; el acoplamiento conserva el hueco del programa entero.
    xm = res.x
    integ = np.array(m.integ)
    lb0, ub0 = list(m.lb), list(m.ub)
    for i in np.where(integ == 1)[0]:
        m.lb[i] = m.ub[i] = round(xm[i])
    m.integ = [0] * len(m.integ)
    res2 = m.resolver(1e-9, TLIM)
    gap_mip = getattr(res, 'mip_gap', np.nan)
    if res2.x is not None:
        res = res2
        # 13/10/2026: el repaso lineal puede mover el despacho y pasar de su limite una rama no vigilada
        # (ocurrio en la prueba sin nuclear, dias 206-207). Se comprueban solo las 24 horas aplicadas, las
        # mismas que la salida registra en ramas_sobre_limite_final, que vale 0 en todas las horas de todos
        # los casos de v5: en v5 esto no actua y no cambia sus resultados. Se anaden esas ramas y se repite.
        for _ in range(MAX_ITER):
            F2 = flujos(res.x)
            nuevas2 = sorted(set(np.where((np.abs(F2[:, :24]) > dat['rating'][:, None] * 1.001).any(1))[0]) - vigiladas)
            if not nuevas2:
                break
            nuevas_total += len(nuevas2)
            for l in nuevas2:
                vigiladas.add(l)
                for t in range(T_):
                    i, f, lo, hi = fila_flujo(l, t); m.fila(i, f, lo, hi)
            res3 = m.resolver(1e-9, TLIM)
            if res3.x is None:
                break
            res = res3
    x = res.x
    A_ = 24
    nv = np.array([[x[n[k][t]] for t in range(T_)] for k in range(nC)]).round()
    pv = np.array([[c.Pmin[k] * nv[k, t] + sum(x[sg[k][j][t]] for j in range(3)) for t in range(T_)] for k in range(nC)])
    sv = np.array([[x[s[k][t]] for t in range(A_)] for k in range(nC)]).round()
    dv = np.array([[x[dd[k][t]] for t in range(A_)] for k in range(nC)]).round()
    nuevo = {'n': nv[:, A_ - 1], 'p': pv[:, A_ - 1], 'soc': x[soc[A_ - 1]],
             'hs': [np.concatenate([estado['hs'][k], sv[k]])[-HIST:] for k in range(nC)],
             'hd': [np.concatenate([estado['hd'][k], dv[k]])[-HIST:] for k in range(nC)],
             'v': {b: (x[H[b][3][A_ - 1]] if H[b][3] is not None else 0.0) for b in H}}
    xs = lambda v: x[v][:A_]
    nv, pv = nv[:, :A_], pv[:, :A_]
    nh_tot = sum(xs(H[b][0]) for b in H)
    E = (c.e.values[:, None] * nv).sum(0) + dat['e_hidro'] * nh_tot + dat['e_sc']
    wv = sum(xs(w[j]) for j in range(len(w)))
    V = Vb[:A_].sum(1)
    hid = sum(xs(H[b][1]) for b in H)
    hid_disp = sum(H[b][4][:A_] for b in H)
    vert_h = sum(xs(H[b][2]) for b in H)
    coste = (c.c0.values[:, None] * nv).sum() + (c.cs.values[:, None] * sv).sum() + sum(
        c['m%d' % (j + 1)][k] * x[sg[k][j][:A_]].sum() for k in range(nC) for j in range(3))
    carga = np.abs(F[:, :A_]) / dat['rating'][:, None]
    # flujos de la solucion final, tras el repaso lineal: las ramas vigiladas estan en el modelo,
    # pero una rama no vigilada podria pasarse de su limite al recalcular el despacho
    carga_fin = np.abs(flujos(x)[:, :A_]) / dat['rating'][:, None]
    out = pd.DataFrame({
        'hora': np.arange(d0 * 24, d0 * 24 + A_), 'demanda': Ls[:A_], 'vre_disp': V, 'vre': wv,
        'recorte': V - wv, 'hidro_disp': hid_disp, 'hidro': hid, 'hidro_vertida': vert_h,
        'embalse_MWh': sum((xs(H[b][3]) if H[b][3] is not None else 0) for b in H),
        'carga_alm': xs(ch), 'desc_alm': xs(dis), 'termica': pv.sum(0),
        'nuclear_on': nv[c.grupo.values == 'U400'].sum(0), 'unid_on': nv.sum(0),
        'grandes_on': nv[c.grande.values].sum(0), 'hidro_on': nh_tot, 'E_MWs': E,
        'sinc_MW': (c.Pmax.values[:, None] * nv).sum(0) + 50.0 * nh_tot,
        'snsp': (wv + xs(dis)) / (Ls[:A_] + xs(ch)), 'deslastre': xs(shed), 'reserva_falta': xs(rsh),
        'ramas_al_limite': (carga > 0.999).sum(0), 'carga_max_rama': carga.max(0),
        'carga_max_final': carga_fin.max(0), 'ramas_sobre_limite_final': (carga_fin > 1.001).sum(0),
        'coste_dia': coste, 'gap': gap_mip, 'seg': time.time() - t1,
        'iter_red': it + 1, 'ramas_nuevas': nuevas_total,
    })
    # unidades acopladas y produccion por grupo, para la comprobacion dinamica a posteriori
    for gr in sorted(set(c.grupo)):
        sel = c.grupo.values == gr
        out['n_' + gr] = nv[sel].sum(0)
        out['p_' + gr] = pv[sel].sum(0)
    # v5: produccion de cada planta eolica, mayor perdida de la hora y tasa ante el peor candidato,
    # todo con la energia del conjunto corregido (para --dyn coincide con E_MWs). Candidatos: la
    # nuclear por su produccion, cada eolica por la suya, cada equivalente termico acoplado por su
    # Pmax con su energia. Diagnostico: no entra en el programa.
    k_n = [k for k in range(nC) if c.grupo[k] == 'U400'][0]
    Ecorr = (c.e_corr.values[:, None] * nv).sum(0) + dat['e_hidro_corr'] * nh_tot + dat['e_sc_corr']
    out['E_corr_MWs'] = Ecorr
    rf = {'U400': F0 * pv[k_n] / (2 * (Ecorr - c.e_corr[k_n] * nv[k_n]))}
    for j, uid in dat['eolicas']:
        out['w_' + uid] = xs(w[j])
        out['wd_' + uid] = Vb[:A_, j]
        rf[uid] = F0 * xs(w[j]) / (2 * Ecorr)
    for k in range(nC):
        if k != k_n:
            rf['%s_%d' % (c.grupo[k], k)] = np.where(nv[k] > 0.5, F0 * c.Pmax[k] / (2 * (Ecorr - c.e_corr[k])), 0.0)
    if NUCLIBRE:
        # nuclibre: las demas termicas acopladas son candidatas por su Pmax; la nuclear, solo acoplada
        out['dP_MW'] = np.maximum.reduce([pv[k_n]] + [xs(w[j]) for j, _ in dat['eolicas']]
                                         + [np.where(nv[k] > 0.5, c.Pmax[k], 0.0) for k in range(nC) if k != k_n])
        out['dP_prog_MW'] = xs(dP)
        out['pv_mayor_disp_MW'] = dat['pv_mayor'][h][:A_]
        out['huella'] = huella_codigo()
    else:
        out['dP_MW'] = np.maximum.reduce([pv[k_n]] + [xs(w[j]) for j, _ in dat['eolicas']])
    RF = pd.DataFrame(rf)
    out['rocof_peor'] = RF.max(axis=1).values
    out['suceso_peor'] = RF.idxmax(axis=1).values
    return out, nuevo


def huella_codigo():
    """md5 de uc_mof1_red.py y de uc_mof1.py, los dos ficheros que definen el programa."""
    import hashlib
    h = []
    for f in (os.path.abspath(__file__), os.path.abspath(U.__file__)):
        with open(f, 'rb') as fh:
            h.append(hashlib.md5(fh.read()).hexdigest()[:12])
    return '-'.join(h)


def main():
    global HMIN
    nombre = sys.argv[1]
    dyn = 'sabot' if '--dynsabot' in sys.argv else ('corr' if '--dyn' in sys.argv else False)
    gruesa = '--gruesa' in sys.argv
    flota = 0
    for x in sys.argv:
        if x.startswith('--flota='):
            flota = int(x.split('=')[1])
    global RES_N1, POR_NUDO, SUCESO, ESUMA, NUCLIBRE, SINNUCLEAR
    if '--suceso=mayor' in sys.argv:
        SUCESO = 'mayor'
    elif any(x.startswith('--suceso=') and x != '--suceso=nuclear' for x in sys.argv):
        raise SystemExit('suceso desconocido: use --suceso=mayor o --suceso=nuclear')
    # la agregacion corregida de la energia cinetica va con el suceso nuevo, salvo que se fuerce
    ESUMA = SUCESO == 'mayor'
    if '--esuma' in sys.argv:
        ESUMA = True
    if '--sinesuma' in sys.argv:
        ESUMA = False
    if '--nuclibre' in sys.argv or '--sinnuclear' in sys.argv:
        if SUCESO != 'mayor' or '--sinN1' in sys.argv:
            raise SystemExit('--nuclibre y --sinnuclear solo con --suceso=mayor y la reserva N-1')
        if '--nuclibre' in sys.argv and '--sinnuclear' in sys.argv:
            raise SystemExit('--nuclibre y --sinnuclear son excluyentes')
        NUCLIBRE = True
        SINNUCLEAR = '--sinnuclear' in sys.argv
    if '--sinN1' in sys.argv:
        RES_N1 = False
    if '--pornudo' in sys.argv:
        POR_NUDO = True
    if '--hmin' in sys.argv:
        HMIN = 15.0
    global TLIM, GAP
    for x in sys.argv:
        if x.startswith('--tlim='):
            TLIM = float(x.split('=')[1])
        if x.startswith('--gap='):
            # Comprobacion v4c (09/10/2026): hueco relativo mas estrecho para los casos clave
            GAP = float(x.split('=')[1])
    args = [a for a in sys.argv[2:] if not a.startswith('--')]
    a, b = (int(args[0]), int(args[1])) if len(args) >= 2 else (0, 366)
    c, dat = cargar(dyn, gruesa, flota)
    cfg = U.config(nombre)
    print('caso %s: suceso %s, energia de los equivalentes %s' % (nombre, SUCESO, 'media de sus unidades' if ESUMA else 'primera unidad (v4)'), flush=True)
    print('huella del codigo %s' % huella_codigo(), flush=True)
    if SINNUCLEAR:
        print('sin nuclear: la nuclear no esta disponible en ninguna hora', flush=True)
    elif NUCLIBRE:
        kn = c.index[c.grupo == 'U400'][0]
        print('nuclear libre: minimo tecnico %.0f MW, %d h de funcionamiento y %d h de parada minimos, rampa %.0f MW/h, arranque %.0f $'
              % (c.Pmin[kn], c.UT[kn], c.DT[kn], c.RU[kn], c.cs[kn]), flush=True)
    if cfg.get('sinc8') or cfg.get('ekin8'):
        dat['nivel_P'], dat['nivel_E'] = niveles_conjunto8(dyn)
        print('niveles del conjunto de ocho: %.1f MW, %.1f MW s' % (dat['nivel_P'], dat['nivel_E']), flush=True)
    estado = {'n': c.N.values.astype(float), 'p': (c.Pmin * c.N).values, 'soc': 75.0,
              'hs': [np.zeros(HIST) for _ in range(len(c))], 'hd': [np.zeros(HIST) for _ in range(len(c))],
              'v': {bb: dat['hb'][bb]['v0'] for bb in dat['hyd_bus']}}
    if SINNUCLEAR:
        # la nuclear empieza el ano desacoplada y sin produccion
        sel = (c.grupo == 'U400').values
        estado['n'] = np.where(sel, 0.0, estado['n'])
        estado['p'] = np.where(sel, 0.0, np.asarray(estado['p'], dtype=float))
    # Semilla de ramas vigiladas: las que ya ataron en corridas anteriores (solo acelera; las que
    # falten se anaden igual por el bucle perezoso).
    vigiladas = set()
    if os.path.exists('ramas_semilla.txt'):
        nombres = open('ramas_semilla.txt').read().split()
        vigiladas = {dat['br'].index(x) for x in nombres if x in dat['br']}
    os.makedirs('resultados_red', exist_ok=True)
    partes = []
    dias = list(range(a, b))
    suf = {False: '', 'corr': '_dyn', 'sabot': '_dynsabot'}[dyn] + ('_gruesa' if gruesa else '') + ('_f%d' % flota if flota and not gruesa else '') + ('_hmin' if HMIN > 0 else '') + ('_gap%g' % (100 * GAP) if GAP != 5e-3 else '') + ('_mp' if SUCESO == 'mayor' else '') + ('_esuma' if ESUMA and SUCESO != 'mayor' else '') + ('_e1' if not ESUMA and SUCESO == 'mayor' else '') + ('_sinN1' if not RES_N1 else '') + ('_nudo' if POR_NUDO else '') + ('_sem' if '--semanas' in sys.argv else '') + ('_sinnuclear' if SINNUCLEAR else ('_nuclibre' if NUCLIBRE else ''))
    rango = '' if (a, b) == (0, 366) else '_%d_%d' % (a, b)
    salida = 'resultados_red/%s%s%s.csv' % (nombre, suf, rango)
    # Punto de control: cada 15 dias se guardan las horas resueltas y el estado, para poder
    # reanudar un caso interrumpido con la misma orden sin cambiar el resultado. El punto de control
    # lleva la huella del codigo (md5 de uc_mof1_red.py y de uc_mof1.py) y no se reanuda uno escrito
    # con otra version, que mezclaria dias resueltos con dos programas distintos.
    import pickle
    huella = huella_codigo()
    ckpt = salida + '.ckpt'
    hechos = set()
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as fh:
            guardado = pickle.load(fh)
        if len(guardado) != 5 or guardado[4] != huella:
            raise SystemExit(
                'El punto de control %s se escribio con otra version del codigo (huella %s; la actual es %s).\n'
                'No se reanuda para no mezclar dias resueltos con dos programas distintos. Apartelo con el\n'
                'convenio de copias (p. ej. %s-backupAAAAMMDD-HHMM.ckpt) y vuelva a lanzar el caso, que empezara\n'
                'desde el dia inicial.' % (ckpt, guardado[4] if len(guardado) == 5 else 'sin huella', huella,
                                           os.path.splitext(ckpt)[0]))
        partes, estado, vigiladas, hechos = guardado[:4]
        print('reanudado con %d dias hechos' % len(hechos), flush=True)
    if '--semanas' in sys.argv:
        # Submuestra: semanas alternas (la 1.a, 3.a, 5.a...), con el estado arrastrado de un bloque al
        # siguiente. Las cifras anuales se obtienen escalando por 366/dias resueltos.
        dias = [d for d in dias if (d // 7) % 2 == 0]
    for d0 in dias:
        if d0 in hechos:
            continue
        o, estado = dia(c, dat, cfg, d0, estado, vigiladas)
        partes.append(o)
        hechos.add(d0)
        if d0 % 30 == 0:
            print(nombre, d0, '%.1fs' % o.seg.iloc[0], 'ramas vigiladas', len(vigiladas), flush=True)
        if len(hechos) % 15 == 0:
            with open(ckpt + '.tmp', 'wb') as fh:
                pickle.dump((partes, estado, vigiladas, hechos, huella), fh)
            os.replace(ckpt + '.tmp', ckpt)
    pd.concat(partes).to_csv(salida, index=False)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print('vigiladas', sorted(dat['br'][l] for l in vigiladas))
    print('escrito', salida, flush=True)


if __name__ == '__main__':
    main()
