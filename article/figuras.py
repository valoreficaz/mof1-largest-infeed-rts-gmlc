#!/usr/bin/env python3
"""Figuras del articulo «Curtailed for security» (The Electricity Journal), desde los ficheros horarios.

Uso, desde la carpeta article/:   MOF1_RESULTADOS=../results python3 figuras.py
Escribe figs/fig1_reglas.pdf, figs/fig2_reparto.pdf y figs/fig3_datos.pdf. Lee los mismos
ficheros que cifras.py (results/, RTS-GMLC/, inertia_data/).
Las figuras se dibujan al ancho de texto de elsarticle preprint (390 pt = 5,4 in), de modo que
el tamano de letra del fichero es el impreso (7,5 pt como minimo).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


FIGS = os.environ.get('MOF1_FIGS', os.path.join(SALIDA, 'figs'))
os.makedirs(FIGS, exist_ok=True)
TOL = 1e-6
ANCHO = 390 / 72.27
GRANDES = ['303_WIND_1', '317_WIND_1', '122_WIND_1']
EOLICAS = GRANDES + ['309_WIND_1']
# paleta categorica validada (orden fijo): azul, naranja, aguamarina, amarillo, magenta
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
TINTA, TINTA2, REJILLA = '#0b0b0b', '#52514e', '#d9d8d4'
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5,
                     'axes.edgecolor': TINTA2, 'axes.labelcolor': TINTA, 'xtick.color': TINTA2,
                     'ytick.color': TINTA2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})


def leer(n):
    return pd.read_csv(ruta_csv(os.path.join(RR, n + '.csv')))


def coste(d):
    return d.coste_dia.groupby(d.hora // 24).first().sum() / 1e6


def horas(d):
    return 100 * (d.rocof_peor > 1 + TOL).mean()


def rejilla(ax, eje='y'):
    ax.grid(axis=eje, color=REJILLA, lw=0.6)
    ax.set_axisbelow(True)


CONF = [('_mp', '(a) With the 400 MW nuclear unit', 'ekin13781'),
        ('_mp_sinnuclear', '(b) Without the nuclear unit', 'ekin11661')]
# una entrada por regla, con marcador y color propios (la misma en los dos paneles)
REGLAS = [('base', 'No rule', 'o', TINTA2),
          ('muon8', 'Unit count', 's', C[0]),
          ('snsp75', 'SNSP cap', 'D', C[2]),
          ('nuc', 'Fixed floor, synchronous loss', '^', C[3]),
          ('rocof1', 'Indexed RoCoF limit', 'P', C[1]),
          ('ekin25410', 'Fixed floor, largest plant', 'v', C[4])]

# ------------------------------------------------------------------ figura 1
fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.75), sharey=True)
for ax, (suf, tit, nucr) in zip(axs, CONF):
    b = leer('base_dyn' + suf)
    cb = coste(b)
    ins = ax.inset_axes([0.30, 0.38, 0.40, 0.42])
    for r, et, mk, col in REGLAS:
        caso = nucr if r == 'nuc' else r
        d = b if r == 'base' else leer(caso + '_dyn' + suf)
        x, y = coste(d) - cb, horas(d)
        for a in (ax, ins):
            a.scatter([x], [y], s=26, marker=mk, color=col, edgecolor='white', linewidth=0.8, zorder=3, label=et)
    ins.set_xlim(-1.0, 4.0)
    ins.set_ylim(27.5, 32.0)
    ins.tick_params(labelsize=7)
    ins.grid(color=REJILLA, lw=0.5)
    ins.set_axisbelow(True)
    for sp in ins.spines.values():
        sp.set_color(TINTA2)
        sp.set_linewidth(0.6)
    ax.indicate_inset_zoom(ins, edgecolor=TINTA2, alpha=0.8)
    ax.set_title(tit, color=TINTA)
    ax.set_xlabel('Additional operating cost (M\\$/year)')
    ax.set_xlim(-5, 70)
    ax.set_ylim(-2, 36)
    rejilla(ax, 'both')
axs[0].set_ylabel('Hours with RoCoF > 1 Hz/s (%)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0), handletextpad=0.3, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(os.path.join(FIGS, 'fig1_reglas.pdf'), bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ figura 2
gen = pd.read_csv(os.path.join(RTS, 'RTS_Data', 'SourceData', 'gen.csv')).set_index('GEN UID')
cats = ['%s\n(%.0f)' % (p.split('_')[0], gen.loc[p, 'PMax MW']) for p in EOLICAS] + ['Solar']
fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.6), sharey=True)
for ax, (suf, tit, _) in zip(axs, CONF):
    b = leer('base_dyn' + suf)
    for i, (r, et, col) in enumerate([('rocof1', 'Indexed RoCoF limit', C[1]),
                                      ('ekin25410', 'Fixed floor, largest plant', C[4])]):
        d = leer(r + '_dyn' + suf)
        per = [((d['wd_' + p] - d['w_' + p]).sum() - (b['wd_' + p] - b['w_' + p]).sum()) / 1e3 for p in EOLICAS]
        tot = (d.recorte.sum() - b.recorte.sum()) / 1e3
        per.append(tot - sum(per))
        xs = np.arange(len(cats)) + (i - 0.5) * 0.38
        ax.bar(xs, per, width=0.36, color=col, edgecolor='white', linewidth=0.8, label=et, zorder=3)
    ax.axhline(0, color=TINTA2, lw=0.6, zorder=2)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats)
    ax.set_title(tit, color=TINTA)
    rejilla(ax)
    ax.set_xlabel('Wind plant (rated MW) and solar plants')
axs[0].set_ylabel('Additional curtailment (GWh/year)')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.tight_layout(rect=(0, 0.07, 1, 1))
fig.savefig(os.path.join(FIGS, 'fig2_reparto.pdf'), bbox_inches='tight')
plt.close(fig)

# ------------------------------------------------------------------ figura 3
par_ = [os.path.join(RAIZ, 'inertia_data', 'tabla_RTS_par_publicado.csv'),
        os.path.join(RAIZ, 'demostracion_inercia', 'salida', 'tabla_RTS_par_publicado.csv')]
par = pd.read_csv([p for p in par_ if os.path.exists(p)][0])
H_pub = par.groupby('grupo').generator_H.mean().to_dict()
H_int = dict(H_pub, U350=1006.5 / 384.0, U355=1006.5 / 384.0, U400=1990.0 / 500.0)
grupos = ['U155', 'U350', 'U355', 'U400']
S = {gp: float(np.hypot(gen[gen['Unit Group'] == gp]['PMax MW'], gen[gen['Unit Group'] == gp]['QMax MVAR']).iloc[0]) for gp in grupos}
fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.5), gridspec_kw=dict(width_ratios=[1, 1.3]))
ax = axs[0]
xs = np.arange(len(grupos))
ax.bar(xs - 0.19, [H_pub[gp] * S[gp] for gp in grupos], 0.36, color=C[0], edgecolor='white', label='Published', zorder=3)
ax.bar(xs + 0.19, [H_int[gp] * S[gp] for gp in grupos], 0.36, color=C[2], edgecolor='white', label='Intended', zorder=3)
ax.set_xticks(xs)
ax.set_xticklabels(['155', '350', '355', '400'])
ax.set_xlabel('Unit size (MW)')
ax.set_ylabel('Kinetic energy per unit (MW s)')
ax.set_title('(a) Kinetic energy per unit', color=TINTA)
ax.legend(frameon=False, loc='upper left')
ax.set_ylim(0, 2300)
rejilla(ax)
ax = axs[1]
r0 = np.sort(leer('base_dyn_mp').rocof_peor.values)[::-1]
ax.plot(100 * np.arange(1, len(r0) + 1) / len(r0), r0, color=TINTA2, lw=1.0, ls=':', label='No rule')
for n, et, col in [('rocof1_dyn_mp', 'Indexed limit, intended data', C[2]),
                   ('rocof1_dynsabot_mp', 'Indexed limit, published data', C[0])]:
    r = np.sort(leer(n).rocof_peor.values)[::-1]
    ax.plot(100 * np.arange(1, len(r) + 1) / len(r), r, color=col, lw=1.8, label=et)
ax.axhline(1.0, color=TINTA2, lw=0.8, ls='--')
ax.text(98, 1.01, '1 Hz/s', ha='right', va='bottom', fontsize=7.5, color=TINTA2)
# tramo en que el limite esta activo con los datos pretendidos (tasa >= 0,999 Hz/s, como en cifras.py)
act = 100 * (leer('rocof1_dyn_mp').rocof_peor >= 1 - 1e-3).mean()
ax.axvspan(0, act, color=C[2], alpha=0.10, lw=0, zorder=0)
ax.annotate('limit binding\n(%.0f%% of hours)' % act, xy=(act / 2, 1.0), xytext=(act / 2 + 16, 1.17),
            fontsize=7.5, color=TINTA2, ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color=TINTA2, lw=0.6))
ax.set_xlabel('Hours of the year, sorted (%)')
ax.set_ylabel('RoCoF, intended data (Hz/s)')
ax.set_title('(b) Initial RoCoF, worst loss of each hour', color=TINTA)
ax.set_xlim(0, 100)
ax.set_ylim(0.4, 1.35)
ax.legend(frameon=False, loc='lower left')
rejilla(ax, 'both')
fig.tight_layout()
fig.savefig(os.path.join(FIGS, 'fig3_datos.pdf'), bbox_inches='tight')
plt.close(fig)
print('figuras escritas en', FIGS)
