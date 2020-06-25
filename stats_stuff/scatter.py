

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.stats
from stats_stuff.open_ds import *

plt.style.use('ggplot')

select = {
    'O3': lambda ds: ds['SpeciesConc_O3']*1e9,
    'NOx': lambda ds: np.log10(ds['SpeciesConc_NO'] + ds['SpeciesConc_NO2']),
    'CO': lambda ds: ds['SpeciesConc_CO']*1e9,
    'OH': lambda ds: ds['SpeciesConc_OH']*1e12,
    'PM25': lambda ds: ds['PM25'],
}

limits = {
    'O3': (0.2e-7*1e9, 0.9e-7*1e9),
    'NOx': (-10.75, -7.8),
    'CO': (0.5e-7*1e9, 3.2e-7*1e9),
    'OH': (10**-13.8*1e12, 10**-12.3*1e12),
    'PM25': (0, 30),
}

select_troposphere = lambda ds: ds.where((ds.Met_PMID > ds.Met_TropP) & (ds.Met_PMID > 300))['Met_PMID']

x = open_ds('C96', 'C96')
pressure_x = select_troposphere(open_ds('C96', 'C96'))
y1 = open_ds('S48', 'C96')
y2 = open_ds('C94', 'C96')


fig = plt.figure(figsize=(3.23772, 7), constrained_layout=True)
gs = plt.GridSpec(ncols=2, nrows=6, figure=fig, height_ratios=[1, 1, 1, 1, 1, 0.1], wspace=0)

row = ['O3', 'NOx', 'CO', 'OH', 'PM25']
row_label = ['O$_3$, ppb', 'log$_{10}$ NO$_\\mathrm{x}$', 'CO, ppb', 'OH, ppt', 'PM$_{2.5}$, $\\mu$g m$^{-3}$']

ax_col1 = []
ax_col2 = []


for i, (r, l) in enumerate(zip(row, row_label)):
    ax1 = fig.add_subplot(gs[i, 0])
    ax2 = fig.add_subplot(gs[i, 1])

    ax_col1.append(ax1)
    ax_col2.append(ax2)

    # ax2.yaxis.tick_right()
    # ax2.yaxis.set_label_position("right")

    ticker = matplotlib.ticker.MaxNLocator(2)


    def format_axis(ax, lim):
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_aspect('equal', adjustable='box')
        ax.xaxis.set_major_locator(ticker)
        ax.yaxis.set_major_locator(ticker)
        ax.plot(lim, lim, color='k', linestyle='--', linewidth=0.5)


    species = r
    for ax, y, label in zip([ax1, ax2], [y1, y2], ['S48', 'C94']):

        format_axis(ax, limits[species])

        x_values = select[species](x).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        y_values = select[species](y).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        pressures = pressure_x.transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()

        isfinite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(pressures)

        x_values = x_values[isfinite]
        y_values = y_values[isfinite]
        pressures = pressures[isfinite]

        p = np.random.permutation(x_values.size)
        x_values = x_values[p]
        y_values = y_values[p]
        pressures = pressures[p]

        # ax.set_xlabel('C96', fontsize=10)
        ax.set_ylabel(f'Simulated mean, {label}', fontsize=6)

        ax.tick_params(axis='both', which='major', labelsize=7)

        ax.scatter(x_values, y_values, c=pressures, edgecolor='', cmap='Spectral_r', norm=plt.Normalize(300, 1000), s=2)
        ax.text(0.05, 0.95, f'{l}', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=6)

ax1.set_xlabel('Simulated mean, C96', fontsize=6)
ax2.set_xlabel('Simulated mean, C96', fontsize=6)
fig.align_ylabels(ax_col1)
fig.align_ylabels(ax_col2)
ax = fig.add_subplot(gs[-1,:])

import matplotlib.cm
cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(300, 1000), cmap='Spectral_r'), cax=ax, orientation='horizontal')
cbar.set_label('Pressure, [hPa]', fontsize=10)
cbar.ax.invert_xaxis()

# plt.tight_layout()
# plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/validation_scatters.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
