

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import scipy.stats
import sklearn.metrics
from stats_stuff.open_ds import *
import os.path

plt.style.use('ggplot')

COLOR = 'k'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR
plt.rcParams['xtick.color'] = COLOR
plt.rcParams['ytick.color'] = COLOR

select = {
    'O3': lambda ds: ds['SpeciesConc_O3']*1e9,
    'NOx': lambda ds: np.log10((ds['SpeciesConc_NO'] + ds['SpeciesConc_NO2'])*1e9),
    'CO': lambda ds: ds['SpeciesConc_CO']*1e9,
    'OH': lambda ds: ds['SpeciesConc_OH']*1e12,
    'PM25': lambda ds: ds['PM25'],
}

limits = {
    'O3': (0.18e-7*1e9, 0.7e-7*1e9),
    'NOx': (-11.2+9, -7.6+9),
    'CO': (0.5e-7*1e9, 3.2e-7*1e9),
    #'OH': (10**-13.8*1e12, 10**-12.3*1e12),
    'OH': (0, 30e-14*1e12),
    'PM25': (0, 30),
}

select_troposphere = lambda ds: ds.where((ds.Met_PMID > ds.Met_TropP) & (ds.Met_PMID > 300))['Met_PMID']

x = open_ds('C96', 'C96')
pressure_x = select_troposphere(open_ds('C96', 'C96'))
y1 = open_ds('S48', 'C96')
y2 = open_ds('C94', 'C96')


fig = plt.figure(figsize=(3.26772, 6.8), constrained_layout=True)
gs = plt.GridSpec(ncols=2, nrows=6, figure=fig, height_ratios=[1, 1, 1, 1, 1, 0.1], wspace=0)

row = ['O3', 'NOx', 'CO', 'OH', 'PM25']
row_label = ['O$_3$, ppb', 'log$_{10}$ NO$_\\mathrm{x}$, ppb', 'CO, ppb', 'OH, ppt', 'PM$_{2.5}$, $\\mathrm{\\mu}$g m$^{-3}$']

def hide_ax(ax):
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    ax.set_facecolor('none')
    ax.grid(False)
    ax.set_yticks([])

tax = fig.add_subplot(gs[4,:])
hide_ax(tax)
tax.set_xlabel('Simulated concentration, C96-global', fontsize=10)

tax = fig.add_subplot(gs[:5,0])
hide_ax(tax)
tax.set_ylabel('Simulated concentration, C96e-NA', fontsize=10, labelpad=30)

tax = fig.add_subplot(gs[:5,-1])
hide_ax(tax)
tax.set_ylabel('Simulated concentration, C94-global', fontsize=10, labelpad=30)
tax.yaxis.tick_right()
tax.yaxis.set_label_position("right")

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
    for ax, y, label in zip([ax1, ax2], [y1, y2], ['C96e-NA', 'C94-global']):

        format_axis(ax, limits[species])

        # x_values = select[species](x).isel(lev=slice(1,None)).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        # y_values = select[species](y).isel(lev=slice(1,None)).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        # pressures = pressure_x.isel(lev=slice(1,None)).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        x_values = select[species](x).isel(lev=slice(0,30)).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        y_values = select[species](y).isel(lev=slice(0,30)).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values.flatten()
        pressures = pressure_x.isel(lev=slice(0,30)).transpose('time', 'nf', 'Ydim', 'lev', 'Xdim').values

        pressures_copy = pressures.flatten()
        pressures[pressures > 1000] = 999
        pressures[:,:,:,0,:] = 1001
        pressures = pressures.flatten()

        isfinite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(pressures_copy) & (pressures_copy > 300)

        # if os.path.exists('isfinite.npy'):
        #     isfinite &= np.load('isfinite.npy')
        #
        # np.save('isfinite', isfinite)

        x_values = x_values[isfinite]
        y_values = y_values[isfinite]
        pressures = pressures[isfinite]

        p = np.random.permutation(x_values.size)
        x_values = x_values[p]
        y_values = y_values[p]
        pressures = pressures[p]

        # def print_sm(x, y):
        #     x_mean = np.mean(x)
        #     y_mean = np.mean(y)
        #     x_std = np.std(x)
        #     y_std = np.std(y)
        #     r2 = sklearn.metrics.r2_score(x, y)
        #     rmse = np.sqrt(sklearn.metrics.mean_squared_error(x, y))
        #     mae = sklearn.metrics.mean_absolute_error(x, y)
        #     mb = np.mean(y) - np.mean(x)
        #
        #     print(f'x({species};{label}): {x_mean} & {x_std}')
        #     print(f'y({species};{label}): {y_mean} & {y_std} & {mb} & {mae} & {rmse}')
        #
        # print_sm(x_values, y_values)

        # ax.set_xlabel('C96', fontsize=10)
        # ax.set_ylabel(f'Sim. mean, {label}', fontsize=6)

        ax.tick_params(axis='both', which='major', labelsize=7)

        if label == 'C94-global':
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
        # pressures[pressures>999] = 999
        cmap = plt.get_cmap('Spectral_r')
        cmap.set_over('tab:brown')
        cmap.set_under('green')
        #pressures[pressures<300] = 299
        # indexes near 4,40,9
        ax.scatter(x_values, y_values, c=pressures, edgecolor='', cmap=cmap, norm=plt.Normalize(300, 1000), s=2)
        ax.text(0.05, 0.95, f'{l}', horizontalalignment='left', verticalalignment='top', transform=ax.transAxes, fontsize=6)

        # x_values = select[species](x).isel(lev=0).transpose('time', 'nf', 'Ydim', 'Xdim').values.flatten()
        # y_values = select[species](y).isel(lev=0).transpose('time', 'nf', 'Ydim','Xdim').values.flatten()
        # pressures = pressure_x.isel(lev=0).transpose('time', 'nf', 'Ydim', 'Xdim').values.flatten()
        # isfinite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(pressures)
        # x_values = x_values[isfinite]
        # y_values = y_values[isfinite]
        # pressures = pressures[isfinite]
        # p = np.random.permutation(x_values.size)
        # x_values = x_values[p]
        # y_values = y_values[p]
        # pressures = pressures[p]
        # pressures[:] = 1001
        # ax.scatter(x_values, y_values, c=pressures, edgecolor='', cmap=cmap, norm=plt.Normalize(300, 1000), s=2)

        # x_values = select[species](x).isel(nf=4, Ydim=slice(38, 42), Xdim=slice(7,11)).transpose('time', 'lev', 'Ydim', 'Xdim').values.flatten()
        # y_values = select[species](y).isel(nf=4, Ydim=slice(38, 42), Xdim=slice(7,11)).transpose('time', 'lev', 'Ydim','Xdim').values.flatten()
        # pressures = pressure_x.isel(nf=4, Ydim=slice(38, 42), Xdim=slice(7,11)).transpose('time', 'lev', 'Ydim', 'Xdim').values.flatten()
        # isfinite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(pressures)
        # x_values = x_values[isfinite]
        # y_values = y_values[isfinite]
        # pressures = pressures[isfinite]
        # p = np.random.permutation(x_values.size)
        # x_values = x_values[p]
        # y_values = y_values[p]
        # pressures = pressures[p]
        # pressures[:] = 100
        # ax.scatter(x_values, y_values, c=pressures, edgecolor='', cmap=cmap, norm=plt.Normalize(300, 1000), s=2)


# ax1.set_xlabel('Sim. mean, C96-global', fontsize=6)
# ax2.set_xlabel('Sim. mean, C96-global', fontsize=6)
fig.align_ylabels(ax_col1)
fig.align_ylabels(ax_col2)
ax = fig.add_subplot(gs[-1,:])


import matplotlib.cm
cbar = plt.colorbar(matplotlib.cm.ScalarMappable(norm=plt.Normalize(300, 1000), cmap='Spectral_r'), cax=ax, orientation='horizontal', extend='max', extendrect=True)
cbar.set_label('Pressure (hPa)', fontsize=10)
cbar.ax.invert_xaxis()

# plt.tight_layout()
plt.savefig('/home/liam/mapping/stats_stuff/foo2.png', dpi=300, bbox_inches='tight', pad_inches=0.01)
plt.show()
