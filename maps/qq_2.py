import numpy as np
import xarray as xr
import maps
import  scipy.stats
import statsmodels.api
import matplotlib.pyplot as plt

grids = {
    'C60-global': xr.open_dataset('/extra-space/C60_global/comparison_grid.nc'),
    'C180-global': xr.open_dataset('/extra-space/C180_global/comparison_grid.nc'),
    'C180e-US': xr.open_dataset('/extra-space/C180e_US/comparison_grid.nc'),
}


masks = {}
for grid_name, grid in grids.items():
    xc = grid['grid_boxes_centers'].isel(XY=0).values
    yc = grid['grid_boxes_centers'].isel(XY=1).values

    region = maps.get_countries('/home/liam/Downloads').loc['United States of America'].geometry.simplify(0.1)
    masks[grid_name] = maps.mask_outside(xc, yc, region)

simulations = {
    'C60-global': xr.open_dataset('/extra-space/C60_global/GCHP_NO2_July2018.nc'),
    'C180-global': xr.open_dataset('/extra-space/C180_global/GCHP_NO2_July2018.nc'),
    'C180e-US': xr.open_dataset('/extra-space/C180e_US/GCHP_NO2_July2018.nc'),
}

tropomi = {
    'C180-global': xr.open_dataset('/extra-space/C180_global/TROPOMI_NO2_July2018.nc'),
    'C180e-US': xr.open_dataset('/extra-space/C180e_US/TROPOMI_NO2_July2018.nc'),
}

sim_values = {name: np.log10(simulations[name].GCHP_NO2.values[~masks[name]]) for name in simulations.keys()}
trop_values = {name: tropomi[name].TROPOMI_NO2_molec_per_cm2.values[~masks[name]] for name in tropomi.keys()}
tropomi_values = np.concatenate([*trop_values.values()])

log_tropomi = np.log10(tropomi_values)
log_tropomi_distparams = scipy.stats.distributions.norm.fit(log_tropomi)

plt.figure(figsize=(3.26772, 4.5))
ax = plt.gca()
colors = plt.get_cmap('Set1').colors
plot_kwargs=dict(markersize=4, alpha=0.5, linewidth=2, linestyle='solid', marker='.', fit=False)
statsmodels.api.qqplot(log_tropomi, dist=scipy.stats.distributions.norm, loc=log_tropomi_distparams[0], scale=log_tropomi_distparams[1], ax=ax, label='TROPOMI', c=colors[3], **plot_kwargs, zorder=10)
statsmodels.api.qqplot(sim_values['C180-global'], dist=scipy.stats.distributions.norm, loc=log_tropomi_distparams[0], scale=log_tropomi_distparams[1], ax=ax, label='C180-global', c=colors[0], **plot_kwargs)
statsmodels.api.qqplot(sim_values['C180e-US'], dist=scipy.stats.distributions.norm, loc=log_tropomi_distparams[0], scale=log_tropomi_distparams[1], ax=ax, label='C180e-US', c=colors[1], **plot_kwargs)
statsmodels.api.qqplot(sim_values['C60-global'], dist=scipy.stats.distributions.norm, loc=log_tropomi_distparams[0], scale=log_tropomi_distparams[1], ax=ax, label='C60-global', c=colors[2], **plot_kwargs, zorder=11)
plt.legend(loc='upper center',  framealpha=1, bbox_to_anchor=(0.48, 1.2), fontsize=8, ncol=2)
box = ax.get_position()
ax.set_position([box.x0+0.05, box.y0, box.width, box.height*0.95])
ax.set_aspect('equal')
xlim = ax.get_xlim()
ylim = ax.get_ylim()
plt.plot([14, 16], [14, 16], linewidth=0.8, linestyle='dashed', zorder=-10, c='k')
ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
plt.ylabel('Distribution quantiles ($\\log_{10} [\\mathrm{NO_2}]$)')
plt.xlabel('Log-normal dist. quantiles ($\\log_{10} [\\mathrm{NO_2}]$)')
# plt.show()
plt.savefig('/home/liam/gmd-sg-manuscript-2020/figures/US_qqplot.png', dpi=300)

# import pandas as pd
#
# df = pd.DataFrame()
# for name in sim_values.keys():
#     df2 = pd.DataFrame({'Column Density': all_data[name]})
#     df2['Simulation'] = name
#     df2['Source'] = 'Simulated'
#
#     df3 = pd.DataFrame({'Column Density': log_tropomi})
#     df3['Simulation'] = name
#     df3['Source'] = 'TROPOMI'
#
#     df2 = pd.concat([df2, df3])
#     df = pd.concat([df, df2])
#
# import seaborn
# seaborn.violinplot(x='Simulation', y='Column Density', split=True, hue='Source', data=df)


print('hello')