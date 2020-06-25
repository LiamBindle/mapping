

import numpy as np
import xarray as xr

# data = {
#     'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/species.june.nc'),
#     'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/species.june.C96.nc'),
#     'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/species.june.C96.nc'),
#     'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/species.june.C96.nc'),
#     'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/species.june.C96.nc'),
# }

def open_ds(grid, on_grid=None):
    if on_grid is None:
        return xr.open_dataset(f'/extra-space/sg-stats/July/{grid}/GCHP.SpeciesConc.July.nc')
    else:
        return xr.open_dataset(f'/extra-space/sg-stats/July/{grid}/GCHP.SpeciesConc.July.{on_grid}.nc')

residuals = [
    open_ds('C90', 'C92') - open_ds('C92'),
    # open_ds('C92', 'C90') - open_ds('C90'),

    open_ds('C92', 'C94') - open_ds('C94'),
    # open_ds('C94', 'C92') - open_ds('C92'),

    open_ds('C94', 'C96') - open_ds('C96'),
    # open_ds('C96', 'C94') - open_ds('C94'),

    open_ds('C96', 'C98') - open_ds('C98'),
    # open_ds('C98', 'C96') - open_ds('C96'),

    open_ds('C90', 'C96') - open_ds('C96'),
    open_ds('C92', 'C96') - open_ds('C96'),
    open_ds('C94', 'C96') - open_ds('C96'),
    open_ds('C98', 'C96') - open_ds('C96'),
]
test = open_ds('S48', 'C96') - open_ds('C96')


rmse = lambda x: np.nanstd(x)           # std of residuals -> RMSE
mae = lambda x: np.nanmean(np.abs(x))   # mean of abs -> MAE
mb = lambda x: np.nanmean(x)            # mean -> MB

evaluator = mb

import matplotlib.pyplot as plt

def make_line(r, name):
    values = []
    levels = np.arange(0, 25)
    for lev in levels:
        val = evaluator(r.isel(time=0, lev=lev)['SpeciesConc_O3']).item()
        values.append(val)

    plt.plot(values, levels, label=name)

make_line(residuals[0], 'CS')
make_line(residuals[1], 'CS')
make_line(residuals[2], 'CS')
make_line(residuals[3], 'CS')
make_line(residuals[4], 'CS')
make_line(residuals[5], 'CS')
make_line(residuals[6], 'CS')
make_line(residuals[7], 'CS')


make_line(test, 'SG')

plt.legend()
plt.show()


# selector = lambda x: x.isel(time=0, lev=slice(0, 20))['SpeciesConc_O3']
# residuals = [selector(r) for r in residuals]
# test = selector(test)
#
# samples = [evaluator(r) for r in residuals]
# value = evaluator(test)
#
#
#
#
# print('Samples: ', samples)
# print('Samples mean: ', np.mean(samples))
# print('Samples std: ', np.std(samples))
# print('Value: ', value)
# print('Score: ', (value - np.mean(samples))/np.std(samples, ddof=1))


#
#
#
#
#
# import scipy.stats
# import sklearn.metrics
#
# def estimate_sample_stats(dataarrays: list, eval_stat: callable):
#     samples = [eval_stat(da) for da in dataarrays]
#     mean = np.mean(samples)
#     # se = scipy.stats.sem(samples)
#     se = np.std(samples, ddof=1)
#     return mean, se, samples
#
# data = {k: v.isel(time=0, lev=slice(0, 20)) for k, v in data.items()}
# data = {k: v['SpeciesConc_O3'].where(v['max_intersect'] > 0.5) if 'max_intersect' in v else v['SpeciesConc_O3'] for k, v in data.items()}
#
# residuals = {
#     'C98': data['C98'] - data['C96'],
#     'C100': data['C100'] - data['C96'],
#     'S48': data['S48'] - data['C96'],
#     'C90': data['C90'] - data['C96'],
# }
#
# # func = lambda x: np.nanmean(np.abs(x))
# func = lambda x: np.nanstd(x)
# # func = lambda x: np.sqrt(np.nansum(x**2)/np.count_nonzero(np.isfinite(x)))
#
#
# variance_i = lambda x: (x-np.nanmean(x))**2
# variance = lambda x: np.nanmean(variance_i(x))
#
#
# mean, se, samples = estimate_sample_stats([residuals['C98'], residuals['C100'], residuals['C90']], func)
#
#
# X = func(residuals['S48'])
#
# print('Samples: ', samples)
# print('X: ', X)
#
# print((X-mean)/se)
#
#
