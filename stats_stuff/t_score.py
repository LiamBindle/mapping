

import numpy as np
import xarray as xr
grids = {
    'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/grid_box_outlines_and_centers.nc'),
    'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/grid_box_outlines_and_centers.nc'),
    'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/grid_box_outlines_and_centers.nc'),
    'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/grid_box_outlines_and_centers.nc'),
    'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/grid_box_outlines_and_centers.nc'),
}

data = {
    'C96': xr.open_dataset('/extra-space/sg-stats/June/C96/species.june.nc'),
    'C98': xr.open_dataset('/extra-space/sg-stats/June/C98/species.june.C96.nc'),
    'C100': xr.open_dataset('/extra-space/sg-stats/June/C100/species.june.C96.nc'),
    'S48': xr.open_dataset('/extra-space/sg-stats/June/S48/species.june.C96.nc'),
    'C90': xr.open_dataset('/extra-space/sg-stats/June/C90/species.june.C96.nc'),
}


import scipy.stats
import sklearn.metrics

def estimate_sample_stats(dataarrays: list, eval_stat: callable):
    samples = [eval_stat(da) for da in dataarrays]
    mean = np.mean(samples)
    # se = scipy.stats.sem(samples)
    se = np.std(samples, ddof=1)
    return mean, se, samples

data = {k: v.isel(time=0, lev=slice(0, 20)) for k, v in data.items()}
data = {k: v['SpeciesConc_O3'].where(v['max_intersect'] > 0.5) if 'max_intersect' in v else v['SpeciesConc_O3'] for k, v in data.items()}

residuals = {
    'C98': data['C98'] - data['C96'],
    'C100': data['C100'] - data['C96'],
    'S48': data['S48'] - data['C96'],
    'C90': data['C90'] - data['C96'],
}

# func = lambda x: np.nanmean(np.abs(x))
func = lambda x: np.nanstd(x)
# func = lambda x: np.sqrt(np.nansum(x**2)/np.count_nonzero(np.isfinite(x)))


variance_i = lambda x: (x-np.nanmean(x))**2
variance = lambda x: np.nanmean(variance_i(x))


mean, se, samples = estimate_sample_stats([residuals['C98'], residuals['C100'], residuals['C90']], func)


X = func(residuals['S48'])

print('Samples: ', samples)
print('X: ', X)

print((X-mean)/se)


