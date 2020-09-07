import xarray as xr

def open_ds(grid, on_grid=None):
    if on_grid is None:
        return xr.open_dataset(f'/extra-space/sg-stats/Sept-2/{grid}/GCHP.SpeciesConc.Sept.nc', decode_times=False)
    else:
        return xr.open_dataset(f'/extra-space/sg-stats/Sept-2/{grid}/GCHP.SpeciesConc.Sept.{on_grid}.nc', decode_times=False)


def open_percent_diff(grid_a, grid_b):
    ds_a = open_ds(grid_a, grid_b)
    ds_b = open_ds(grid_b)
    return (ds_a - ds_b)/(0.5*(ds_a + ds_b))


def open_diff(grid_a, grid_b):
    ds_a = open_ds(grid_a, grid_b)
    ds_b = open_ds(grid_b)
    return ds_a - ds_b


def open_avg(grid_a, grid_b):
    ds_a = open_ds(grid_a, grid_b)
    ds_b = open_ds(grid_b)
    return 0.5*(ds_a + ds_b)
