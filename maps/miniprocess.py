import argparse
import xarray as xr


# def compute_no2_column(ds_species, ds_metc, ds_mete):
#     Md = 28.9647e-3 / 6.0221409e+23  # [kg molec-1]
#     no2_area_density = ds_metc['Met_AIRDEN'] * ds_metc['Met_BXHEIGHT'] * ds_species['SpeciesConc_NO2'] / Md
#
#     def sum_below_tropopause(column, pfloor, tropp):
#         in_troposphere = pfloor > tropp
#         return column[in_troposphere].sum(axis=0)
#
#     pfloor = ds_mete['Met_PEDGE'].isel(lev=slice(0, -1)).assign_coords({'lev': no2_area_density['lev']})
#     tropospheric_no2 = xr.apply_ufunc(
#         sum_below_tropopause,
#         no2_area_density,
#         pfloor,
#         ds_metc['Met_TropP'],
#         input_core_dims=[['lev'], ['lev'], []],
#         vectorize=True
#     )
#     tropospheric_no2 = tropospheric_no2 / 100**2  # [molec m-2] -> [molec cm-2]
#
#     ds_out = xr.Dataset({'TroposphericColumn_NO2': tropospheric_no2})
#     ds_out['TroposphericColumn_NO2'].attrs['units'] = 'molec cm-2'
#     return ds_out


def mixing_ratio_to_area_density(ds, ds_metc):
    Md = 28.9647e-3 / 6.0221409e+23  # [kg molec-1]
    ds = ds_metc['Met_AIRDEN'] * ds_metc['Met_BXHEIGHT'] * ds / Md
    return ds


def compute_column(ds, ds_mete):
    def sum_below_tropopause(column, pfloor, tropp):
        in_troposphere = pfloor > tropp
        return column[in_troposphere].sum(axis=0)

    pfloor = ds_mete['Met_PEDGE'].isel(lev=slice(0, -1)).assign_coords({'lev': ds.coords['lev']})
    ds = xr.apply_ufunc(
        sum_below_tropopause,
        ds,
        pfloor,
        ds_metc['Met_TropP'],
        input_core_dims=[['lev'], ['lev'], []],
        vectorize=True
    )
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ds',
                        type=str,
                        required=True)
    parser.add_argument('--vars',
                        nargs='+',
                        type=str,
                        required=True),
    parser.add_argument('--mixing_ratio',
                        action='store_true')
    parser.add_argument('--metc',
                        type=str,
                        required=True)
    parser.add_argument('--mete',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        type=str,
                        required=True)
    args = parser.parse_args()

    ds_metc = xr.open_dataset(args.metc)
    ds_mete = xr.open_dataset(args.mete)

    ds = xr.open_dataset(args.ds)
    ds = ds.drop([n for n in ds.data_vars if n not in args.vars])

    if args.mixing_ratio:
        ds = mixing_ratio_to_area_density(ds, ds_metc)
    else:
        ds = ds.sortby('lev', ascending=False)

    ds = compute_column(ds, ds_mete)
    ds.to_netcdf(args.o)