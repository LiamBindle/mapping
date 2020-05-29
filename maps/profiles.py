import matplotlib.pyplot as plt
import matplotlib.colors as colors
import xarray as xr
import numpy as np

def central_angle(x0, y0, x1, y1):
    RAD2DEG = 180 / np.pi
    DEG2RAD = np.pi / 180
    x0 = x0 * DEG2RAD
    x1 = x1 * DEG2RAD
    y0 = y0 * DEG2RAD
    y1 = y1 * DEG2RAD
    return np.arccos(np.sin(y0) * np.sin(y1) + np.cos(y0) * np.cos(y1) * np.cos(np.abs(x0-x1))) * RAD2DEG

plt.figure()

species = xr.open_dataset('/home/liam/Downloads/GCHP.JulyMean.NoMasking.nc')
metc = xr.open_dataset('/home/liam/Downloads/GCHP.JulyMean.NoMasking.MetC.nc')
mete = xr.open_dataset('/home/liam/Downloads/GCHP.JulyMean.NoMasking.MetE.nc').isel(lev=slice(0, -1)).assign_coords(lev=np.arange(1, 73, dtype=float))

nf=5
Xdim=33
Ydim_slice=slice(65, 80)

species = species.isel(nf=nf, Ydim=Ydim_slice, Xdim=Xdim).squeeze()
metc = metc.isel(nf=nf, Ydim=Ydim_slice, Xdim=Xdim).squeeze()
mete = mete.isel(nf=nf, Ydim=Ydim_slice, Xdim=Xdim).squeeze()


# vertical_grid = np.logspace(np.log10(100), np.log10(metc.Met_PMID.max()), num=500)[::-1]

lats = species.lats[:-1] + species.lats.diff(dim='Ydim').values
lons = species.lons[:-1] + species.lons.diff(dim='Ydim').values

print(np.moveaxis([lats, lons], 0, -1))

distance = central_angle(lons[-1], lats[-1], lons, lats)
distance[-1] = 0

distance = distance*np.pi/180 * 6371


pedge = mete.Met_PEDGE

pedge_edges = pedge[:, :-1] + pedge.diff(dim='Ydim').values
pedge_centers = pedge[:, 1:-1]

vertical_grid = np.zeros((73, len(distance)))
vertical_grid[:-1, :] = pedge_edges
vertical_grid[-1, :] = 1e-10

conc=species.SpeciesConc_NO2.isel(Ydim=slice(1, -1))


X, y = np.meshgrid(np.arange(73), distance, indexing='ij')



names = ['Lancaster', 'Angeles Forest', 'Pasadena', 'LA']
coords = np.array([(-117.98, 34.67), (-117.97, 34.30), (-118.03, 34.16), (-118.02, 33.91)])

xticks = central_angle(lons[-1].item(), lats[-1].item(), coords[:,0], coords[:,1]) *np.pi/180 * 6371


c = plt.pcolormesh(y, vertical_grid, conc, norm=colors.LogNorm(vmin=1e-10, vmax=1e-8))
plt.colorbar(c)
plt.xticks(xticks, names)
plt.gca().set_yscale('log')
plt.ylim([600, vertical_grid.max()])
plt.gca().invert_yaxis()
plt.show()
