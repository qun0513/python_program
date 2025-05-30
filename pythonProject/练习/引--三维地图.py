import itertools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
import cartopy.feature
from cartopy.mpl.patch import geos_to_path
import cartopy.crs as ccrs
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False#负号

fig = plt.figure(figsize=(10,8),dpi=200)
ax = Axes3D(fig, xlim=[-180, 180], ylim=[-90, 90])
ax.set_zlim(-0.1,0.5)
##############################################################
concat = lambda iterable: list(itertools.chain.from_iterable(iterable))
target_projection = ccrs.PlateCarree()
#目标投影，这里用最常见的一种
feature = cartopy.feature.NaturalEarthFeature('physical', 'land', '110m')
geoms = feature.geometries()
geoms = [target_projection.project_geometry(geom, feature.crs)
         for geom in geoms]
paths = concat(geos_to_path(geom) for geom in geoms) #geom转path
polys = concat(path.to_polygons() for path in paths) #path转poly
lc = PolyCollection(polys, edgecolor='black',
                    facecolor='yellow', closed=False)
ax.add_collection3d(lc)
ax.set_xlabel('经度')
ax.set_ylabel('纬度')
ax.set_zlabel('高度')

lc = PolyCollection(polys, edgecolor='black',
                    facecolor='yellow', closed=False)
lc2 = PolyCollection(polys, edgecolor='black',
                    facecolor='green', closed=False)
lc3 = PolyCollection(polys, edgecolor='black',
                    facecolor='white', closed=False)
ax.add_collection3d(lc,zs=0)
ax.add_collection3d(lc2,zs=0.25)
ax.add_collection3d(lc3,zs=0.5)

plt.show()