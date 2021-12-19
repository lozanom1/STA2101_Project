#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 16:19:19 2021

@author: marialozano
"""
import csv
import pwlf
import struct
import xarray
import pyresample
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc
import datetime as dt
import cartopy.crs as ccrs
import cartopy.util as cutil
import statsmodels.api as sm
import cartopy.feature as cft
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import mpl_toolkits.basemap as bm
import matplotlib.ticker as mticker
from matplotlib.path import Path
from matplotlib.patches import Polygon

from affine import Affine
from pyproj import Proj
from rasterio import features
from mpl_toolkits.basemap import Basemap
from pyresample import image, geometry, plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

def nc_read(filename, variable):
    '''
    Read variable data from a NetCDF source
    :param filename:(string) complete path and file name
    :param variable: (string) name of variable
    :return: numpy array containing data
    '''
    
    data = nc.Dataset(filename)
    var = np.squeeze(data[variable][:])
    
    return var

def var_fix(fn, var_name, sic=0):
    if sic==0:
        var = nc_read(fn, var_name)
        var = np.einsum('kji->ijk', var)
        var = var[:,::-1,:]
    elif sic==1:
        var = nc_read(fn, var_name)
        var = np.einsum('kji->ijk', var)
        var = var[:,::-1,:]
        var = np.nan_to_num(var)
    return var

def draw_screen_poly(lats, lons, m):
    x, y = m( lons, lats)
    xy = zip(x,y)
    poly = Polygon(list(xy), edgecolor='red', facecolor='none', linewidth=2)
    print(poly)
    plt.gca().add_patch(poly)
    return poly
    
def plot_piomas(field, lat, lon, clevs, units, lat_o_plt, plot_title, path_out, season, save=0):
    plt.style.use('seaborn-bright')
    clevs = np.arange(-0.09, 0.091, 0.02)
    fig = plt.figure(figsize=[12,14])
    m = bm.Basemap(projection='npstere', boundinglat=lat_o_plt, lat_0=90, lon_0=0, resolution ='l', round=True)
    x,y = m(lon, lat)

    m.drawcoastlines(linewidth=0.5)
    circle = m.drawmapboundary(linewidth=(1),color='k')
    circle.set_clip_on(False)
    m.drawparallels(np.arange(20,90,10), linewidth=0.1)
    m.drawmeridians(np.arange(-180,181,45), linewidth=0.1, latmax=90)
    cc = m.contourf(x,y,field, cmap='RdBu')
    if season=='Winter':
        pg1_lat = [ 70, 82, 77, 69 ]
        pg1_lon = [ 35, 35, 99, 66 ]
        pg2_lat = [ 70, 80, 82, 72 ]
        pg2_lon = [ -18, -18, 35, 35 ]
        pg3_lat = [ 82, 89, 74, 72 ]
        pg3_lon = [ -15, -5, -140, -110 ]
        pg4_lat = [ 70, 85, 85, 70 ]
        pg4_lon = [ -135, -120, 175, 180 ]
        pg5_lat = [ 68, 81, 82, 72 ]
        pg5_lon = [ 166, 175, 80, 112 ]
        poly1 = draw_screen_poly(pg1_lat, pg1_lon, m)
        poly2 = draw_screen_poly(pg2_lat, pg2_lon, m)
        poly3 = draw_screen_poly(pg3_lat, pg3_lon, m)
        poly4 = draw_screen_poly(pg4_lat, pg4_lon, m)
        poly5 = draw_screen_poly(pg5_lat, pg5_lon, m)
    # if season=='Summer':
    #     pg1_lat = [ 82, 88, 74, 72 ]
    #     pg1_lon = [ -60, -65, -135, -110 ]
    #     pg2_lat = [ 70, 82, 81, 70 ]
    #     pg2_lon = [ -140, -145, 165, 170 ]
    #     pg3_lat = [ 68, 81, 82, 72 ]
    #     pg3_lon = [ 166, 165, 120, 112 ]
    #     pg4_lat = [ 75, 88, 82,75 ]
    #     pg4_lon = [ -16, -20, 120, 90 ]
    #     draw_screen_poly(pg1_lat, pg1_lon, m)
    #     draw_screen_poly(pg2_lat, pg2_lon, m)
    #     draw_screen_poly(pg3_lat, pg3_lon, m)
    #     draw_screen_poly(pg4_lat, pg4_lon, m)
    
    
    # for col in cc.collections:
    #     col.set_clip_path(poly1)
        # col.set_clip_path(poly2)
        # col.set_clip_path(poly3)
        # col.set_clip_path(poly4)
        # col.set_clip_path(poly5)
    
    cbar = plt.colorbar(cc, orientation='horizontal')
    # cc = m.contourf(x,y,field, clevs, cmap='RdBu')
    # cbar = plt.colorbar(cc, ticks = clevs, orientation='horizontal')
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label(units, fontsize=25)
    plt.title(plot_title, fontsize=35, pad=30) 


def least_sqr_adj(X,Y):
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    n = len(X)
    num = 0
    den = 0 
    
    for i in range(n):
        num += ((X[i]-X_mean)*(Y[i]-Y_mean))
        den += ((X[i]-X_mean)**2)
    # print(X_mean, X)
    m = num/den
    c = Y_mean - m*X_mean
    
    Y_pred = m*X + c
    
    return Y_pred, m

  
def cont_trend(X,Y):
    model = pwlf.PiecewiseLinFit(X,Y)
    res = model.fit(2)
    
    xHat = np.arange(min(X), max(X), 1)
    yHat = model.predict(xHat)
    
    return xHat, yHat

def SIV_plot(sit):
    sit_w = sit[:,[0,1,2],:,:]
    sit_s = sit[:,6:9,:,:]
    sit_all = np.nanmean(sit, axis=1)
    sit_w = np.nanmean(sit_w, axis=1)
    sit_s = np.nanmean(sit_s, axis=1)
    
    sit_all = np.nanmean(sit_all, axis=1)
    sit_all = np.nanmean(sit_all, axis=1)
    sit_w = np.nanmean(sit_w, axis=1)
    sit_w = np.nanmean(sit_w, axis=1)
    sit_s = np.nanmean(sit_s, axis=1)
    sit_s = np.nanmean(sit_s, axis=1)
    
    sit_mn = np.nanmean(sit_all)
    sit_w_mn = np.nanmean(sit_w)
    sit_s_mn = np.nanmean(sit_s)
    sit_anom = sit_all - sit_mn
    sit_w_anom = sit_w - sit_w_mn
    sit_s_anom = sit_s - sit_s_mn
    
    years=np.arange(1979,2021,1)
    sit_trend, m = least_sqr_adj(years, sit_anom)
    
    plt.style.use('seaborn-dark')
    plt.figure(figsize=[20,10])
    plt.plot(np.arange(0,42,1),sit_anom,linewidth=2, label = 'Annual', color = 'k')
    plt.plot(np.arange(0,42,1),sit_w_anom,linewidth=2,label= 'Winter (JFM)', color = 'blue')
    plt.plot(np.arange(0,42,1),sit_s_anom,linewidth=2, label = 'Summer (JAS)', color = 'orange')
    plt.plot(np.arange(0,42,1), sit_trend, linewidth=2,label = "LS regression line", color="red")
    plt.legend(fontsize=20)
    plt.xticks(np.arange(0,42,1), years,rotation=60, fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('Sea Ice Thickness Anomalies', fontsize=30, pad=20)
    plt.xlabel('Year', fontsize=25)
    plt.ylabel('Ice volume (m)', fontsize=25)

def trend_map(sit, lat, lon, yrs, season):
    sit = np.nanmean(sit, axis=1)
    trends = np.empty([len(lat), len(lon)])
    for i in range(len(lat)):
        for j in range(len(lon)):
            if sit[0,i,j] == np.nan:
                trends[i,j] = np.nan
            elif sit[0,i,j] != np.nan:
                trend, m = least_sqr_adj(yrs, sit[:,i,j])
                trends[i,j] = m

    plot_piomas(trends, lat, lon, 0, 'Slope', 65, 'Plot of '+season+' SIT Linear Trend Slopes',season,'')    

def trend_SIT_plot(sit, yrs, season):
    if season == 'JFM':
        seas = "Winter"
    elif season == 'JAS':
        seas="Summer"
    sit = np.nanmean(sit, axis=3)
    sit = np.nanmean(sit, axis=2)
    sit_mn = np.nanmean(sit,axis=1)
    sit_seas = np.empty([len(sit[:,0])*len(sit[0,:])])
    n=3
    y_n = len(yrs)

    for i in range(len(yrs)):
        sit_tmp = sit[i,:]
        sit_seas[i*n:(i+1)*n] = sit_tmp
    years=np.arange(1979,2017,1)
    
    x_cont, y_cont = cont_trend(np.arange(0,y_n,1/n), sit_seas)
    
    plt.style.use('seaborn-dark')
    plt.figure(figsize=[20,10])
    plt.scatter(np.arange(0,y_n,1/n), sit_seas, color = 'b', label = season )
    plt.plot(np.arange(0,y_n,1),sit_mn,linewidth=1, color='k',label= season+' mean')
    plt.plot(x_cont, y_cont, color='r', linewidth=3, label='continuous piecewise trend')        
    plt.legend(fontsize=20)
    plt.xticks(np.arange(0,y_n,1), years,rotation=60, fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(seas+' Sea Ice Thickness \n Piecewise Linear Behaviour', fontsize=30, pad=20)
    plt.xlabel('Year', fontsize=25)
    plt.ylabel('Ice volume (m)', fontsize=25)
    
# def anomaly_plot(sit,yrs,season):

def reduce_field(field, lon, lat, yrs, poly_verts, joined=0):
    if len(field[0,:,0]) == 93:
        t = len(field[:,0,0])
        nx, ny = 93, 93
        x,y = lon.flatten(), lat.flatten()
        points = np.vstack((x,y)).T
        if joined==0:
            path = Path(poly_verts)
            grid = path.contains_points(points)
            grid = grid.reshape((ny, nx))
            print(np.shape(grid))
            # print(np.argwhere(grid==True))
            sit_rg = np.ma.empty([t, nx, ny])
            for i in range(len(field[:,0,0])):
                sit_tmp = np.ma.array(field[i,:,:], mask = ~grid)
                sit_rg[i,:,:] = sit_tmp
        elif joined!=0:
            path1 = Path(poly_verts)
            path2 = Path(joined)
            grid1 = path1.contains_points(points)
            grid1 = grid1.reshape(nx,ny)
            grid2 = path2.contains_points(points)
            grid2 = grid2.reshape(nx,ny)
            sit_rg = np.ma.empty([t,ny,nx])
            for i in range(len(field[:,0,0])):
                sit_tmp = np.ma.array(field[i,:,:], mask = ~(grid1+grid2))
                sit_rg[i,:,:] = sit_tmp[:,:]
    if len(field[0,:,0]) != 93:
        t = len(field[:,0,0])
        nx, ny = 896, 250
        x,y = lon.flatten(), lat.flatten()
        points = np.vstack((x,y)).T
        if joined==0:
            path = Path(poly_verts)
            grid = path.contains_points(points)
            grid = grid.reshape((ny, nx))
            # print(np.argwhere(grid==True))
            sit_rg = np.ma.empty([t, ny, nx])
            for i in range(len(field[:,0,0])):
                sit_tmp = np.ma.array(field[i,:,:], mask = ~grid)
                sit_rg[i,:,:] = sit_tmp
        elif joined!=0:
            path1 = Path(poly_verts)
            path2 = Path(joined)
            grid1 = path1.contains_points(points)
            grid1 = grid1.reshape(nx,ny)
            grid2 = path2.contains_points(points)
            grid2 = grid2.reshape(nx,ny)
            sit_rg = np.ma.empty([t,ny,nx])
            for i in range(len(field[:,0,0])):
                sit_tmp = np.ma.array(field[i,:,:], mask = ~(grid1+grid2))
                sit_rg[i,:,:] = sit_tmp[:,:]
    
    plot_piomas(sit_rg[0,:,:], lat, lon, 0, '', 65, '','Winter','')
    sit_rg = np.nanmean(sit_rg, axis=1)
    sit_rg = np.nanmean(sit_rg, axis=1)
    print((sit_rg))    
    
    # plt.figure(figsize=[20,12])
    # plt.plot(yrs,sit_rg)
    # plt.xlabel('Years', fontsize=25)
    # plt.ylabel('thickness (m)', fontsize=25)
    # plt.xticks(yrs, np.arange(1988,2017,1), fontsize=20, rotation=75)
    # plt.yticks(fontsize=20)
    # plt.title('Mean Yearly Winter SIT: Region 2')
    
    return sit_rg    

def retrieve_ease_grid(grid_id, grid_name, wind=0):
    '''
    Returns the target grid using projection library 
    To find area_extent transform lat of interest with lon -45 using projection:
        p = Proj("+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs", preserve_units=False)
    '''
    area_id = grid_id
    name = grid_name
    proj_id = grid_id
    x_size = 93
    y_size = 93
    area_extent = (-2746911.7958511068, -2746911.7958511068, 2746911.7958511068, 2746911.7958511068) #(-2322128.165220568, -2322128.165220568, 2322128.165220568, 2322128.165220568) #(-2212704.2634010427, -2212704.2634010427, 2212704.2634010427, 2212704.2634010427) #(-7399581.862216598, -7399581.862216598, 7399581.862216598, 7399581.862216598) # for a bounding lat of 60deg but used for 70? (-2212704.2634010427, -2212704.2634010427, 2212704.2634010427, 2212704.2634010427)#(-2212704.2634010427, -2212704.2634010427, 2212704.2634010427, 2212704.2634010427)
    proj_dict = {'a': '6371228.0', 'units':'m', 'lon_O':'0', 'proj':'laea', 'lat_0':'+90'}
    projection = "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +a=6371228 +b=6371228 +units=m +no_defs"
    targ_def = pyresample.geometry.AreaDefinition(area_id, name, proj_id, projection, x_size, y_size, area_extent)
    return targ_def
    
def interpolate(data, lon_o, lat_o):
    '''
    Interpolates data for northern hemisphere Arctic EASE coordinate grid
    Retrieves grid using retrieve_ease_grid function. Data is then interpolated 
    for every time index using basemap cubic spline interpolation (order=3)
    '''
    targ_def = retrieve_ease_grid('ease_nh', 'Arctic EASE grid')
    lons_targ, lats_targ = targ_def.get_lonlats()   
    # print(lons_targ[0,20], lats_targ[0,20])
    data_int = np.zeros((len(data[:,0,0]), len(lats_targ[:,0]), len(lons_targ[0,:])))

    for i in range(len(data[:,0,0])):
        data_int[i,:,:] = bm.interp(data[i,:,:], lon_o, lat_o, lons_targ, lats_targ, checkbounds=False, masked=False, order =3)
    
    return data_int, lons_targ, lats_targ

def return_ease(fn, var, region, region2=0, sic=0):
    if region2==0:
        ts = nc_read(fn, 'time')
        ts = ts[9:38]
        _lat = nc_read(fn, 'lat')
    
        lon = nc_read(fn, 'lon')
        
        data = var_fix(fn, var, sic)
        data = data[9:38,:,:]
        lat = _lat[::-1]
        lons, lats = np.meshgrid(lon, lat)

        intp_data, lons_intp, lats_intp = interpolate(data, lon, lat)
        plot_piomas(intp_data[0,:,:], lats_intp, lons_intp, 0, '', 65, 'plot_title', 'path_out', 'season')

        var_ts = reduce_field(intp_data, lons_intp, lats_intp, ts, region)
    
    if region2!=0:
        ts = nc_read(fn, 'time')
        ts = ts[9:38]
        _lat = nc_read(fn, 'lat')
    
        lon = nc_read(fn, 'lon')
        data = var_fix(fn, var, sic)
        data = data[9:38,:,:]
        lat = _lat[::-1]
        lons, lats = np.meshgrid(lon, lat)

        intp_data, lons_intp, lats_intp = interpolate(data, lon, lat)
        var_ts = reduce_field(intp_data, lons_intp, lats_intp, ts, region, joined=region2)
    
    return var_ts

def piomas_data(poly_verts, region2=0):
    fn_sit = '/Users/marialozano/Python/PIOMAS_data/SIT/PIOMAS_SIT_1979-2020_EASE_65N.nc'
    fn_ofx = '/Users/marialozano/Python/PIOMAS_data/OceanFlux/PIOMAS_OFLX_1979-2016_EASE_65N.nc'
    fn_sia = '/Users/marialozano/Python/PIOMAS_data/SIA/PIOMAS_Advection_1979-2016_EASE_65N.nc'
    fn_sig = '/Users/marialozano/Python/PIOMAS_data/SIGrowth/PIOMAS_SIG_1979-2016_EASE_65N.nc'
    fn_sic = '/Users/marialozano/Python/PIOMAS_data/SIC/PIOMAS_SIC_1979-2016_EASE_65N.nc'
    
    sit = nc_read(fn_sit, 'sit')
    sia = nc_read(fn_sia, 'advect')
    sig = nc_read(fn_sig, 'sig')
    sic = nc_read(fn_sic, 'sic')
    ofx = nc_read(fn_ofx, 'oflx')
    yrs = nc_read(fn_sia,'year')
    lat = nc_read(fn_sia, 'lat')
    lon = nc_read(fn_sia, 'lon')
    lon = lon*(-1)
    lat_t = nc_read(fn_sit, 'lat')
    lon_t = nc_read(fn_sit, 'lon')
    

    yrs = yrs[9:]
    sit_w = sit[9:38,[0,1,2],:,:]
    sit_w = np.nanmean(sit_w, axis=1)
    sia_w = sia[9:, [0,1,2],:,:]
    sia_w = np.nanmean(sia_w, axis=1)
    ofx_w = ofx[9:, [0,1,2],:,:]
    ofx_w = np.nanmean(ofx_w, axis=1)
    sic_w = sic[9:, [0,1,2],:,:]
    sic_w = np.nanmean(sic_w, axis=1)
    sig_w = sig[9:, [0,1,2],:,:]
    sig_w = np.nanmean(sig_w, axis=1)
    
    if region2==0:
        sit_ts = reduce_field(sit_w, lon_t, lat_t, yrs, poly_verts)
        sia_ts = reduce_field(sia_w, lon, lat, yrs, poly_verts)
        sig_ts = reduce_field(sig_w, lon, lat, yrs, poly_verts)
        ofx_ts = reduce_field(ofx_w, lon, lat, yrs, poly_verts)
        sic_ts = reduce_field(sic_w, lon, lat, yrs, poly_verts)
    elif region2!=0:
        sit_ts = reduce_field(sit_w, lon_t, lat_t, yrs, poly_verts, joined=region2)
        sia_ts = reduce_field(sia_w, lon, lat, yrs, poly_verts, joined=region2)
        sig_ts = reduce_field(sig_w, lon, lat, yrs, poly_verts, joined=region2)
        ofx_ts = reduce_field(ofx_w, lon, lat, yrs, poly_verts, joined=region2)
        sic_ts = reduce_field(sic_w, lon, lat, yrs, poly_verts, joined=region2)

    return yrs, sia_ts, sig_ts, sit_ts, ofx_ts, sic_ts

def ncep_ncar_data(region, region2=0):
    fn = '/Users/marialozano/Python/ERA5_data/HR/era5_mm-JFM_1979_2020-from_3hrs.nc'
    slp_ts = return_ease(fn, 'slp', region, region2, sic=0)
    sat_ts = return_ease(fn, 't2m', region, region2, sic=0)
    u10_ts = return_ease(fn, 'u10', region, region2, sic=0)
    v10_ts = return_ease(fn, 'v10', region, region2, sic=0)
    # sat_ts=0
    # u10_ts=0
    # v10_ts=0
    return slp_ts, sat_ts, u10_ts, v10_ts

def make_csv(i, region, region2=0):
    yrs, sia, sig, sit, ofx, sic = piomas_data(region, region2)
    slp,sat,u10,v10 = ncep_ncar_data(region, region2)
    # For region2 [(-20,70),(-20,82),(1,82),(1,70)]
    # For region4 [(-135,70),(-120,85),(-180,85),(-180,70)]
    d = {'yrs':yrs,
          'sit':sit,
          'sia':sia,
          'sig':sig,
          'sic':sic,
          'sat':sat,
          'slp':slp,
          'u10':u10,
          'v10':v10,
          }
    df = pd.DataFrame(data=d, columns = ['yrs','sit','sia','sig','sic','sat','slp','u10','v10'])
    df.to_csv(r'/Users/marialozano/Python/STA2101/JFM_TS_VARS_R{}.csv'.format(i), index=True, header=True)
    
    
def main():
    fn_piomas = '/Users/marialozano/Python/PIOMAS_data/SIT/PIOMAS_SIT_1979-2020_EASE_65N.nc'
    
    sit = nc_read(fn_piomas, 'sit')
    yrs = nc_read(fn_piomas,'year')
    lat = nc_read(fn_piomas, 'lat')
    lon = nc_read(fn_piomas, 'lon')
    sit[sit==0] = np.nan
    

    sit_w = sit[:38,[0,1,2],:,:]
    sit_s = sit[:,6:9,:,:]
    
    SIV_plot(sit)
    trend_SIT_plot(sit_w[:38,:,:,:], yrs[:38], 'JFM')
    # trend_SIT_plot(sit_s, yrs, 'JAS')
    
    sit_w = sit_w[9:,:,:,:]
    trend_map(sit_w, lat, lon, yrs[9:38], 'Winter')
    # trend_map(sit_s, lat, lon, yrs[:], 'Summer')
    
    
    sit_w = np.nanmean(sit_w[9:,:,:,:], axis=1)
    # sit_s = np.nanmean(sit_s, axis=1)

    poly_verts = [[(35,70), (35,82), (99,77), (66,69)], 
                  [(0,70),(0,82),(35,82),(35,72)],[(340,70), (340,82), (361,82), (361,70)], 
                  [(215,74),(215,82),(355,89),(345,82)],
                  [(180,70),(174,85),(240,85),(225,70)],
                  [(112,72),(80,82),(175,81),(166,66)]]

    
    make_csv(0, poly_verts[0])
                
    
if __name__ == '__main__':
    main()  