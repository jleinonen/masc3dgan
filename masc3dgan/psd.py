from datetime import datetime, timedelta
import os

import netCDF4
import numpy as np
import pandas as pd
from scipy.integrate import trapz


def read_drop_file(fn):
	file_datestamp = os.path.basename(fn)[1:6]
	df = pd.read_csv(fn, sep=' ', parse_dates=[0],
		date_parser=lambda s: datetime.strptime(
			file_datestamp+' '+s,'%y%j %H:%M:%S:%f'))

	return df


def psd_for_drops(width, height, velocity, 
	D_range=(1e-4, 10e-2), num_points=128, vel_threshold=1e-3,
	min_drops=50):

	size = np.vstack((width,height)).max(axis=0)
	N = len(size)
	if N < min_drops:
		return {"N": N}

	log_size = np.log(size)

	log_D = np.linspace(np.log(D_range[0]), np.log(D_range[1]),
		num_points)

	v = log_size.var() * (N**(-1./(1+4))) # Scott's rule for KDE
	flux_spectrum = np.zeros_like(log_D)
	vel_by_size = np.zeros_like(log_D)
	for i in range(N):
		kernel = np.exp(-0.5*(log_size[i]-log_D)**2/v)
		kernel /= kernel.sum()
		flux_spectrum += kernel
		vel_by_size += kernel*velocity[i]

	vel_by_size /= flux_spectrum
	vel_by_size[flux_spectrum < flux_spectrum.max()*vel_threshold] = np.nan
	flux_spectrum *= N/trapz(flux_spectrum,log_D)

	return {
		"log_size": log_D,
		"flux_spectrum": flux_spectrum,
		"vel_by_size": vel_by_size,
		"N": N
	}


def psd_for_day(df, dt=timedelta(minutes=10), min_drops=50):
	start_time = pd.Timestamp(df["timestamp"][0].date())
	end_time = start_time + timedelta(days=1)

	psd_index = {}
	time = start_time
	while time < end_time:
		drops = df[(time <= df["timestamp"]) & (df["timestamp"] < time+dt)]

		widthA = np.array(drops["widthA"])*1e-3
		heightA = np.array(drops["heightA"])*1e-3
		velocity = np.array(drops["velocity"])

		psd = psd_for_drops(widthA,heightA,velocity,min_drops=min_drops)
		if psd["N"] >= min_drops:
			psd_index[(time,dt)] = psd
		time += dt

	return psd_index


def psd_for_all(dir):
	psd_index = {}
	for fn in sorted(os.listdir(dir)):
		print(fn)
		df = read_drop_file(os.path.join(dir,fn))
		if len(df) == 0:
			continue
		psd_day = psd_for_day(df)
		psd_index.update(psd_day)

	return psd_index


def mass_size_rel(mass,size,log_D,mass_threshold=1e-3):
	N = len(mass)
	log_mass = np.log(mass)
	log_size = np.log(size)

	p = np.polyfit(log_size, log_mass, 1)
	(alpha, beta) = (np.exp(p[1]), p[0])
	mass_by_size = np.exp(p[0]*log_D+p[1])

	return {
		"N_masc": N,
		"mass_by_size": mass_by_size, 
		"masc_size": size,
		"masc_mass": mass,
		"m-D-fit": (alpha, beta)
	}


def psd_mass_all(psd, masc_triplets_fn, masc_grid_fn, bfr_size=256,
	min_flakes=10):

	with netCDF4.Dataset(masc_grid_fn, 'r') as ds:
		mass = []
		for k in range(0,ds["mass_3d"].shape[0],bfr_size):
			mass_3d = np.array(ds["mass_3d"][k:k+bfr_size], copy=False)
			mass.append(mass_3d.sum((1,2,3)))
	mass = np.concatenate(mass)

	with netCDF4.Dataset(masc_triplets_fn, 'r') as ds:
		time = np.array(ds["time"][:], copy=False)
		proj_size = np.array(ds["proj_size"][:,0], copy=False)
	time = [datetime(1970,1,1)+timedelta(seconds=int(t)) for t in time]

	psd_mass = {}
	keys = sorted(psd.keys())

	mass_by_size_ref = mass_size_rel(mass, proj_size,
		psd[keys[0]]["log_size"])["mass_by_size"]

	for k in keys:
		(start_time, dt) = k
		end_time = start_time+dt
		in_timespan = np.array([(start_time <= t < end_time) for t in time])
		if np.count_nonzero(in_timespan) < min_flakes:
			continue
		
		m = mass[in_timespan]
		s = proj_size[in_timespan]
		psd_mass[k] = mass_size_rel(m,s,psd[k]["log_size"])
		psd_mass[k]["snow_rate"] = snow_rate_2DVD(psd[k]["log_size"],
			psd[k]["flux_spectrum"], psd_mass[k]["mass_by_size"], dt=dt)
		psd_mass[k]["snow_rate_ref"] = snow_rate_2DVD(psd[k]["log_size"],
			psd[k]["flux_spectrum"], mass_by_size_ref, dt=dt)

	return psd_mass

rho_w = 1000.0
def snow_rate_2DVD(log_D, flux_spectrum, mass_by_size,
	dt=timedelta(minutes=10)):

	dt = dt.total_seconds()

	D = np.exp(log_D)
	collection_width = 100e-3
	collection_area_by_size = (collection_width-D/2)**2

	return trapz(mass_by_size/collection_area_by_size * flux_spectrum,
		log_D) / (rho_w*dt)


def read_pluvio_file(fn):
	df = pd.read_csv(fn, delim_whitespace=True, parse_dates=[1],
		date_parser=lambda s: datetime.strptime(s,'%Y%m%d%H%M'),
		na_values=["PPP","NP","/"], index_col=1)

	return df


def pluvio_snow_rates(psd_mass, pluvio_dir):
	pluvio_files = {}

	keys = sorted(psd_mass.keys())
	snow_rates = {}

	for k in keys:
		(start_time, dt) = k
		month = start_time.strftime("%Y%m")
		if month not in pluvio_files:
			fn_pluvio = os.path.join(pluvio_dir,
				"MFOWFJ_VMSW42_{}.m".format(month))
			try:
				pluvio_files[month] = read_pluvio_file(fn_pluvio)
			except:
				pluvio_files[month] = None

		if pluvio_files[month] is None:
			continue

		snow_accumul = pluvio_files[month]["ott003t0"]
		start_S = snow_accumul[start_time] * 1e-3
		end_S = snow_accumul[start_time+dt] * 1e-3
		S = (end_S-start_S)/dt.total_seconds()
		print(start_time, S)
		snow_rates[k] = S

	return snow_rates