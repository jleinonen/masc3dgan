from matplotlib import cm, colors, gridspec, pyplot as plt
from mayavi import mlab
import numpy as np
from scipy.ndimage import zoom


def plot_proj(proj, extent=None):
	plt.imshow(proj.T, origin='lower', cmap="gray",
		norm=colors.Normalize(0,1), aspect='equal', extent=extent)
	ax = plt.gca()
	ax.tick_params(left=False, bottom=False,
        labelleft=False, labelbottom=False)


def plot_3d_view(grid_3d, angle, contours=None):
	if contours is None:
		contours = 2**-np.arange(1,7, dtype=float)
	contour_colors = cm.viridis_r(np.linspace(0,1,len(contours)))
	alpha = 1.0
	mlab.figure(size=(512,512), bgcolor=(0.85,0.85,0.85))
	for (cont, color) in zip(contours, contour_colors):
		if not grid_3d[grid_3d>cont].any():
			continue
		surf = mlab.contour3d(
			zoom(grid_3d,4,order=1), contours=[cont,], 
			color=colors.to_rgb(color), opacity=alpha)
		surf.actor.property.specular = 0.1
		alpha *= 0.25
		alpha = max(alpha, 0.1)
	distance = mlab.view()[2]
	mlab.view(azimuth=angle, elevation=90, distance=0.75*distance)


def square_crop(img):
	if img.shape[0] < img.shape[1]:
		j0 = (img.shape[1]-img.shape[0])//2
		j1 = j0+img.shape[0]
		return img[:,j0:j1,...]
	elif img.shape[1] < img.shape[0]:
		i0 = (img.shape[0]-img.shape[1])//2
		i1 = i0+img.shape[1]
		return img[i0:i1,:,...]
	else:
		return img


def plot_particle(proj, grid_3d, proj_size, contours=None):
	plt.figure(figsize=(10,6))
	gs = gridspec.GridSpec(2,3,hspace=0.05,wspace=0.05)
	angles = np.array([-36.0, 0.0, 36.0])

	for (j,angle) in enumerate(angles):
		plt.subplot(gs[0,j])
		plot_proj(proj[:,:,j], extent=[0,proj_size*1e3,0,proj_size*1e3])
		if j==0:
			plt.gca().tick_params(left=True, labelleft=True)
			plt.ylabel("mm")

		plt.subplot(gs[1,j])
		plot_3d_view(grid_3d, angle, contours=contours)
		mlab.gcf().scene._lift()
		img = mlab.screenshot(antialiased=True)
		img = square_crop(img)
		mlab.close()
		plt.imshow(img, aspect='equal')
		ax = plt.gca()
		ax.set_axis_off()


def plot_3d_from_mass_grid(proj, mass_3d, proj_size, grid_size,
	n_contours=8):
	elem_size = grid_size/mass_3d.shape[0]
	elem_vol = elem_size**3
	rho_3d = mass_3d / elem_vol
	contours = 916.7 * 2**-np.arange(1,n_contours+1, dtype=float)
	plot_3d_view(rho_3d, 0.0, contours=contours)


def plot_particle_from_mass_grid(proj, mass_3d, proj_size, grid_size):
	elem_size = grid_size/mass_3d.shape[0]
	elem_vol = elem_size**3
	rho_3d = mass_3d / elem_vol
	contours = 916.7 * 2**-np.arange(1,9, dtype=float)
	plot_particle(proj, rho_3d, proj_size, contours=contours)
