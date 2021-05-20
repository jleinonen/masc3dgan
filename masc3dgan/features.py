import os

import dask
import netCDF4
import numpy as np
from scipy.signal import convolve
from scipy.stats import gaussian_kde
from skimage import measure, transform
from sklearn.decomposition import PCA

import aggproj


def image_features(img, resolution=35e-6):
    perimeter = measure.perimeter(img, neighbourhood=8)
    try:
        regionprops = measure.regionprops(img)[0]
    except IndexError:
        raise ValueError("No regions found in image.")
    area = regionprops["area"]
    convex_area = regionprops["convex_area"]
    solidity = regionprops["solidity"]
    eq_area_diameter = regionprops["equivalent_diameter"]
    eq_area_perimeter = np.pi*eq_area_diameter
    perimeter_ratio = perimeter/eq_area_perimeter
    moments_hu = regionprops["moments_hu"]
    major_axis_length = regionprops["major_axis_length"]
    minor_axis_length = regionprops["minor_axis_length"]
    aspect_ratio = minor_axis_length/major_axis_length
    fractal_index = 2 * np.log(perimeter/4) / np.log(area)
    shape_index = perimeter / (4*np.sqrt(area))

    return {
        "aspect_ratio": aspect_ratio,
        "log_eq_area_diameter": np.log(eq_area_diameter*resolution),
        "fractal_index": fractal_index,
        "log_moments_hu_0": np.log(moments_hu[0]),
        "log_moments_hu_1": np.log(moments_hu[1]),
        "log_perimeter_ratio": np.log(perimeter_ratio),
        "log_shape_index": np.log(shape_index),
        "solidity": solidity
    }


def features_dataset(data_files, resolution=35e-6):
    ds_len = 0
    for fn in data_files:
        with netCDF4.Dataset(fn, 'r') as ds:
            ds_len += ds["images"].shape[0]
            img_shape = ds["images"].shape

    images = np.empty((ds_len,)+img_shape[1:], dtype=np.uint8)

    img_ind = 0
    for fn in data_files:
        with netCDF4.Dataset(fn, 'r') as ds:
            N_ds = ds["images"].shape[0]
            images[img_ind:img_ind+N_ds,...] = np.array(
                ds["images"][:], copy=False)
            img_ind += N_ds

    features = {
        "aspect_ratio": np.empty(ds_len),
        "log_eq_area_diameter": np.empty(ds_len),
        "fractal_index": np.empty(ds_len),
        "log_moments_hu_0": np.empty(ds_len),
        "log_moments_hu_1": np.empty(ds_len),
        "log_perimeter_ratio": np.empty(ds_len),
        "log_shape_index": np.empty(ds_len),
        "solidity": np.empty(ds_len)
    }

    for i in range(ds_len):
        if (i%1000==0):
            print("{}/{}".format(i,ds_len))
        img = images[i,:,:,0].copy()
        img[img>0] = 1

        regionprops = measure.regionprops(img)[0]
        (i0, j0, i1, j1) = regionprops["bbox"]
        i_size = i1-i0
        j_size = j1-j0
        size = max(i_size, j_size)
        if size != img.shape[0]:
            if i_size < j_size:
                i0 -= size//2 - i_size//2
                i0 = max(0,i0)
                i0 = min(img.shape[0]-size,i0)
                i1 = i0 + size
            elif j_size < i_size:
                j0 -= size//2 - j_size//2
                j0 = max(0,j0)
                j0 = min(img.shape[1]-size,j0)
                j1 = j0 + size
        size_ratio = size/img.shape[0]
        img_box = img[i0:i1,j0:j1]
        img = transform.resize(img_box.astype(np.float32), img.shape)
        img[img < 0.5] = 0
        img[img >= 0.5] = 1
        img = img.astype(np.uint8)
        
        feat = image_features(img, resolution=resolution*size_ratio)
        for k in features:
            features[k][i,...] = feat[k]

    return features


def features_model(proj, proj_size):
    feat_all = {}
    for k in range(proj.shape[0]):
        for m in range(proj.shape[3]):
            res = proj_size[k]/proj.shape[1]
            feat = image_features(proj[k,:,:,m], res)
            for f in feat:
                if f not in feat_all:
                    feat_all[f] = []
                feat_all[f].append(feat[f])

    for f in feat_all:
        feat_all[f] = np.array(feat_all[f])
    
    return feat_all


def pca_features(features, n_components=3):
    pca = PCA(n_components=n_components, whiten=True)
    f = np.vstack([features[k] for k in sorted(features.keys())]).T
    pca.fit(f)

    return pca


def grid_coordinates(components, comp_min, comp_max, grid_size=128):
    ijk = (components-comp_min)/(comp_max-comp_min) * grid_size
    ijk = ijk.astype(np.uint32)
    ijk.clip(max=grid_size-1, out=ijk)
    return ijk


def component_kde_grid(components, grid_size=128, comp_lim=None):
    if comp_lim is None:
        comp_min = components.min(axis=0)
        comp_max = components.max(axis=0)
    else:
        (comp_min, comp_max) = comp_lim

    comp_grid = np.zeros((grid_size,)*components.shape[1])
    ijk = grid_coordinates(components, comp_min, comp_max,
        grid_size=grid_size)
    for (i,j,k) in ijk:
        comp_grid[i,j,k] += 1
    sig_grid = components.std(axis=0)/(comp_max-comp_min) * grid_size
    # Scott's rule for KDE
    sig_grid *= components.shape[0]**(-1./(components.shape[1]+4))
    l = (4*sig_grid).round().astype(int)
    (ki, kj, kk) = np.mgrid[-l[0]:l[0]+1,-l[1]:l[1]+1,-l[2]:l[2]+1]
    k = np.exp(-0.5*
        ((ki/sig_grid[0])**2+(kj/sig_grid[1])**2+(kk/sig_grid[2])**2)
    )
    k /= k.sum()
    comp_grid = convolve(comp_grid, k, mode='same')
    return (comp_grid, comp_min, comp_max)


def sample_similar(feat_fn, model_file_dir, sample_fn, grid_size=128,
    random_seed=1234):

    with np.load(feat_fn) as features:
        pca = pca_features(features)
        feat = np.vstack([features[k] for k in sorted(features.keys())]).T
    components = pca.transform(feat)

    feat_model = []

    files = os.listdir(model_file_dir)
    files = sorted([fn for fn in files if fn.startswith("snowflake_batch")])
    for fn in files:
        with np.load(model_file_dir+"/"+fn) as ds:
            proj = ds["proj"]
            proj_size = ds["proj_size"]
            for k in range(proj.shape[0]):
                p = proj[k,:,:,1]
                f = image_features(p, 
                    resolution=proj_size[k]/proj.shape[1])
                f = np.array([f[k] for k in sorted(f.keys())])
                feat_model.append(f)

    feat_model = np.vstack(feat_model)
    comp_model = pca.transform(feat_model)

    (grid_obs, gmin, gmax) = component_kde_grid(components)
    (grid_model, gmin, gmax) = component_kde_grid(comp_model,
        comp_lim=(gmin,gmax))

    ijk = grid_coordinates(comp_model, gmin, gmax)
    accept_prob = np.array(
        [grid_obs[i,j,k]/grid_model[i,j,k] for (i,j,k) in ijk]
    )
    accept_prob /= np.percentile(accept_prob, 90)
    accept_prob.clip(0,1,out=accept_prob)
    prng = np.random.RandomState(random_seed)
    accept = prng.rand(comp_model.shape[0]) < accept_prob

    proj = []
    proj_size = []
    grid_3d = []
    grid_3d_size = []
    args = []
    k = 0
    for fn in files:
        with np.load(model_file_dir+"/"+fn, allow_pickle=True) as ds:
            N = ds["proj"].shape[0]
            print(k,N)
            acc = accept[k:k+N]
            proj.append(ds["proj"][acc,...])
            proj_size.append(ds["proj_size"][acc])
            grid_3d.append(ds["grid_3d"][acc,...])
            grid_3d_size.append(ds["grid_3d_size"][acc])
            args += [a for (i,a) in enumerate(ds["args"]) if acc[i]]
            k += N
    proj = np.concatenate(proj, axis=0)
    proj_size = np.concatenate(proj_size, axis=0)
    grid_3d = np.concatenate(grid_3d, axis=0)
    grid_3d_size = np.concatenate(grid_3d_size, axis=0)

    kwargs = {
        "proj": proj,
        "proj_size": proj_size,
        "grid_3d": grid_3d,
        "grid_3d_size": grid_3d_size,
        "args": args
    }
    np.savez_compressed(sample_fn, **kwargs)


@dask.delayed
def create_snowflake():
    while True:
        try:
            # keep trying until we have a snowflake that fits
            (agg, proj, proj_size, grid_3d, grid_3d_size, args) = \
                aggproj.create_random_snowflake()
        except ValueError:
            continue
        break
    return (proj, proj_size, grid_3d, grid_3d_size, args)


def create_snowflake_batch(N=256, n_proj=12):
    proj_batch = np.zeros((N,128,128,n_proj), dtype=np.uint8)
    proj_size_batch = np.zeros(N)
    grid_3d_batch = np.zeros((N,32,32,32), dtype=np.float32)
    grid_3d_size_batch = np.zeros(N)
    valid_batch = np.ones(N, dtype=bool)
    args_batch = []

    snowflakes = dask.compute(
        [create_snowflake() for i in range(N)],
        scheduler="multiprocessing"
    )[0]

    for i in range(N):
        (proj, proj_size, grid_3d, grid_3d_size, args) = snowflakes[i]
        proj = proj.astype(np.uint8)
        proj_batch[i,...] = proj
        proj_size_batch[i] = proj_size
        grid_3d_batch[i,...] = grid_3d
        grid_3d_size_batch[i] = grid_3d_size
        args_batch.append(args)
        for k in range(n_proj):
            try:
                feat = image_features(proj[...,k])
                feat_valid = np.isfinite([feat[k] for k in feat])
                if not feat_valid.all():
                    valid_batch[i] = False
                    break
            except ValueError:
                valid_batch[i] = False
                break

    kwargs = {
        "proj": proj_batch[valid_batch,...],
        "proj_size": proj_size_batch[valid_batch],
        "grid_3d": grid_3d_batch[valid_batch,...],
        "grid_3d_size": grid_3d_size_batch[valid_batch],
        "args": [a for (a,v) in zip(args_batch,valid_batch) if v]
    }

    for i in range(10000):
        fn = "../data/snowflake_batch_{:04d}.npz".format(i)
        if not os.path.exists(fn):
            np.savez_compressed(fn, **kwargs)
            break


if __name__=="__main__":
    create_snowflake_batch()
