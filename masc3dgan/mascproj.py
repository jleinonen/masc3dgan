from datetime import datetime, timedelta
import fnmatch
import os

try:
    import netCDF4
    from skimage import measure, transform
except ModuleNotFoundError:
    pass
import numpy as np
from scipy import io as sio


def files_in_dir_recursive(top, pattern="*", include_dir=True):
    for (root, dirs, files) in os.walk(top):
        match_files = (fn for fn in files if 
            fnmatch.fnmatchcase(fn, pattern))
        if include_dir:
            match_files = (os.path.join(root,fn) for fn in match_files)
        for fn in match_files:
            yield fn


def find_matched(data_dir, min_files=3):
    files = {}
    for fn_full in files_in_dir_recursive(
        data_dir, pattern="*_flake_*_cam_?.mat"):

        fn = fn_full.split("/")[-1]
        fn = ".".join(fn.split(".")[:-1])
        fn_parts = fn.split("_")
        cam = int(fn_parts[-1])
        flake_id = int(fn_parts[-3])
        timestamp = "_".join(fn.split("_")[:2])
        time = datetime.strptime(timestamp, "%Y.%m.%d_%H.%M.%S")

        key = (time,flake_id)
        if key not in files:
            files[key] = {}
        files[key][cam] = fn_full

    print(len(files))
    files = {k: files[k] for k in files if len(files[k])>=min_files}
    print(len(files))

    delete_keys = []
    for (i,k) in enumerate(files):
        if i%1000==0:
            print("{}/{}, {} deleted".format(i,len(files),len(delete_keys)))
        if any(not valid_file(files[k][c]) for c in files[k]):
            delete_keys.append(k)
    for k in delete_keys:
        del files[k]

    print(len(files))

    return files


def valid_file(fn, xhi_min=8.5, max_intens_min=0.03,
    melting_max=0, min_size=8, max_size=256):

    m = sio.loadmat(fn)

    xhi = m["roi"]["xhi"][0,0][0,0]
    if xhi < xhi_min:
        return False

    max_intens = m["roi"]["max_intens"][0,0][0,0]
    if max_intens < max_intens_min:
        return False

    melting_id = m["roi"]["melting_ID"][0,0][0,0]
    if melting_id > melting_max:
        return False

    shape = m["roi"]["data"][0,0].shape
    size = np.max(shape)
    if not (min_size <= size <= max_size):
        return False

    label_name = str(m["roi"]["label_name"][0,0][0])
    if label_name == 'smal':
        return False

    return True


def valid_triplet(triplet_files, min_size=16, max_size_var=1.5):
    mat = [sio.loadmat(triplet_files[i]) for i in range(3)]

    def get_size(m):
        shape = m["roi"]["data"][0,0].shape
        return np.max(shape)

    sizes = [get_size(m) for m in mat]
    largest = max(sizes)
    smallest = min(sizes)

    return (largest>=min_size) and (largest/smallest<=max_size_var)


def filter_triplets(files):
    return {k: files[k] for k in files if valid_triplet(files[k])}


def image_centroid(img, threshold=2):
    (i,j) = np.mgrid[:img.shape[0],:img.shape[1]]
    img_sil = (img >= threshold)
    mi = int(round(i[img_sil].mean()))
    mj = int(round(j[img_sil].mean()))
    return (mi, mj)


def place_at_center(source, dest):
    (mi, mj) = image_centroid(source)

    i0 = max(dest.shape[0]//2-mi,0)
    i1 = i0+source.shape[0]
    if i1 > dest.shape[0]:
        di = i1-dest.shape[0]
        i0 -= di
        i1 -= di
    j0 = max(dest.shape[1]//2-mj,0)
    j1 = j0+source.shape[1]
    if j1 > dest.shape[1]:
        dj = j1-dest.shape[1]
        j0 -= dj
        j1 -= dj

    dest[i0:i1,j0:j1] = source


def equalize_image_size(images):
    def size_needed(img):
        (mi, mj) = image_centroid(img)
        mi = int(round(mi))
        mj = int(round(mj))
        i_size = 2*max(mi,img.shape[0]-mi)+1
        j_size = 2*max(mj,img.shape[1]-mj)+1
        return max(i_size,j_size)

    sizes = [size_needed(img) for img in images]
    size = max(sizes)

    images_matched = np.zeros((size,size,len(images)),
        dtype=images[0].dtype)

    for (i,img) in enumerate(images):
        place_at_center(img, images_matched[...,i])

    return images_matched


def create_triplet_dataset(triplet_files, out_fn, img_size=128,
    num_cameras=3):

    N = len(triplet_files)
    images = np.zeros((N,img_size,img_size,3), dtype=np.uint8)
    proj_size = np.zeros((N,1), dtype=np.float32)
    particle_id = np.zeros(N, dtype=np.uint32)
    time = np.zeros(N, dtype=np.int64)

    for (i,k) in enumerate(sorted(triplet_files.keys())):
        if i%1000 == 0:
            print("{}/{}".format(i,len(triplet_files)))
        triplet = triplet_files[k]
        triplet = [sio.loadmat(triplet[i]) for i in range(3)]
        triplet = [m["roi"]["data"][0,0] for m in triplet]
        triplet = equalize_image_size(triplet)
        (triplet, ps) = process_triplet(triplet)
        images[i,...] = triplet
        proj_size[i,0] = ps
        time[i] = int((k[0]-datetime(1970,1,1)).total_seconds())
        particle_id[i] = k[1]

    with netCDF4.Dataset(out_fn, 'w') as ds:
        dim_samples = ds.createDimension("dim_samples", N)
        dim_imgsize = ds.createDimension("dim_imgsize", img_size)
        dim_cameras = ds.createDimension("dim_cameras", num_cameras)
        dim_one = ds.createDimension("dim_one", 1)
        var_params = {"zlib": True, "complevel": 9}

        def write_data(data, name, dims, **params):
            dtype = params.pop("dtype", np.float32)
            var = ds.createVariable(name, dtype, dims, **params)
            var[:] = data

        write_data(images, "images",
            ("dim_samples","dim_imgsize","dim_imgsize","dim_cameras"),
            chunksizes=(1,img_size,img_size,1), dtype=np.uint8, **var_params)
        write_data(proj_size, "proj_size",
            ("dim_samples","dim_one"), **var_params)
        write_data(time, "time", ("dim_samples",),
            dtype=np.int64, **var_params)
        write_data(particle_id, "particle_id", ("dim_samples",),
            dtype=np.uint32, **var_params)


def process_triplet(triplet, threshold=2, resolution=35e-6):
    size = triplet.shape[1]
    proj_size = size*resolution
    
    rescaled_triplet = np.zeros((128,128,3), np.uint8)
    for i in range(3):
        img = triplet[:,:,i].astype(np.float32)/255
        img = transform.resize(img, (128,128))
        img *= 255
        rescaled_triplet[:,:,i] = (img>=threshold).astype(np.uint8)

    return (rescaled_triplet, proj_size)


def process_all(masc_dir, out_fn):
    # this runs all the processing to create the masc images
    
    triplet_files = find_matched(masc_dir)
    triplet_files = filter_triplets(triplet_files)
    create_triplet_dataset(triplet_files, out_fn)
