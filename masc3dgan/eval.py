import os
try:
    import netCDF4
    from skimage import measure, transform
except ModuleNotFoundError:
    pass
import numpy as np



rho_ice = 916.7


def mass(grid_3d, grid_3d_size):
    grid_3d = grid_3d.clip(min=0.0)
    d_cell = grid_3d_size / grid_3d.shape[0]
    return d_cell**3 * rho_ice * grid_3d.sum(axis=(-3,-2,-1))


def power_spectrum(grid_3d, grid_3d_size):
    grid_3d = grid_3d.clip(min=0.0)
    d_cell = grid_3d_size / grid_3d.shape[0]
    mass_density = d_cell**3 * rho_ice * grid_3d
    grid_reduced = mass_density.sum(axis=(-3,-2))
    dft = np.fft.fft(grid_reduced, axis=-1)
    power = abs(dft)**2
    freq = np.vstack([np.fft.fftfreq(dft.shape[0], d) for d in d_cell])
    power = power[:,:freq.shape[1]//2]
    freq = freq[:,:freq.shape[1]//2]
    
    return (freq, power)


def vertical_projected_area(grid_3d, grid_3d_size):
    d_cell = grid_3d_size / grid_3d.shape[0]
    nonzero = (grid_3d > 0).any(axis=-1)
    area = nonzero.sum(axis=(-2,-1)) * d_cell**2
    return area


def principal_axes(grid_3d, grid_3d_size):
    (x,y,z) = np.mgrid[
        :grid_3d.shape[-3],:grid_3d.shape[-2],:grid_3d.shape[-1]
    ]
    grid_3d = grid_3d.clip(min=0.0)

    def pc(g3d, g3d_size):
        d_cell = g3d_size / grid_3d.shape[0]
        gsum = g3d.sum()
        xc = (x*g3d).sum()/gsum
        yc = (y*g3d).sum()/gsum
        zc = (z*g3d).sum()/gsum
        cxx = ((x-xc)**2 * g3d).sum()/gsum
        cyy = ((y-zc)**2 * g3d).sum()/gsum
        czz = ((z-zc)**2 * g3d).sum()/gsum
        cxy = ((x-xc)*(y-yc) * g3d).sum()/gsum
        cxz = ((x-xc)*(z-zc) * g3d).sum()/gsum
        cyz = ((y-xc)*(z-zc) * g3d).sum()/gsum
        cov = np.array([
            [cxx, cxy, cxz],
            [cxy, cyy, cyz],
            [cxz, cyz, czz]
        ])
        (l,v) = np.linalg.eigh(cov)
        return v * np.sqrt(l) * d_cell

    pa = np.zeros((grid_3d.shape[0],3,3))
    for k in range(grid_3d.shape[0]):
        pa[k,...] = pc(grid_3d[k,...], grid_3d_size[k])

    return pa


def axis_ratio(grid_3d, grid_3d_size):
    pa = principal_axes(grid_3d, grid_3d_size)
    axis_length = np.sqrt((pa**2).sum(axis=-2))
    return axis_length[:,-1]/axis_length[:,0]


def orientation_angle(grid_3d, grid_3d_size, grid_3d_gen, grid_3d_size_gen):
    longaxis = principal_axes(grid_3d, grid_3d_size)[:,:,-1]
    longaxis_gen = principal_axes(grid_3d_gen, grid_3d_size_gen)[:,:,-1]
    longaxis_len = np.sqrt((longaxis**2).sum(axis=1))
    longaxis_gen_len = np.sqrt((longaxis_gen**2).sum(axis=1))

    # The axis vector might be in the opposite direction
    # Using abs here fixes that
    cos_angle = abs((longaxis*longaxis_gen).sum(axis=1) / \
        (longaxis_len*longaxis_gen_len))
    return np.arccos(cos_angle)
    

def eval_mass(batch_gen, gen, num_batches=64, noise_dim=64):
    m = []
    mg = []
    w = []
    wg = []
    gs = []
    gsg = []

    for i in range(num_batches):
        (proj, log_proj_size, grid_3d, log_grid_3d_size) = next(batch_gen)
        proj_size = np.exp(log_proj_size)
        grid_3d_size = np.exp(log_grid_3d_size)
        noise = np.random.randn(proj.shape[0],noise_dim)
        (grid_3d_gen, log_grid_3d_size_gen) = gen.predict([
            proj, log_proj_size, noise])
        grid_3d_size_gen = np.exp(log_grid_3d_size_gen)
        m.append(
            mass(grid_3d[...,0].clip(min=0.0)*0.12, 
                grid_3d_size[:,0]*0.002)
        )
        mg.append(
            mass(grid_3d_gen[...,0].clip(min=0.0)*0.12, 
                grid_3d_size_gen[:,0]*0.002)
        )

        w.append(grid_3d.clip(min=0.0).sum(axis=(-4,-3,-2,-1)))
        wg.append(grid_3d_gen.clip(min=0.0).sum(axis=(-4,-3,-2,-1)))

        gs.append(grid_3d_size)
        gsg.append(grid_3d_size_gen)

    m = np.concatenate(m, axis=0)
    mg = np.concatenate(mg, axis=0)
    w = np.concatenate(w, axis=0)
    wg = np.concatenate(wg, axis=0)
    gs = np.concatenate(gs, axis=0)
    gsg = np.concatenate(gsg, axis=0)

    log_m = np.log(m)
    log_mg = np.log(mg)
    log_w = np.log(w)
    log_wg = np.log(wg)
    log_gs = np.log(gs)
    log_gsg = np.log(gsg)

    print("Avg. size diff.: {:.3f}".format((gsg-gs).mean()))
    print("Avg. log size diff.: {:.3f}".format((log_gsg-log_gs).mean()))
    print("Log size RMSE: {:.3e}".format(np.sqrt(((log_gsg-log_gs)**2).mean())))
    print("Avg. grid sum diff.: {:.3f}".format((wg-w).mean()))
    print("Avg. log grid sum diff.: {:.3f}".format((log_wg-log_w).mean()))
    print("Log grid sum RMSE: {:.3e}".format(np.sqrt(((log_wg-log_w)**2).mean())))
    print("Avg. mass diff.: {:.3e}".format((mg-m).mean()))
    print("Avg. log mass diff.: {:.3f}".format((log_mg-log_m).mean()))
    print("Log mass RMSE: {:.3e}".format(np.sqrt(((log_mg-log_m)**2).mean())))


def eval_grid(batch_gen, gen, num_batches=64, noise_dim=64):
    grid_mse = []
    grid_mae = []
    grid_overlap = []

    for i in range(num_batches):
        (proj, log_proj_size, grid_3d, log_grid_3d_size) = next(batch_gen)
        noise = np.random.randn(proj.shape[0],noise_dim)
        grid_3d_gen = gen.predict([proj, noise])

        grid_3d.clip(min=0.0, out=grid_3d)
        grid_3d_gen.clip(min=0.0, out=grid_3d_gen)

        mask_3d = grid_3d > 0
        mask_3d_gen = grid_3d_gen > 0
        overlap = np.count_nonzero((mask_3d&mask_3d_gen), axis=(1,2,3,4)) / \
            np.count_nonzero((mask_3d|mask_3d_gen), axis=(1,2,3,4))
        grid_diff = (grid_3d-grid_3d_gen)

        grid_mse.append((grid_diff**2).mean(axis=(1,2,3,4)))
        grid_mae.append(abs(grid_diff).mean(axis=(1,2,3,4)))
        grid_overlap.append(overlap)

    grid_mse = np.concatenate(grid_mse)
    grid_mae = np.concatenate(grid_mae)
    grid_overlap = np.concatenate(grid_overlap)

    print("Grid RMSE: {:.3e}".format(np.sqrt(grid_mse.mean())))
    print("Grid MAE: {:.3e}".format(grid_mae.mean()))
    print("Grid mean overlap: {:.3e}".format(grid_overlap.mean()))


def extract_name(name_enc):
    return "".join(b.decode() for b in 
        netCDF4.stringtochar(name_enc)[::4,0])


def group_images_by_cam(fn_list):
    names = []
    for fn in fn_list:
        with netCDF4.Dataset(fn, 'r') as ds:
            names.append(ds["names"][:])
    names = np.concatenate(names, axis=0)

    index = {}

    for i in range(names.shape[0]):
        name = extract_name(names[i,:])
        parts = name.split("_")
        date = "_".join(parts[:2])
        unique_id = int(parts[-3])
        cam = int(parts[-1])

        key = (date,unique_id)
        if key not in index:
            index[key] = {}
        index[key][cam] = i

    return index


def masc_triplets(fn):
    cam_index = group_images_by_cam([fn])
    cam_index = {k: cam_index[k] for k in cam_index if
        len(cam_index[k])==3}

    triplets = np.zeros((len(cam_index),128,128,3), dtype=np.uint8)
    proj_sizes = np.zeros((len(cam_index),1), dtype=np.float32)
    keys = list(cam_index.keys())
    keys.sort()

    with netCDF4.Dataset(fn, 'r') as ds:
        images = np.array(ds["images"][:], copy=False)

    try:
        for (i,k) in enumerate(keys):
            for j in range(3):
                indices = [cam_index[k][j] for j in range(3)]
                triplet = images[indices]
                triplet = triplet.reshape(triplet.shape[:-1])
                triplet = triplet.transpose(1,2,0)
                (triplet, proj_size) = process_triplet(triplet)
                triplets[i,...] = triplet
                proj_sizes[i,0] = proj_size
    finally:
        del images

    return (triplets, proj_sizes)


def masc_grids(gen, pred, fn_masc, fn_out, batch_size=64,
    noise_dim=64, grid_dim=32, size_div=0.002):

    with netCDF4.Dataset(fn_masc, 'r') as ds:
        proj = np.array(ds["images"][:], copy=False)
        proj_size = np.array(ds["proj_size"][:], copy=False)

    N = proj.shape[0]
    mass_3d = np.zeros((N,grid_dim,grid_dim,grid_dim), dtype=np.float32)
    grid_size = np.zeros(N, dtype=np.float32)

    for k in range(0,N,batch_size):
        print("{}/{}".format(k,N))
        p = proj[k:k+batch_size,...].astype(np.float32)
        ps = proj_size[k:k+batch_size,...] / size_div
        log_ps = np.log(ps)

        noise = np.random.rand(p.shape[0],noise_dim)

        grid_3d = gen.predict([p,noise])[...,0]
        grid_3d[grid_3d<0] = 0
        (log_mass, log_gs) = pred.predict([p,log_ps])
        mass = np.exp(log_mass[:,0])
        gs = np.exp(log_gs[:,0]) * size_div
        mass_ratio = mass/grid_3d.sum(axis=(1,2,3))
        grid_3d *= mass_ratio[:,None,None,None]        

        mass_3d[k:k+batch_size,...] = grid_3d
        grid_size[k:k+batch_size] = gs

    with netCDF4.Dataset(fn_out, 'w') as ds:
        dim_samples = ds.createDimension("dim_samples", N)
        dim_gridsize = ds.createDimension("dim_gridsize", grid_dim)
        var_params = {"zlib": True, "complevel": 9}

        def write_data(data, name, dims, **params):
            dtype = params.pop("dtype", np.float32)
            var = ds.createVariable(name, dtype, dims, **params)
            var[:] = data

        write_data(mass_3d, "mass_3d",
            ("dim_samples","dim_gridsize","dim_gridsize","dim_gridsize"),
            chunksizes=(1,grid_dim,grid_dim,grid_dim), **var_params)
        write_data(grid_size, "grid_size", ("dim_samples",), **var_params)
