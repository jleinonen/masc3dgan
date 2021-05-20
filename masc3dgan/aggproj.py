import numpy as np
from scipy.ndimage import convolve, zoom

from aggregation import riming, rotator


def aggregate(mono_size=650e-6, mono_min_size=100e-6,
    mono_max_size=3000e-6, mono_type="dendrite", grid_res=35e-6,
    num_monos=5, riming_lwp=0.0, riming_mode="subsequent",
    rime_pen_depth=120e-6, compact_dist=0.0):

    mono_generator = riming.gen_monomer(psd="exponential", size=mono_size, 
        min_size=mono_min_size, max_size=mono_max_size,
        mono_type=mono_type, grid_res=grid_res, rimed=True)
        
    agg = riming.generate_rimed_aggregate(mono_generator, N=num_monos,
        align=False, riming_lwp=riming_lwp, riming_mode=riming_mode,
        rime_pen_depth=rime_pen_depth, compact_dist=compact_dist)
    agg.rotate(rotator.UniformRotator())

    return agg


def projection(**kwargs):
    grid_size_3d = 32
    p_size = 256
    p_downscale = 2
    grid_res = 35e-6
    k = np.array([0.25,0.5,0.25])
    kernel_2d = k[None,:]*k[:,None]
    kernel_3d = k[None,None,:]*k[None,:,None]*k[:,None,None]
    agg = aggregate(**kwargs)

    cam_angles = np.array([
        -36.0, 0.0, 36.0, 54.0, 90.0, 126.0,
        144.0, 180.0, 216.0, 234.0, 270.0, 306.0
    ]) * (np.pi/180)
    
    # compute 2d projections
    def project(angle):
        p = agg.project_on_dim(direction=(angle, 0.0))
        (i,j) = np.mgrid[:p.shape[0],:p.shape[1]]
        i_c = int(round((i*p).sum()/p.sum()))
        j_c = int(round((j*p).sum()/p.sum()))
        i0 = p_size//2 - i_c
        i1 = i0+p.shape[0]
        j0 = p_size//2 - j_c
        j1 = j0+p.shape[1]
        if (i0 < 0) or (i1 >= p_size) or (j0 < 0) or (j1 > p_size):
            raise ValueError("Cannot fit projection in bounds.")
        p_eq = np.zeros((p_size,p_size), dtype=np.uint8)
        p_eq[i0:i1,j0:j1] = p
        return p_eq.astype(np.float32)

    proj = np.stack([project(a) for a in cam_angles], axis=-1)
    proj_any = (proj>0).any(axis=-1)
    (i,j) = np.mgrid[:proj.shape[0],:proj.shape[1]]
    i_act = i[proj_any]
    i_ext = max(proj.shape[0]//2-i_act.min(), i_act.max()-proj.shape[0]//2)
    j_act = j[proj_any]
    j_ext = max(proj.shape[1]//2-j_act.min(), j_act.max()-proj.shape[1]//2)
    proj_ext = max(i_ext,j_ext)+1
    proj_box = proj[
        proj.shape[0]//2-proj_ext:proj.shape[0]//2+proj_ext,
        proj.shape[1]//2-proj_ext:proj.shape[1]//2+proj_ext,
        :
    ]
    proj_size = proj_box.shape[0]*grid_res
    zoom_factor = p_size//p_downscale / proj_box.shape[0] + 1e-8
    proj = zoom(proj_box, (zoom_factor,zoom_factor,1), order=1)
    for k in range(proj.shape[-1]):
        proj[:,:,k] = convolve(proj[:,:,k], kernel_2d, mode='constant')
    proj[proj<0.5] = 0
    proj[proj>=0.5] = 1

    
    ext = (
        min(s0 for (s0,s1) in agg.extent)-grid_res*0.5,
        max(s1 for (s0,s1) in agg.extent)+grid_res*0.5
    )
    d_ext = (ext[1]-ext[0])/grid_size_3d
    ext = (ext[0]-d_ext, ext[1]+d_ext)
    grid_edges = np.linspace(ext[0], ext[1], grid_size_3d+1)
    grid_3d = np.zeros((grid_size_3d,grid_size_3d,grid_size_3d))
    cell_vol = (grid_edges[1]-grid_edges[0])**3

    for (i,gx0) in enumerate(grid_edges[:-1]):
        gx1 = grid_edges[i+1]
        for (j,gy0) in enumerate(grid_edges[:-1]):
            gy1 = grid_edges[j+1]
            for (k,gz0) in enumerate(grid_edges[:-1]):
                gz1 = grid_edges[k+1]
                X = agg.X
                in_cell = \
                    (gx0 <= X[:,0]) & (X[:,0] < gx1) & \
                    (gy0 <= X[:,1]) & (X[:,1] < gy1) & \
                    (gz0 <= X[:,2]) & (X[:,2] < gz1)

                cell_ice_vol = grid_res**3 * \
                    np.count_nonzero(in_cell)
                grid_3d[i,j,k] = cell_ice_vol / cell_vol
    grid_3d = convolve(grid_3d, kernel_3d, mode='constant').astype(np.float32)
    grid_3d_size = grid_edges[-1]-grid_edges[0]

    return (agg, proj, proj_size, grid_3d, grid_3d_size)


def create_random_snowflake():
    kwargs = {}
    kwargs["riming_lwp"] = np.random.choice(
        [0.0, 0.0, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0], 1)[0]
    kwargs["num_monos"] = max(1,int(round((7*np.random.rand())**2)))
    kwargs["mono_size"] = 200e-6 + 800e-6*np.random.rand()
    mono_types = ["dendrite", "dendrite", "dendrite", "needle"]
    if kwargs["num_monos"] < 5:
        mono_types += ["rosette", "plate", "column"]
    kwargs["mono_type"] = np.random.choice(mono_types,1)[0]
    kwargs["compact_dist"] = 0.62 * 35e-6 * np.random.rand()
    kwargs["riming_mode"] = np.random.choice(
        ["simultaneous", "subsequent"], 1)[0]

    print(kwargs)

    return (projection(**kwargs)+(kwargs,))
