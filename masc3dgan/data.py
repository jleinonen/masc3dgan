import os

import numpy as np
from scipy.ndimage import convolve


class BatchGenerator:
    def __init__(self, file_dir, batch_size=32, random_seed=None,
        size_div=1.0, grid_empty_val=0.0, grid_div=1.0):

        self.batch_size = batch_size
        self.size_div = size_div
        self.grid_empty_val = grid_empty_val
        self.grid_div = grid_div

        files = os.listdir(file_dir)
        files = [file_dir+"/"+fn for fn in files 
            if fn.startswith("snowflake_sample")]
        self.N = 0
        for fn in files:
            with np.load(fn) as p:
                self.N += p["proj"].shape[0]

        self.proj = np.zeros((self.N,128,128,12), dtype=np.uint8)
        self.proj_size = np.zeros((self.N,1), dtype=np.float32)
        self.grid_3d = np.zeros((self.N,32,32,32,1), dtype=np.float32)
        self.grid_3d_size = np.zeros((self.N,1), dtype=np.float32)

        k = 0
        for fn in files:
            with np.load(fn) as p:
                N = p["proj"].shape[0]
                self.proj[k:k+N,...] = p["proj"]
                self.proj_size[k:k+N,0] = p["proj_size"]
                self.grid_3d[k:k+N,...,0] = p["grid_3d"]
                self.grid_3d_size[k:k+N,0] = p["grid_3d_size"]
                k += N

        self.proj_size /= size_div
        self.grid_3d_size /= size_div
        self.grid_3d /= grid_div
        self.grid_3d[self.grid_3d<=0] = grid_empty_val

        self.blur_kernels = {}

        self.reset(random_seed=random_seed)     

    def reset(self, random_seed=None):
        self.prng = np.random.RandomState(seed=random_seed)
        self.next_ind = np.array([], dtype=int)

    def __iter__(self):
        return self

    def __next__(self):
        while len(self.next_ind) < self.batch_size:
            next_ind = np.arange(self.N)
            self.prng.shuffle(next_ind)
            self.next_ind = np.concatenate([
                self.next_ind, next_ind
            ])

        next_ind = self.next_ind[:self.batch_size]
        self.next_ind = self.next_ind[self.batch_size:]

        proj = self.proj[next_ind,...].copy()
        proj_sel = np.zeros((proj.shape[:-1]+(3,)), dtype=np.float32)
        proj_size = self.proj_size[next_ind,:].copy()
        grid_3d = self.grid_3d[next_ind,...].copy()
        grid_3d_size = self.grid_3d_size[next_ind,:].copy()

        for k in range(self.batch_size):
            (proj_sel[k,...], grid_3d[k,...]) = self.augment(
                proj[k,...], grid_3d[k,...], proj_size[k,0]*self.size_div)

        log_proj_size = np.log(proj_size)
        log_grid_3d_size = np.log(grid_3d_size)

        return (proj_sel, log_proj_size, grid_3d, log_grid_3d_size)

    def augment(self, proj, grid_3d, proj_size):
        num_rot = self.prng.randint(4)
        if num_rot > 0:
            grid_3d = np.rot90(grid_3d, k=-num_rot, axes=(0,1))
        proj = proj[...,num_rot*3:num_rot*3+3]

        flip_ud = bool(self.prng.randint(2))
        if flip_ud:
            grid_3d = grid_3d[:,:,::-1,:]
            proj = proj[:,::-1,:]
        flip_lr = bool(self.prng.randint(2))
        if flip_lr:
            grid_3d = grid_3d[:,::-1,:,:]
            proj = proj[::-1,:,::-1]

        for k in range(proj.shape[-1]):
            proj[...,k] = self.lens_blur(proj[...,k], proj_size)

        return (proj, grid_3d)

    def lens_blur(self, img, proj_size, max_pixels=2, threshold=0.02):
        pixel_size = proj_size/img.shape[0]
        d_pix = self.prng.rand()*max_pixels
        d = int(round(d_pix * 35e-6/pixel_size))
        if d <= 1:
            return img

        k = self.blur_kernels.get(d)
        if k is None:
            (x,y) = np.mgrid[:d,:d]
            mid = (d-1)*0.5
            r = d*0.5
            k = ((x-mid)**2+(y-mid)**2 <= r**2).astype(np.float32)
            k /= k.sum()
            self.blur_kernels[d] = k

        img = convolve(img, k, mode='constant')
        return (img > threshold).astype(img.dtype)
        

def save_examples(gen, batch_gen, out_fn, noise_dim=64):
    (proj, log_proj_size, grid_3d, log_grid_3d_size) = next(batch_gen)
    proj_size = np.exp(log_proj_size)
    grid_3d_size = np.exp(log_grid_3d_size)
    noise = np.random.randn(proj.shape[0],noise_dim)
    grid_3d_gen = gen.predict([proj, noise])
    kwargs = {
        "proj": proj,
        "proj_size": proj_size,
        "grid_3d": grid_3d,
        "grid_3d_size": grid_3d_size,
        "grid_3d_gen": grid_3d_gen
    }
    np.savez_compressed(out_fn, **kwargs)
