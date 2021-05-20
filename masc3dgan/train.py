import gc
import os
import sys

import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

import data
import eval
import gan
import models


def setup_batch_gen(train_dir, valid_dir, batch_size=32):
    batch_gen_train = data.BatchGenerator(train_dir, batch_size=batch_size,
        size_div=0.002, grid_div=0.12, grid_empty_val=-0.2)
    batch_gen_valid = data.BatchGenerator(valid_dir, batch_size=batch_size,
        size_div=0.002, grid_div=0.12, grid_empty_val=-0.2)

    return (batch_gen_train, batch_gen_valid)


def setup_gan(train_dir, valid_dir, batch_size=32, lr_disc=0.0001, lr_gen=0.0001):

    gen = models.generator()
    disc = models.discriminator()
    wgan = gan.ConditionalWGANGP(gen, disc, lr_disc=lr_disc, lr_gen=lr_gen)
    
    (batch_gen_train, batch_gen_valid) = setup_batch_gen(
        train_dir, valid_dir, batch_size=batch_size)

    gc.collect()

    return (wgan, batch_gen_train, batch_gen_valid)


def train_gan(wgan, batch_gen_train, batch_gen_valid, steps_per_epoch,
    num_epochs, training_ratio=1):

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1,num_epochs))
        wgan.train(batch_gen_train, num_gen_batches=steps_per_epoch,
            training_ratio=training_ratio)
        out_fn = os.path.dirname(os.path.abspath(__file__)) + \
            "/../data/progress.npz"
        data.save_examples(wgan.gen, batch_gen_valid,
            out_fn=out_fn)


def setup_predictor(train_dir, valid_dir, batch_size=32):
    pred = models.predictor()

    (batch_gen_train, batch_gen_valid) = setup_batch_gen(
        train_dir, valid_dir, batch_size=batch_size)

    return (pred, batch_gen_train, batch_gen_valid)


def train_predictor(pred, batch_gen_train, batch_gen_valid,
    steps_per_epoch, num_epochs, validation_steps=0):

    def batches(batch_gen):
        while True:
            (proj, log_proj_size, grid_3d, log_grid3d_size) = \
                next(batch_gen)
            grid_3d_size = np.exp(log_grid3d_size)
            mass = eval.mass(grid_3d[...,0].clip(min=0.0)*0.12, 
                grid_3d_size[:,0]*0.002)
            log_mass = np.log(mass)
            yield ([proj, log_proj_size], [log_mass, log_grid3d_size])

    batches_train = batches(batch_gen_train)
    batches_valid = batches(batch_gen_valid)

    for epoch in range(num_epochs):
        batch_gen_valid.reset(random_seed=1234)
        pred.fit(batches_train, epochs=1,
            steps_per_epoch=steps_per_epoch,
            validation_data=batches_valid, validation_steps=validation_steps)


def setup_triplet_validator(train_dir, valid_dir, batch_size=32):
    validator = models.triplet_validator()

    (batch_gen_train, batch_gen_valid) = setup_batch_gen(
        train_dir, valid_dir, batch_size=batch_size)

    return (validator, batch_gen_train, batch_gen_valid)


def train_triplet_validator(validator, batch_gen_train, batch_gen_valid,
    steps_per_epoch, num_epochs, validation_steps=0):

    def batches(batch_gen):
        while True:
            (proj, log_proj_size, grid_3d, log_grid3d_size) = \
                next(batch_gen)
            
            swap = np.random.rand(proj.shape[0]) > 0.5
            num_swapped = np.count_nonzero(swap)
            source_ind = np.arange(proj.shape[0])
            dest_ind = source_ind.copy()
            source_ind_swap = source_ind[swap].copy()
            while (source_ind_swap==source_ind[swap]).any():
                np.random.shuffle(source_ind_swap)
            source_ind[swap] = source_ind_swap
            
            source_cam = np.random.randint(3, size=proj.shape[0])
            dest_cam = source_cam.copy()
            source_cam = source_cam[source_ind]

            proj_swapped = proj.copy()
            proj_swapped[dest_ind,:,:,dest_cam] = \
                proj[source_ind,:,:,source_cam]
            valid_triplet = (~swap).astype(np.float32)

            yield (proj_swapped, valid_triplet)

    batches_train = batches(batch_gen_train)
    batches_valid = batches(batch_gen_valid)

    for epoch in range(num_epochs):
        batch_gen_valid.reset(random_seed=1234)
        validator.fit(batches_train, epochs=1,
            steps_per_epoch=steps_per_epoch,
            validation_data=batches_valid, validation_steps=validation_steps)


if __name__ == "__main__":
    import sys
    train_dir = sys.argv[1]
    valid_dir = sys.argv[2]
    (wgan, batch_gen_train, batch_gen_valid) = setup_gan(
        train_dir, valid_dir)
    while True: 
        train_gan(wgan, batch_gen_train, batch_gen_valid, 400, 1, training_ratio=5) 
        batch_gen_valid.reset(random_seed=1234) 
        eval.eval_grid(batch_gen_valid, wgan.gen) 
        wgan.save(os.path.dirname(os.path.abspath(__file__)) +
            "/../models/masc3dgan")
