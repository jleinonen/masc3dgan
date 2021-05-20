import json
import time
import gc
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Average, Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import generic_utils
from layers import GradientPenalty, RandomWeightedAverage, LogMass
from meta import Nontrainable, save_opt_weights, load_opt_weights


class ConditionalWGANGP(object):

    def __init__(self, gen, disc, load_fn_root=None,
        gradient_penalty_weight=10, disc_mag_weight=1e-4,
        lr_disc=0.0001, lr_gen=0.0001,
        proj_shape=(128,128), num_proj=3, grid3d_shape=(32,32,32),
        noise_dim=64):

        self.gen = gen
        self.disc = disc
        self.gradient_penalty_weight = gradient_penalty_weight
        self.disc_mag_weight = disc_mag_weight
        self.lr_disc = lr_disc
        self.lr_gen = lr_gen
        self.proj_shape = proj_shape
        self.num_proj = num_proj
        self.grid3d_shape = grid3d_shape
        self.noise_dim = noise_dim
        self.build()


    def filenames_from_root(self, root):
        fn = {
            "gen_weights": root+"-gen_weights.h5",
            "disc_weights": root+"-disc_weights.h5",
            "gen_opt_weights": root+"-gen_opt_weights.h5",
            "disc_opt_weights": root+"-disc_opt_weights.h5",
            "gan_params": root+"-gan_params.json"
        }
        return fn


    def load(self, load_files):
        self.gen.load_weights(load_files["gen_weights"])
        self.disc.load_weights(load_files["disc_weights"])
        
        self.disc.trainable = False
        self.gen_trainer._make_train_function()
        load_opt_weights(self.gen_trainer,
            load_files["gen_opt_weights"])
        self.disc.trainable = True
        self.gen.trainable = False
        self.disc_trainer._make_train_function()
        load_opt_weights(self.disc_trainer,
            load_files["disc_opt_weights"])
        self.gen.trainable = True


    def save(self, save_fn_root):
        paths = self.filenames_from_root(save_fn_root)
        self.gen.save_weights(paths["gen_weights"], overwrite=True)
        self.disc.save_weights(paths["disc_weights"], overwrite=True)
        save_opt_weights(self.disc_trainer, paths["disc_opt_weights"])
        save_opt_weights(self.gen_trainer, paths["gen_opt_weights"])
        params = {
            "gradient_penalty_weight": self.gradient_penalty_weight,
            "lr_disc": self.lr_disc,
            "lr_gen": self.lr_gen
        }
        with open(paths["gan_params"], 'w') as f:
            json.dump(params, f)


    def build(self):
        # Create optimizers
        self.opt_disc = Adam(self.lr_disc, beta_1=0.5, beta_2=0.9)
        self.opt_gen = Adam(self.lr_gen, beta_1=0.5, beta_2=0.9)

        # Create generator training network
        with Nontrainable(self.disc):
            proj = Input(shape=self.proj_shape+(self.num_proj,))
            noise = standard_normal_noise((self.noise_dim,))(proj)
            grid3d_gen = self.gen([proj, noise])
            disc_out_gen = self.disc([proj, grid3d_gen])
            self.gen_trainer = Model(inputs=proj, 
                outputs=disc_out_gen)

        # Create discriminator training network
        with Nontrainable(self.gen):
            proj = Input(shape=self.proj_shape+(self.num_proj,))
            grid3d = Input(shape=self.grid3d_shape+(1,))
            noise = standard_normal_noise((self.noise_dim,))(proj)
            
            grid3d_gen = self.gen([proj, noise])
            disc_out_gen = self.disc([proj, grid3d_gen])
            disc_out_real = self.disc([proj, grid3d])
            
            grid3d_avg = RandomWeightedAverage()([grid3d, grid3d_gen])
            disc_out_avg = self.disc([proj, grid3d_avg])
            disc_gp = GradientPenalty()([disc_out_avg, grid3d_avg])

            self.disc_trainer = Model(
                inputs=[proj, grid3d],
                outputs=[disc_out_real, disc_out_gen, disc_gp]#, disc_mag]
            )

        self.compile()

    def compile(self, opt_gen=None, opt_disc=None):
        if opt_gen is None:
            opt_gen = self.opt_gen
        if opt_disc is None:
            opt_disc = self.opt_disc

        with Nontrainable(self.disc):
            self.gen_trainer.compile(loss=wasserstein_loss,
                optimizer=opt_gen)

        with Nontrainable(self.gen):
            self.disc_trainer.compile(
                loss=[wasserstein_loss, wasserstein_loss, 'mse'],#, 'mse'], 
                loss_weights=[1.0, 1.0, 
                    self.gradient_penalty_weight],#, self.disc_mag_weight],
                optimizer=opt_disc
            )

    def train(self, batch_gen, num_gen_batches=1, 
        training_ratio=1, show_progress=True):

        disc_target_real = None
        if show_progress:
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(
                num_gen_batches*batch_gen.batch_size)

        for k in range(num_gen_batches):
        
            # train discriminator
            disc_loss = None
            disc_loss_n = 0
            with Nontrainable(self.gen):
                for rep in range(training_ratio):
                    # generate some real samples
                    (proj, log_proj_size, grid3d, log_grid3d_size) = \
                        next(batch_gen)

                    if disc_target_real is None: # on the first iteration
                        # run discriminator once just to find the shapes
                        disc_outputs = self.disc_trainer.predict(
                            [proj, grid3d])
                            #[Y_future, Y_past]+noise)
                        disc_target_real = np.ones(disc_outputs[0].shape,
                            dtype=np.float32)
                        disc_target_fake = -disc_target_real
                        gen_target = disc_target_real
                        gp_target = np.zeros(disc_outputs[2].shape, 
                            dtype=np.float32)
                        #mag_target = np.ones(disc_outputs[3].shape, 
                        #    dtype=np.float32)
                        del disc_outputs

                    dl = self.disc_trainer.train_on_batch(
                        [proj, grid3d],
                        [disc_target_real, disc_target_fake, gp_target]#, 
                            #mag_target]
                    )

                    if disc_loss is None:
                        disc_loss = np.array(dl)
                    else:
                        disc_loss += np.array(dl)
                    disc_loss_n += 1

                    del proj, log_proj_size, grid3d, log_grid3d_size

                disc_loss /= disc_loss_n

            # train generator
            with Nontrainable(self.disc):
                (proj, log_proj_size, grid3d, log_grid3d_size) = next(batch_gen)
                
                gen_loss = self.gen_trainer.train_on_batch(proj, gen_target)

                del proj, log_proj_size, grid3d, log_grid3d_size

            if show_progress:
                losses = []
                for (i,dl) in enumerate(disc_loss):
                    losses.append(("D{}".format(i), dl))
                #for (i,gl) in enumerate(gen_loss):
                #    losses.append(("G{}".format(i), gl))
                losses.append(("G0", gen_loss))
                progbar.add(batch_gen.batch_size, 
                    values=losses)

            gc.collect()


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred, axis=-1)


def standard_normal_noise(shape):
    def std_noise(dummy_input):
        batch_shape = K.shape(dummy_input)[:1]
        full_shape = K.constant(shape, shape=(len(shape),), dtype=np.int32)
        full_shape = K.concatenate([batch_shape,full_shape])
        return K.random_normal(full_shape, 0, 1)

    return Lambda(std_noise)
