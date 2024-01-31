from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import sys

from scheduler import DDPM
from data import CircleDataset, SphereDataset
from UNet import ConditionalUnet1D

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

if __name__ == "__main__":
    N = 10000
    L = 24

    EPOCHS = 50
    B = 128
    LR = 0.0005

    N_H = 2
    H = 128

    T = 100

    dataset = CircleDataset(N, L)
    dataloader = DataLoader(dataset, B, shuffle=True)
    batch = next(iter(dataloader))
    print("batch shape", batch[0].shape, batch[1].shape)
    fig, ax = plt.subplots(1)
    color = lambda r: "red" if r < 1 else "blue"
    c = [color(r) for r in batch[1].reshape(-1)]
    ax.scatter(batch[0][:, :, 0].reshape(-1, 2), batch[0][:, :, 1].reshape(-1, 2), color=c)
    plt.savefig("./results/data.png")
    plt.show()

    device = "cpu"

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim = 2,
        global_cond_dim = L * 1,
        diffusion_step_embed_dim=128,
        down_dims=[64, 128, 256],
        kernel_size=3,
        n_groups=8
    )  

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=T,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(
        parameters=noise_pred_net.parameters(),
        power=0.75)

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=noise_pred_net.parameters(),
        lr=LR, weight_decay=1e-6)

    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * EPOCHS
    )
    
    if len(sys.argv) > 1 and sys.argv[1] == "-t":
        with tqdm(range(EPOCHS), desc='Epoch') as tglobal:
            # epoch loop
            for epoch_idx in tglobal:
                epoch_loss = list()
                # batch loop
                with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                    for points, radius in tepoch:
                        B = points.shape[0]

                        # sample noise to add to actions
                        noise = torch.randn(points.shape, device=device)

                        # sample a diffusion iteration for each data point
                        timesteps = torch.randint(
                            0, noise_scheduler.config.num_train_timesteps,
                            (B,), device=device
                        ).long()

                        # add noise to the clean images according to the noise magnitude at each diffusion iteration
                        # (this is the forward diffusion process)
                        noisy_actions = noise_scheduler.add_noise(
                            points, noise, timesteps)

                        # predict the noise residual
                        noise_pred = noise_pred_net(
                            noisy_actions, timesteps, global_cond=radius)

                        # L2 loss
                        loss = nn.functional.mse_loss(noise_pred, noise)

                        # optimize
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()
                        # step lr scheduler every batch
                        # this is different from standard pytorch behavior
                        lr_scheduler.step()

                        # update Exponential Moving Average of the model weights
                        ema.step(noise_pred_net.parameters())

                        # logging
                        loss_cpu = loss.item()
                        epoch_loss.append(loss_cpu)
                        tepoch.set_postfix(loss=loss_cpu)
                tglobal.set_postfix(loss=np.mean(epoch_loss))

        torch.save(noise_pred_net.state_dict(), f"./results/ddpm_circle.pth")

    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(torch.load(f"./results/ddpm_circle.pth"))
    with torch.no_grad():
        # reshape observation to (B,obs_horizon*obs_dim)
        obs_cond = torch.randint(high=2, size=(B, L)) / 2. + 0.5 # Radius .5 or 1 (between 0 and 1)

        print(obs_cond.shape)
        # initialize action from Guassian noise
        noisy_action = torch.randn(
            (B, L, 2), device=device)
        naction = noisy_action

        # init scheduler
        noise_scheduler.set_timesteps(T)

        for k in noise_scheduler.timesteps:
            # predict noise
            noise_pred = ema_noise_pred_net(
                sample=naction,
                timestep=k,
                global_cond=obs_cond
            )

            # inverse diffusion step (remove noise)
            naction = noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=naction
            ).prev_sample

    print(naction.shape)
    
    fig, ax = plt.subplots(1)
    color = lambda r: "red" if r < 1 else "blue"
    c = [color(r) for r in obs_cond.reshape(-1)]
    ax.scatter(naction[:, :, 0].reshape(-1, 2), naction[:, :, 1].reshape(-1, 2), color=c)
    plt.savefig("./results/generated.png")
    plt.show()