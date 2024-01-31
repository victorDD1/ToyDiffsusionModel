from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from scheduler import DDPM
from model import ConditionalMLP
from data import CircleDataset, SphereDataset

if __name__ == "__main__":
    N = 4096
    L = 24

    EPOCHS = 2000
    BATCH_SIZE = 128
    LR = 0.0005

    N_H = 2
    H = 128

    T = 100

    dataset = CircleDataset(N, L)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    batch = next(iter(dataloader))
    print("batch shape", batch.shape)
    fig, ax = plt.subplots(3,3)
    for i in range(9):
        j = i%3
        ax[i//3, j].scatter(batch[i, :, 0], batch[i, :, 1])
    plt.savefig("./results/data.png")
    plt.show()

    model = ConditionalMLP(2, 2, T, N_H, H, torch.nn.Softplus())
    ddpm = DDPM(model, betas=(1e-4, 0.02), n_T=T)
    optim = Adam(ddpm.parameters(), LR)
    criterion = torch.nn.MSELoss()

    pbar = tqdm(range(EPOCHS), leave=True)
    for i in pbar:
        ddpm.train()

        loss_ema = None
        for x in dataloader:
            optim.zero_grad()
            loss = criterion(*ddpm(x))
            loss.backward()
            if loss_ema is None:
                loss_ema = loss.item()
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss.item()
            pbar.set_description(f"loss: {loss_ema:.4f}")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optim.step()

    with torch.no_grad():

        xh = ddpm.sample(9, (1, 3*L, 2), "cpu").squeeze()

        fig, ax = plt.subplots(3,3)
        for i in range(9):
            j = i%3
            ax[i//3, j].scatter(xh[i, :, 0], xh[i, :, 1])
        plt.savefig("./results/generated.png")
        plt.show()
        # save model
        torch.save(ddpm.state_dict(), f"./results/ddpm_circle.pth")