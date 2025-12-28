import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

from ddpm.trainer import DDPMTrainer
from ddpm.model import DiffusionUNet


"""
Training diffusion to generate MNIST images.
Reference: https://arxiv.org/pdf/2006.11239.
"""

# utilities
def save_checkpoint(
    model: DiffusionUNet, ema_model: DiffusionUNet,
    optim: torch.optim.Optimizer, epoch: int, global_step: int, path: str
) -> None:
    torch.save({
        "model": model.state_dict(),
        "model_class": model.__class__.__name__,
        "model_config": model.config,
        "ema_model": ema_model.state_dict(),
        "optimizer": optim.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
    }, path)

def cosine_beta_schedule(T, s=0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1)
    alpha_bar = torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    return betas.clamp(1e-4, 0.999)

# load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
batch_size = 32
trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
validset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)

# create data loaders
class DiffusionMNIST(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx) -> torch.Tensor:
        return self.data[idx][0]

    def __len__(self) -> int:
        return len(self.data)

trainset = DiffusionMNIST(trainset)
validset = DiffusionMNIST(validset)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
validloader = DataLoader(validset, batch_size=32)

model = DiffusionUNet(in_channels=1, base_channels=32, time_dim=128, n_groups=16)
betas = cosine_beta_schedule(100)
device = 'mps'
trainer = DDPMTrainer(model, betas, device)

n_epochs = 1

trainer.train(trainloader, validloader, n_epochs)
save_checkpoint(model, trainer.ema_model, trainer.optim, n_epochs, trainer.global_step, 'model_checkpoint.pt')

# after training
sample = trainer.sample(batch_size=4)
