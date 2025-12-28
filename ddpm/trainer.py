from copy import deepcopy

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F


class DDPMTrainer:
    def __init__(self, model: nn.Module, betas: torch.Tensor, device: torch.device, ema_warmup: int = 500):
        self.model = model.to(device)
        self.optim = optim.AdamW(self.model.parameters(), lr=1e-4)

        # model used to track exponential-moving-averages of parameters of self.model
        self.ema_model = deepcopy(self.model)
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
        self.ema_model.to(device)

        self.ema_warmup = ema_warmup
        self.global_step = 0

        self.betas = betas.to(device)

        self.device = device

        self.sample_size = None

    def noise_prediction_loss(self, preds, targets, snr_t, snr_cap = 5.0):
        weight = torch.clamp(snr_t, max=snr_cap)
        return ( weight * (preds - targets) ** 2 ).mean()
    
    def mse_loss(self, preds, targets):
        return F.mse_loss(preds, targets)

    def train(self, train_loader, valid_loader, n_epochs, snr_cap = 5.0):
        self.model.train()
        self.ema_model.eval()

        T = len(self.betas)
        alphas = 1.0 - self.betas
        alpha_bars = torch.cumprod(alphas, dim=0)     
        snr = alpha_bars / (1 - alpha_bars) # signal-to-noise-ratio

        print('Beginning training...')

        for epoch in range(n_epochs):
            for X_batch in train_loader:
                X_batch = X_batch.to(self.device)
                t = torch.randint(0, T, size=(X_batch.shape[0],), device=self.device)
                alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)

                eps = torch.randn_like(X_batch)
                eps_pred = self.model(torch.sqrt(alpha_bar_t) * X_batch + torch.sqrt(1 - alpha_bar_t) * eps, t)

                loss = self.noise_prediction_loss(eps_pred, eps, snr[t].view(-1, 1, 1, 1), snr_cap)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                self.global_step += 1
                self.update_ema()
            
            valid_loss = self.eval_model(valid_loader, alpha_bars)
            print(f'Epoch {epoch+1}/{n_epochs} - Train loss: {loss.item()}, Validation loss: {valid_loss}')

        self.sample_size = X_batch.shape[1:] # save for sampling

    @torch.no_grad()
    def eval_model(self, valid_loader, alpha_bars):
        T = len(alpha_bars)
        total_loss = 0.0
        n = 0
        self.ema_model.eval()
        for X_batch in valid_loader:
            X_batch = X_batch.to(self.device)
            t = torch.randint(0, T, size=(X_batch.shape[0],), device=self.device)
            alpha_bar_t = alpha_bars[t].view(-1, 1, 1, 1)
            eps = torch.randn_like(X_batch)
            eps_pred = self.ema_model(torch.sqrt(alpha_bar_t) * X_batch + torch.sqrt(1 - alpha_bar_t) * eps, t)
            loss = self.mse_loss(eps_pred, eps) # vanilla mse for validation
            total_loss += loss.item() * X_batch.size(0)
            n += X_batch.size(0)
        return total_loss / n

    @torch.no_grad()
    def update_ema(self, decay=0.995):
        if self.global_step < self.ema_warmup:
            self.ema_model.load_state_dict(self.model.state_dict())
            return

        for param, ema_param in zip(self.model.parameters(), self.ema_model.parameters()):
            ema_param.mul_(decay).add_(param, alpha = 1 - decay)

    @torch.no_grad()
    def sample(self, batch_size: int):
        self.ema_model.eval()

        size = (batch_size, *self.sample_size)

        alphas = 1.0 - self.betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        alpha_bars_prev = torch.cat(
            [torch.ones(1, device=self.device), alpha_bars[:-1]]
        )

        coef_eps = self.betas / torch.sqrt(1 - alpha_bars)

        sigmas = torch.sqrt(
            self.betas * (1 - alpha_bars_prev) / (1 - alpha_bars)
        )

        x = torch.randn(size, device=self.device)
        T = len(self.betas)

        for t in reversed(range(T)):
            t_batch = torch.full((size[0],), t, device=self.device, dtype=torch.long)

            eps_pred = self.ema_model(x, t_batch)

            x0_pred = (
                x - torch.sqrt(1 - alpha_bars[t]) * eps_pred
            ) / torch.sqrt(alpha_bars[t])

            x0_pred = x0_pred.clamp(-1, 1)

            eps_pred = (
                x - torch.sqrt(alpha_bars[t]) * x0_pred
            ) / torch.sqrt(1 - alpha_bars[t])

            z = torch.randn_like(x) if t > 0 else 0
            x = (
                (x - coef_eps[t] * eps_pred) / torch.sqrt(alphas[t])
                + sigmas[t] * z
            )

        return x
