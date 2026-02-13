import math
import logging

import torch
import torch.nn as nn


class SOXLoss(nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma: float,
                 is_sox: bool = True,
                 is_scent: bool = False,
                 ) -> None:
        super().__init__()
        self.data_size = data_size
        self.gamma_orig = gamma
        self.gamma = gamma
        self.is_sox = is_sox
        self.is_scent = is_scent
        if not self.is_sox and self.is_scent:
            raise ValueError("Cannot use SCENT when SOX is disabled.")
        self.nu = torch.zeros(data_size, device="cpu").reshape(-1, 1)

    def adjust_gamma(self, epoch: int, max_epoch: int) -> None:
        if not self.is_sox:
            self.gamma = 0.5 * self.gamma_orig * (1 + torch.cos(torch.tensor(epoch / max_epoch * math.pi)))
            logging.info(f"Adjusted gamma to {self.gamma:.6f} at epoch {epoch}")
        elif not self.is_scent:
            self.gamma = 0.5 * (1.0 - self.gamma_orig) * (1 + torch.cos(torch.tensor(epoch / max_epoch * math.pi))) + self.gamma_orig
            logging.info(f"Adjusted gamma to {self.gamma:.6f} at epoch {epoch}")

    def forward(self,
                logits: torch.Tensor,
                indices: torch.Tensor,
                ) -> dict:
        nu = self.nu[indices].to(logits.device)

        # update nu
        stats_dict = {}
        bad_idx = torch.nonzero(nu == 0.0, as_tuple=True)[0]
        if self.is_scent:
            gamma = 1.0 - 1.0 / (1.0 + torch.exp(self.gamma + nu))
            stats_dict["gamma/mean"] = gamma.mean()
            stats_dict["gamma/median"] = gamma.median()
            stats_dict["gamma/max"] = gamma.max()
            stats_dict["gamma/min"] = gamma.min()
        else:
            gamma = self.gamma
        exp_logits_mean = torch.sum(torch.exp(logits), dim=-1, keepdim=True).detach() / (logits.shape[1] - 1)
        if self.is_sox:
            nu = torch.log((1- gamma) * torch.exp(nu) + gamma * exp_logits_mean)
            nu_for_grad = nu
        else:
            nu_for_grad = nu
            grad_nu = 1.0 - exp_logits_mean / torch.exp(nu)
            nu = nu - gamma * grad_nu
        if bad_idx.shape[0] > 0:
            nu[bad_idx] = torch.log(exp_logits_mean[bad_idx])
            nu_for_grad[bad_idx] = nu[bad_idx]
        self.nu[indices] = nu.cpu()

        # compute loss
        loss = torch.mean(torch.sum(torch.exp(logits - nu_for_grad), dim=-1, keepdim=True) / (logits.shape[1] - 1))
        loss_dict = {"loss": loss} | stats_dict
        return loss_dict


class SoftPlusLoss(nn.Module):
    def __init__(self,
                 data_size: int,
                 gamma: float,
                 rho: float,
                 ) -> None:
        super().__init__()
        self.data_size = data_size
        self.gamma_orig = gamma
        self.gamma = gamma
        self.rho = rho
        self.nu = torch.zeros(data_size, device="cpu").reshape(-1, 1)

    def adjust_gamma(self, epoch: int, max_epoch: int) -> None:
        self.gamma = 0.5 * self.gamma_orig * (1 + torch.cos(torch.tensor(epoch / max_epoch * math.pi)))
        logging.info(f"Adjusted gamma to {self.gamma:.6f} at epoch {epoch}")

    def forward(self,
                logits: torch.Tensor,
                indices: torch.Tensor,
                ) -> dict:
        nu = self.nu[indices].to(logits.device)

        # update nu
        bad_idx = torch.nonzero(nu == 0.0, as_tuple=True)[0]
        gamma = self.gamma
        gamma_mean = torch.tensor(self.gamma)
        gamma_median = torch.tensor(self.gamma)
        exp_logits_nu = torch.exp(logits - nu).detach()
        exp_logits_mean = torch.mean((exp_logits_nu / (1 + self.rho * exp_logits_nu)), dim=-1, keepdim=True)
        nu_for_grad = nu
        grad_nu = 1.0 - exp_logits_mean
        nu = nu - gamma * grad_nu
        if bad_idx.shape[0] > 0:
            nu[bad_idx] = torch.log(exp_logits_mean[bad_idx])
            nu_for_grad[bad_idx] = nu[bad_idx]
        self.nu[indices] = nu.cpu()

        # compute loss
        loss = torch.mean(torch.log(1 + self.rho * torch.exp(logits - nu_for_grad))) / self.rho
        loss_dict = {
            "loss": loss,
            "gamma/mean": gamma_mean,
            "gamma/median": gamma_median,
        }
        return loss_dict


class PrimalLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                logits: torch.Tensor,
                indices: torch.Tensor,
                ) -> dict:
        loss = torch.logsumexp(logits, dim=-1).mean()
        loss_dict = {
            "loss": loss,
        }
        return loss_dict
