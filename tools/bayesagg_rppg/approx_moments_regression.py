# tools/bayesagg_rppg/approx_moments_regression.py
import torch
from .BayesAgg_MTL import ApproxMoments


class ApproxMomentsRegression(ApproxMoments):
    """
    Monte–Carlo moments for an arbitrary scalar regression loss in 48‑D.
    Now fully vectorised (single backward call) and keeps the bias term.
    """
    def __init__(self, *, sqrt_power=1.0, num_mc_samples=512, loss_fn):
        super().__init__(sqrt_power, num_mc_samples)
        self.loss_fn = loss_fn          # callable(preds, gts) -> (S,N) losses

    # ------------------------------------------------------------------
    def dL_dh(self,
              features: torch.Tensor,          # (N,48)
              labels:   torch.Tensor,          # (N,)
              p_t:      torch.distributions.MultivariateNormal):

        S = self.num_mc_samples
        N, D = features.shape                     # D must be 48

        # --- sample (w,b) -------------------------------------------------
        wb = p_t.rsample((S,)).squeeze(1)         # (S, D+1)
        W, b = wb[:, :-1], wb[:, -1]              # (S,D) , (S,)

        # --- replicate features so they require‑grad ---------------------
        feats = features.detach().clone()
        feats = feats.unsqueeze(0).repeat(S, 1, 1).requires_grad_(True)  # (S,N,D)

        # --- forward & loss ----------------------------------------------
        preds  = (feats * W.unsqueeze(1)).sum(-1) + b.unsqueeze(1)        # (S,N)
        losses = self.loss_fn(preds, labels.unsqueeze(0).expand_as(preds))

        # --- single backward pass ----------------------------------------
        grads  = torch.autograd.grad(losses.sum(), feats,
                                     retain_graph=False)[0]              # (S,N,D)

        return grads, losses

    # def compute_moments(self,
    #                     features: torch.Tensor,
    #                     labels:   torch.Tensor,
    #                     p_t      : torch.distributions):

    #     dL_dh, sample_losses = self.dL_dh(features, labels, p_t)   # (S,N,D)

    #     # ---------- 1st and 2nd moments ----------------------------
    #     E_g   = self.first_moment(dL_dh)            # (N,D)
    #     E_gg  = self.second_moment(dL_dh)           # (N,D)

    #     Σ_g   = torch.clamp(E_gg - E_g**2, min=1e-8)
    #     Σ_g   = Σ_g ** self.sqrt_power              # (N,D)

    #     # ---------- add singleton “output” axis --------------------
    #     E_g = E_g.unsqueeze(1)          # (N,1,D)  ← match ExactMoments
    #     Σ_g = Σ_g.unsqueeze(1)          # (N,1,D)

    #     return E_g, Σ_g, sample_losses.mean()       # keep API identical