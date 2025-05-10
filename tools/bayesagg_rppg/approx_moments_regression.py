# --------------  approx_moments_regression.py  (UPDATED)  -------
import torch
from .BayesAgg_MTL import ApproxMoments             # base class

class ApproxMomentsRegression(ApproxMoments):
    """
    Monte‑Carlo moments for *any* scalar regression loss.
    Keeps the sampled bias **and** is fully vectorised.
    """
    def __init__(self, *, sqrt_power=1.0,
                 num_mc_samples=512, loss_fn):
        super().__init__(sqrt_power, num_mc_samples)
        self.loss_fn = loss_fn      # callable (pred, gt) -> loss

    # -------------------------------------------------------------
    def dL_dh(self,
              features: torch.Tensor,   # [N,D]
              labels:   torch.Tensor,   # [N]
              p_t:      torch.distributions.MultivariateNormal):
        S          = self.num_mc_samples
        N, D_model = features.shape

        # -------- sample last‑layer params (w,b) ------------------
        wb = p_t.rsample((S,)).squeeze(1)          # [S, D+1]
        W, b = wb[:, :-1], wb[:, -1]               # keep bias!

        # -------- forward & loss ---------------------------------
        # feats_req will hold gradients
        feats_req = features.detach().clone().requires_grad_(True)   # [N,D]
        feats_exp = feats_req.unsqueeze(0).expand(S, N, D_model)     # [S,N,D]
        preds     = torch.einsum('snd,sd->sn', feats_exp, W) + b[:, None]  # [S,N]

        lbl_exp   = labels.unsqueeze(0).expand_as(preds)             # [S,N]
        losses    = self.loss_fn(preds, lbl_exp)                     # [S,N]

        # -------- ∂L/∂h  (single backward pass) -------------------
        grads = torch.autograd.grad(
            losses.sum(),         # scalar
            feats_req,
            retain_graph=False,
            allow_unused=False
        )[0]                               # [N,D]
        grads = grads.unsqueeze(0).expand(S, N, D_model).clone()     # [S,N,D]

        return grads, losses
    # -------------------------------------------------------------
