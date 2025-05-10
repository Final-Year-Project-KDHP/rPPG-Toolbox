# bayesagg_rppg/approx_moments_regression.py
import torch
from .BayesAgg_MTL import ApproxMoments            # original base class

class ApproxMomentsRegression(ApproxMoments):
    """
    Monte-Carlo estimate of E[dL/dh] and Var[dL/dh] for **any** scalar
    regression loss (Neg-Pearson, frequency hybrid, …).
    """
    def __init__(self, *, sqrt_power=1.0, num_mc_samples=512, loss_fn):
        super().__init__(sqrt_power=sqrt_power, num_mc_samples=num_mc_samples)
        self.loss_fn = loss_fn  # callable: (preds: Tensor[S,N], labels: Tensor[N]) -> Tensor[S,N]

    def dL_dh(self, features: torch.Tensor, labels: torch.Tensor, p_t: torch.distributions.Distribution):
        """
        features: (N, D_model)
        labels:   (N,)
        p_t:      posterior over last‐layer params, loc has shape (1, D_w+1)
        Returns:
          grads:  (S, N, D_w)   where D_w = p_t.mean.size(-1)-1
          loss:   (S, N)
        """
        S = self.num_mc_samples
        N, D_model = features.shape
        D_w = p_t.mean.size(-1)   # number of hidden dims the head expects

        # 1) Crop any extra dims off the representation
        if D_model > D_w:
            features = features[:, :D_w].contiguous()
        elif D_model < D_w:
            raise RuntimeError(f"Representation dim {D_model} < head dim {D_w}")

        # 2) Sample S draws of (w,b) from the posterior: shape (S, D_w+1)
        # wb = p_t.rsample((S,)).squeeze(1)

        # W = wb[:, :-1]   # (S, D_w)
        # b = wb[:, -1]    # (S,)

        W  = p_t.rsample((S,)).squeeze(1)          # (S, D_w)
        # optional bias term (zero) just to keep code below unchanged
        b  = torch.zeros(S, device=features.device)

        # 3) Forward: preds[s,n] = W[s] · features[n] + b[s]
        #    → (S, N) = (S, D_w) @ (D_w, N)
        preds = W.matmul(features.t()) + b.unsqueeze(1)

        # 4) Compute per‐sample loss
        loss = self.loss_fn(preds, labels.unsqueeze(0).expand_as(preds))  # (S, N)

        # 5) Now get dℓ/dh via autograd for each sample
        feats_req = features.detach().clone().requires_grad_(True)
        grads = []
        for s in range(S):
            # ŷ = features · W[s]^T + b[s]
            y_hat = feats_req.matmul(W[s]) + b[s]          # (N,)
            # l_s   = self.loss_fn(y_hat, labels)            # scalar

            # if we're passed a 1D waveform, add a dummy batch dim:
            if y_hat.dim() == 1:
                y_in = y_hat.unsqueeze(0)           # (1, N)
                g_in = labels.unsqueeze(0)          # (1, N)
            else:
                y_in, g_in = y_hat, labels
            l_s   = self.loss_fn(y_in, g_in)        # now (1,) or scalar

            g_s   = torch.autograd.grad(l_s, feats_req, retain_graph=True)[0]  # (N, D_w)
            grads.append(g_s)
        grads = torch.stack(grads, 0)  # (S, N, D_w)

        return grads, loss
        