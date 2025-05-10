# bayesagg_rppg/physmamba_bayesagg.py
import torch
from .BayesAgg_MTL import BayesAggMTL, LastLayerPosteriorRegression, ExactMoments
from .approx_moments_regression import ApproxMomentsRegression

class PhysMambaBayesAgg:
    """
    Holds BayesAgg object + per‑task helper modules for PhysMamba (rPPG, SpO2).
    """

    def __init__(self, gamma=1e-3, sqrt_power=0.5,
                 num_mc_samples=128, hr_loss_fn=None, spo2_loss_fn=None):
        # ------------------------------------------------------------------
        # 1.  Posterior & moment modules per task
        # ------------------------------------------------------------------
        self.post_r = LastLayerPosteriorRegression(num_outputs=1, gamma=gamma)
        self.post_s = LastLayerPosteriorRegression(num_outputs=1, gamma=gamma)

        self.mom_r  = ApproxMomentsRegression(
                          sqrt_power=sqrt_power,
                          num_mc_samples=num_mc_samples,
                          loss_fn=hr_loss_fn)
        self.mom_s  = ExactMoments( sqrt_power=sqrt_power)

        self.hr_loss_fn   = hr_loss_fn   # injected from trainer (frequency hybrid)
        self.spo2_loss_fn = spo2_loss_fn # simple RMSE

        # ------------------------------------------------------------------
        # 2.  BayesAgg core
        # ------------------------------------------------------------------
        self.agg = BayesAggMTL(
            num_tasks=2,
            n_outputs_per_task_group=[1, 1],
            task_types=['regression', 'regression'],
            reg_hps={'gamma': gamma, 'sqrt_power': sqrt_power}
        )

    # ----------------------------------------------------------------------
    # main entry – called instead of .backward()
    # ----------------------------------------------------------------------
    # def backward(self, *,   # keyword‑only for clarity
    #              losses,                # (2,) tensor [loss_r, loss_s]
    #              last_layer_params,     # [wr, br, ws, bs]
    #              representation,        # (B*T, D) tensor
    #              labels_r, labels_s):   # (B*T,) each

    #     # ---------- NEW: ensure 2‑D ---------------------------------
    #     if representation.dim() == 3:          # (B,T,D) → (B*T,D)
    #         B, T, D = representation.shape
    #         representation = representation.view(B*T, D)
    #         labels_r = labels_r.view(B*T)
    #         labels_s = labels_s.view(B*T)
    #     # ------------------------------------------------------------

    #     # 1. Back‑prop through last‑layer weights to get ∂L/∂w
    #     self.agg.backward_last_layer(losses, last_layer_params)

    #     # 2. Posterior per task
    #     p_r = self.post_r.compute_posterior(
    #                 last_layer_params=last_layer_params[:2],
    #                 features=representation,
    #                 labels=labels_r)
    #     p_s = self.post_s.compute_posterior(
    #                 last_layer_params=last_layer_params[2:],
    #                 features=representation,
    #                 labels=labels_s)

    #     # 3. Moments per task
    #     # Eg_r, Sg_r, _ = self.mom_r.compute_moments(
    #     #                     features=representation,
    #     #                     labels=labels_r,
    #     #                     p_t=p_r,
    #     #                     loss_fn=self.hr_loss_fn)

    #     print("repr", representation.shape,
    #         "labels_r", labels_r.shape)


    #     Eg_r, Sg_r, _ = self.mom_r.compute_moments(
    #                  features=representation,
    #                  labels=labels_r,
    #                  p_t=p_r)
    #     Eg_s, Sg_s    = self.mom_s.compute_moments(
    #                         features=representation,
    #                         labels=labels_s,
    #                         p_t=p_s)

    #     # 4. Stack → BayesAgg aggregate
    #     Eg   = torch.stack([Eg_r, Eg_s], dim=1)        # (B*T, 2, D)
    #     Sg   = torch.stack([Sg_r, Sg_s], dim=1)        # (B*T, 2, D)
    #     dLdh = self.agg.agg_scheme.aggregate(Eg, Sg)   # (B*T, D)

    #     # 5. Inject gradient into graph
    #     representation.backward(gradient=dLdh.to(representation.dtype))


    def backward(self, *, losses, last_layer_params,
                        representation, labels_r, labels_s):
        """
        losses            : tensor of shape [2] containing [hr_loss, spo2_loss]
        last_layer_params : [w_r, b_r, w_s, b_s]
        representation    : either [B, T, D_full] or already flattened [N, D_full]
        labels_r          : tensor [N]  (HR targets)
        labels_s          : tensor [N]  (SpO₂ targets)
        """

        # 1) Flatten the representation to [N, D_full]
        if representation.dim() == 3:              # (B, T, D_full)
                B, T, D_full = representation.shape
                rep_flat     = representation.view(B * T, D_full)
        else:
                rep_flat     = representation

        # 2) Pull out each head's weight & bias
        w_r, b_r, w_s, b_s = last_layer_params

        # 3) Compute the size of the truly shared feature subspace
        in_ch_r = w_r.shape[1]  # number of channels HR head actually uses
        in_ch_s = w_s.shape[1]  # number of channels SpO₂ head uses
        D_common = min(in_ch_r, in_ch_s)

        # 4) Slice to that common subspace → [N, D_common]
        rep_common = rep_flat[:, :D_common]

        # 5) Back-prop into the last-layer params (so w_r, b_r, w_s, b_s get their gradients)
        self.agg.backward_last_layer(losses, last_layer_params)

        # 6) Compute the posterior over last-layer params for each task on rep_common
        p_r = self.post_r.compute_posterior(
                last_layer_params=[w_r, b_r],
                features=        rep_common,
                labels=          labels_r
        )
        p_s = self.post_s.compute_posterior(
                last_layer_params=[w_s, b_s],
                features=        rep_common,
                labels=          labels_s.unsqueeze(-1)
        )

        # 7) Compute first & second moments of ∂L/∂h for each task (on the same D_common)
        Eg_r, Sg_r, _ = self.mom_r.compute_moments(
                features=rep_common,
                labels=  labels_r,
                p_t=     p_r
        )
        Eg_s, Sg_s    = self.mom_s.compute_moments(
                features=rep_common,
                labels=  labels_s.unsqueeze(-1),
                p_t=     p_s
        )

        # 8) Stack them into shape [N, 2, D_common]
        Eg = torch.cat([Eg_r.unsqueeze(1),    # → [N,1,D_common]
                        Eg_s],                 # → [N,1,D_common]
                        dim=1)                  # → [N,2,D_common]
        Sg = torch.cat([Sg_r.unsqueeze(1),
                        Sg_s],
                        dim=1)

        # 9) Fuse via Gaussian‐Agg → produces [N, D_common]
        dLdh = self.agg.agg_scheme.aggregate(Eg, Sg)

        # 10) Inject that fused gradient back into the shared trunk
        rep_common.backward(gradient=dLdh.to(rep_common.dtype))
