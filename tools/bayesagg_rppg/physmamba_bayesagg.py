# --------------  physmamba_bayesagg.py  (UPDATED)  --------------
import torch
from .BayesAgg_MTL import BayesAggMTL, LastLayerPosteriorRegression, ExactMoments
from .approx_moments_regression import ApproxMomentsRegression

# helper ----------------------------------------------------------
def _flatten_conv1x1(weight: torch.Tensor) -> torch.Tensor:
    """
    Conv3d 1×1×1 kernel  (1,C,1,1,1)  →  (1,C)  view that shares storage.
    Works for *any* out‑channels.
    """
    return weight.view(weight.size(0), -1)


class PhysMambaBayesAgg:
    """
    BayesAgg wrapper for (rPPG, SpO₂) with projection instead of hard crop.
    """
    def __init__(self, gamma=1e-3, sqrt_power=.5,
                 num_mc_samples=128, hr_loss_fn=None, spo2_loss_fn=None):

        # 1️⃣ Posterior / moments per task -------------------------
        self.post_r = LastLayerPosteriorRegression(num_outputs=1, gamma=gamma)
        self.post_s = LastLayerPosteriorRegression(num_outputs=1, gamma=gamma)

        self.mom_r  = ApproxMomentsRegression(
            sqrt_power=sqrt_power,
            num_mc_samples=num_mc_samples,
            loss_fn=hr_loss_fn)
        self.mom_s  = ExactMoments(sqrt_power=sqrt_power)

        self.agg = BayesAggMTL(
            num_tasks=2,
            n_outputs_per_task_group=[1, 1],
            task_types=['regression', 'regression'],
            reg_hps={'gamma': gamma, 'sqrt_power': sqrt_power}
        )

    # -------------------------------------------------------------
    def backward(self, *, losses, last_layer_params,
                       representation, labels_r, labels_s):
        """
        losses  : tensor([ℓ_hr, ℓ_spo2])
        last_layer_params = [w_r, b_r, w_s, b_s]  (Conv kernels!)
        representation    : (B,T,D_full)  *or*  (N,D_full)
        """
        # # ----------  flatten hidden tank  ------------------------
        # if representation.dim() == 3:
        #     B, T, D_full = representation.shape
        #     rep_flat     = representation.reshape(B*T, D_full)
        # else:
        #     rep_flat     = representation                                # (N,D_full)

        # # ----------  pick weight dims ----------------------------
        # w_r, b_r, w_s, b_s = last_layer_params
        # # w_r_vec = _flatten_conv1x1(w_r)
        # # w_s_vec = _flatten_conv1x1(w_s)
        # # in_r, in_s = w_r_vec.size(1), w_s_vec.size(1)
        # # D_common   = min(in_r, in_s)

        # # ----------  PROJECTION instead of crop ------------------
        # # identity‑like projector of size (max(in_r,in_s), D_common)
        # P = torch.eye(max(in_r, in_s), D_common,
        #               device=rep_flat.device, dtype=rep_flat.dtype)
        # rep_common = rep_flat[:, :P.size(0)].matmul(P)                   # (N, D_common)

        # ---- 1) flatten (B,T,48) → (N,48) -------------------------------
        rep_flat = (representation.reshape(-1, representation.size(-1))
                    if representation.dim() == 3 else representation)

        # ---- 2) flatten conv kernels ------------------------------------
        w_r, b_r, w_s, b_s = last_layer_params
        w_r_vec = _flatten_conv1x1(w_r)          # (1,48)
        w_s_vec = _flatten_conv1x1(w_s)          # (1,48)
        # ----- sanity check ----------------------------------------------
        D_feat = rep_flat.size(1)
        assert w_r_vec.size(1) == w_s_vec.size(1) == D_feat == 48, \
            f"Dim mismatch: features {D_feat}  w_r {w_r_vec.size(1)}  w_s {w_s_vec.size(1)}"

        rep_common = rep_flat                    # keep full 48‑D space      

        # ----------  back‑prop through last‑layer weights --------
        self.agg.backward_last_layer(losses, last_layer_params)

        # ----------  per‑task posteriors -------------------------
        p_r = self.post_r.compute_posterior(
            last_layer_params=[w_r_vec, b_r],
            features=rep_common, labels=labels_r)

        p_s = self.post_s.compute_posterior(
            last_layer_params=[w_s_vec, b_s],
            features=rep_common, labels=labels_s.unsqueeze(-1))

        # # ----------  moments -------------------------------------
        # Eg_r, Sg_r, _ = self.mom_r.compute_moments(rep_common, labels_r, p_r)
        # Eg_s, Sg_s    = self.mom_s.compute_moments(rep_common,
        #                                            labels_s.unsqueeze(-1), p_s)

        # # match dims (fix #5)
        # Eg = torch.cat([Eg_r.unsqueeze(1), Eg_s.unsqueeze(1)], dim=1)    # (N,2,Dc)
        # Sg = torch.cat([Sg_r.unsqueeze(1), Sg_s.unsqueeze(1)], dim=1)    # (N,2,Dc)

        Eg_r, Sg_r, _ = self.mom_r.compute_moments(rep_common, labels_r, p_r)
        Eg_r = Eg_r.unsqueeze(1)     # (N,1,48)       ← add output axis
        Eg_s, Sg_s    = self.mom_s.compute_moments(rep_common,
                                                   labels_s.unsqueeze(-1), p_s)

        # Eg = torch.stack([Eg_r, Eg_s], dim=1)       # (N,2,48)
        # Sg = torch.stack([Sg_r, Sg_s], dim=1)       # (N,2,48)

        Eg = torch.cat([Eg_r, Eg_s], dim=1)           # (N,2,48)
        Sg = torch.cat([Sg_r.unsqueeze(1), Sg_s], dim=1)        

        # ---------- aggregate & inject ---------------------------
        dLdh = self.agg.agg_scheme.aggregate(Eg, Sg)                     # (N,Dc)
        rep_common.backward(gradient=dLdh.to(rep_common.dtype))
