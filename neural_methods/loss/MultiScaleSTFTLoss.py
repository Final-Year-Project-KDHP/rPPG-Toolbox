import math, functools, typing
from collections import namedtuple
import torch, torch.nn as nn, torch.nn.functional as F
import scipy.signal as signal
from librosa.filters import mel as librosa_mel_fn   # install librosa

class MultiScaleSTFTLoss(nn.Module):
    """
    Multi‑scale mel / linear STFT loss for low‑frequency rPPG.
    """

    STFTParams = namedtuple("STFTParams",
                            ["win_length", "hop_length", "window_type", "match_stride"])

    def __init__(
        self,
        sampling_rate: int,
        win_lengths     = (128, 512, 1024),
        n_mels          = (16, 32, 64),
        loss_fn         = nn.L1Loss(),
        log_weight      = 1.0,
        mag_weight      = 0.1,
        clamp_eps       = 1e-5,
        pow             = 1.0,
        mel_fmin        = (0.0, 0.0, 0.0),
        mel_fmax        = (3.0, 3.0, 3.0),
        window_type     = "hann",
        match_stride    = False,
    ):
        super().__init__()
        assert len(win_lengths) == len(n_mels)
        self.sr  = sampling_rate
        self.cfg = [self.STFTParams(w, w//4, window_type, match_stride)
                    for w in win_lengths]
        self.n_mels       = n_mels
        self.loss_fn      = loss_fn
        self.log_weight   = log_weight
        self.mag_weight   = mag_weight
        self.clamp_eps    = clamp_eps
        self.pow          = pow
        self.mel_fmin     = mel_fmin
        self.mel_fmax     = mel_fmax

    # ------------------------------------------------------------------
    @staticmethod
    @functools.lru_cache(None)
    def _window(window_type: str, win_length: int):
        return signal.get_window(window_type, win_length)

    @staticmethod
    @functools.lru_cache(None)
    def _mel(sr, n_fft, n_mels, fmin, fmax):
        return librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels,
                              fmin=fmin, fmax=fmax if fmax else sr/2)

    # ------------------------------------------------------------------
    def _mel_spec(self, wav: torch.Tensor, cfg, n_mels, fmin, fmax):
        B, T = wav.shape
        hop = cfg.hop_length

        # reflect‑pad so #frames*hop == T
        right  = math.ceil(T / hop) * hop - T
        pad    = (cfg.win_length - hop)//2
        wav    = F.pad(wav, (pad, pad+right), mode="reflect")

        window = torch.tensor(self._window(cfg.window_type, cfg.win_length),
                              device=wav.device).float()
        stft = torch.stft(wav, n_fft=cfg.win_length, hop_length=hop,
                          window=window, center=True,
                          return_complex=True)
        mag  = stft.abs()            # [B, Nf, Nt]

        mel_filter = torch.tensor(
            self._mel(self.sr, cfg.win_length, n_mels, fmin, fmax),
            device=wav.device, dtype=mag.dtype)         # [n_mels, Nf]
        mel = torch.matmul(mel_filter, mag)             # [B, n_mels, Nt]
        return mel

    # ------------------------------------------------------------------
    def forward(self, est: torch.Tensor, ref: torch.Tensor):
        """
        est / ref: shape [B, T]  (already zero‑meaned in trainer)
        """
        loss = est.new_tensor(0.)
        for (cfg, n_m, fmin, fmax) in zip(self.cfg, self.n_mels,
                                          self.mel_fmin, self.mel_fmax):
            mel_est = self._mel_spec(est, cfg, n_m, fmin, fmax)
            mel_ref = self._mel_spec(ref, cfg, n_m, fmin, fmax)

            mel_est_log = torch.log( mel_est.clamp(min=self.clamp_eps).pow(self.pow) )
            mel_ref_log = torch.log( mel_ref.clamp(min=self.clamp_eps).pow(self.pow) )

            loss += self.log_weight * self.loss_fn(mel_est_log, mel_ref_log)
            loss += self.mag_weight * self.loss_fn(mel_est, mel_ref)

        return loss / len(self.cfg)
