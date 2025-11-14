from typing import Optional

import torch
from torch import nn

import random

from base.cnns import *
from base.norm import ChannelwiseLayerNorm

class SpeechEncoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        stride = kernel_sizes[0]/2
        paddings = [int((kernel_size - stride)/2) for kernel_size in kernel_sizes]
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            setattr(self, f'conv{i}', nn.Conv1d(in_channels = in_channels,
                                                out_channels = out_channels,
                                                kernel_size = kernel_size,
                                                padding = padding,
                                                stride = int(stride)))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, X):
        """
        X: (N, C, T0)
        """
        out = []
        for i in range(len(self.kernel_sizes)):
            out.append(getattr(self, f'conv{i}')(X))
        return self.relu(torch.concat(out, dim = 1))

class MSFE(nn.Module):
    """
    Multi-scale Feature Extraction Module
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 conv_channels: int,
                 num_dsp: int,
                 spk_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_channels = conv_channels
        self.spk_dim = spk_dim
        self.num_dsp = num_dsp
        self.in_proj = nn.Sequential(ChannelwiseLayerNorm(in_channels),
                                     nn.Conv1d(in_channels, out_channels, 1))
        for i in range(num_dsp):
            setattr(self, f'downsample{i}', DownSample(out_channels, out_channels, 
                                                       kernel_size = 3, stride = 2))
        self.out_proj = nn.Sequential(DownSample(out_channels, out_channels, 
                                                 kernel_size = 3, stride = 2),
                                      nn.Conv1d(out_channels, out_channels, kernel_size = 1))
        
    def forward(self, x: torch.Tensor, lens: Optional[torch.Tensor] = None):
        """
        x: (num_spks, )
        """
        local_embs = []
        x = self.in_proj(x)
        for i in range(self.num_dsp):
            x = getattr(self, f'downsample{i}')(x)
            if lens is not None:
                lens = lens//2
                local_embs.append(x.sum(dim = -1)/lens.unsqueeze(1))
            else:
                local_embs.append(x.mean(dim = -1))
        x = self.out_proj(x)
        if lens is None:
            x = x.mean(dim = -1)
        else:
            lens = lens//2
            x = x.sum(dim = -1)/lens.unsqueeze(1)
        return torch.stack(local_embs), x
    
class MixEncoder(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 conv_channels: int,
                 num_dsp: int,
                 spk_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_channels = conv_channels
        self.spk_dim = spk_dim
        self.num_dsp = num_dsp
        self.in_proj = nn.Sequential(ChannelwiseLayerNorm(in_channels),
                                     nn.Conv1d(in_channels, out_channels, 1))
        self.out_proj = nn.Sequential(nn.Conv1d(spk_dim+out_channels, out_channels, 
                                                kernel_size = 3, padding = 1),
                                      nn.BatchNorm1d(out_channels))
        for i in range(num_dsp):
            setattr(self, f'downsample{i}', DownSample(out_channels if i == 0 else out_channels + spk_dim, out_channels, 
                                                       kernel_size = 3, stride = 2))
    
    def forward(self, 
                x: torch.Tensor,
                aux: torch.Tensor):
        """
        x - torch.Tensor, shape (N, C, T)
        aux - torch.Tensor, shape (N, L, k+1, C)
        """
        N, L, K, C = aux.shape
        x = self.in_proj(x).unsqueeze(1).repeat_interleave(K, dim = 1).reshape(N*K, C, -1)
        x = self.downsample0(x)
        cache = [x]
        for i in range(1, self.num_dsp):
            aux_ = aux[:, i-1].reshape(N*K, -1).unsqueeze(-1).repeat_interleave(x.shape[-1], dim = -1)  # (N*(k+1), C, T)
            x = torch.cat([x, aux_], dim = 1)
            x = getattr(self, f"downsample{i}")(x)
            cache.append(x)
        aux_ = aux[:, i].reshape(N*K, -1).unsqueeze(-1).repeat_interleave(x.shape[-1], dim = -1)  # (N*(k+1), C, T)
        x = torch.cat([x, aux_], dim = 1)
        x = self.out_proj(x)
        return x, cache

class Encoder(nn.Module):
    def __init__(self, 
                 p_em: float,
                 num_max_spk: int,
                 in_channels: int,
                 out_channels: int,
                 spe_kernel_sizes: int,
                 num_layers: int):
        super().__init__()
        self.num_max_spk = num_max_spk
        self.p_em = p_em
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spe_kernel_sizes = spe_kernel_sizes
        self.spe_enc = SpeechEncoder(in_channels, out_channels,
                                     kernel_sizes = spe_kernel_sizes)
        self.ref_enc = MSFE(in_channels = out_channels*3,
                            out_channels = out_channels,
                            conv_channels = out_channels*2, 
                            num_dsp = num_layers)
        self.mix_enc = MixEncoder(in_channels = out_channels*3,
                                  out_channels = out_channels,
                                  conv_channels = out_channels*2, 
                                  num_dsp = num_layers,
                                  spk_dim = out_channels)
        self.Vem = nn.Parameter(torch.randn(self.mix_enc.out_channels))

    def forward(self, 
                x: torch.Tensor, 
                auxs: torch.Tensor,
                spks: Optional[list] = None,
                lens: Optional[list] = None):
        """
        x: (N, 1, To)
        auxs: [(k, 1, T), (k, 1, T), ...] (length N)
        spks: [[spk_idx1, spk_idx2, ...], [spk_idx1, spk_idx2, ...], ...] (length N)
        """
        x = self.spe_enc(x)
        loc_stack = []
        glob_stack = []
        for i in range(len(auxs)):
            aux = self.spe_enc(auxs[i])
            len_ = lens[i]//(self.spe_kernel_sizes[0]//2) if lens is not None else None
            loc_embs, glob_embs = self.forward_per_sample(aux, spks[i] if spks is not None else None, len_)
            loc_stack.append(loc_embs)
            glob_stack.append(glob_embs)
        loc_stack = torch.stack(loc_stack) # (N, L, k+1, C)
        glob_stack = torch.stack(glob_stack) # (N, k+1, C)
        out, cache = self.mix_enc(x, loc_stack)
        return x, out, cache, glob_stack      

    def forward_per_sample(self, 
                           aux: torch.Tensor, 
                           spks: Optional[list] = None,
                           lens: Optional[torch.Tensor] = None):
        """
        aux: (k, C, T)
        spks: [spk_idx1, spk_idx2, ...] (length k)
        """
        count = 0

        act_aux = []
        blk_aux = []
        blk_embs = []
        aux_idxs = []
        for i, au in enumerate(aux):
            spk = spks[i] if spks is not None else 1
            if spk != -1:
                act_aux.append(au)
                aux_idxs.append(count)
                count += 1

        while count < self.num_max_spk:
            if self.training:
                p = random.uniform(0, 1)
            else:
                p = 0
            if p > self.p_em:
                blk_aux.append(aux[count])
                aux_idxs.append(count)
            else:
                blk_embs.append(self.Vem)
            count += 1

        aux_idxs = torch.tensor(aux_idxs).to(aux.device)
        blk_embs.append(self.Vem)
        # len(act_aux) + len(blk_aux) + len(blk_embs) == num_max_spk + 1
        aux = torch.stack(act_aux + blk_aux) # ((len_act + len_blk), C, T)
        loc_embs, glob_embs = self.ref_enc(aux, lens[aux_idxs]) # (L, (len_act + len_blk), C) and ((len_act + len_blk), C)
        blk_embs = torch.stack(blk_embs) # (k+1-len_act-len_blk, C)
        loc_embs = torch.cat([loc_embs, blk_embs.unsqueeze(0).repeat_interleave(len(loc_embs), dim = 0)], dim = 1) # (L, k+1, C)
        glob_embs = torch.cat([glob_embs, blk_embs], dim = 0)
        return loc_embs, glob_embs
    
class Separator(nn.Module):
    def __init__(self,
                 out_channels: int,
                 spk_dim: int,
                 kernel_size: int,
                 num_blks: int,
                 num_tcn_layers: int,
                 num_max_spk: int):
        super().__init__()
        self.TCNs1 = Fusion_Stacked_TCNs(out_channels, out_channels*2,
                                         spk_dim = spk_dim,
                                         kernel_size = kernel_size,
                                         num_blks = num_blks,
                                         num_tcn_layers = num_tcn_layers)
        self.enc1d = nn.Conv1d(in_channels = (num_max_spk + 1)*out_channels,
                                out_channels = out_channels,
                                kernel_size = 1)
        self.TCNs2 = Stacked_TCNs(out_channels, out_channels*2,
                                  kernel_size = kernel_size,
                                  num_blks = num_blks,
                                  num_tcn_layers = num_tcn_layers,
                                  return_intermidate = True)
        self.dec1d = nn.Conv1d(in_channels = out_channels,
                               out_channels = (num_max_spk + 1)*out_channels,
                               kernel_size = 1)

    def forward(self,
                x: torch.Tensor,
                spk_embs: torch.Tensor):
        """
        x: (N*(k+1), C, T')
        spk_embs: (N, k+1, C)
        """
        N, K, C = spk_embs.shape
        x = self.TCNs1(x, spk_embs)
        x = self.enc1d(x.reshape(N, K*C, -1))
        R = self.TCNs2(x)
        x = self.dec1d(R[-1]).reshape(N*K, C, -1)
        return torch.stack(R), x

class DiarizationDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 num_max_spk: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_max_spk = num_max_spk
        self.conv = nn.Conv1d(in_channels, out_channels,
                              kernel_size = kernel_size,
                              stride = kernel_size // 2,
                              padding = kernel_size//4)
        self.linear = nn.Linear(out_channels, num_max_spk + 1)
    
    def forward(self,
                x: torch.Tensor):
        """
        x: (Nt, N, C, T)
        """
        Nt, N, C, T = x.shape
        x = self.conv(x.reshape(Nt*N, C, T)).transpose(1, 2).contiguous()
        x = self.linear(x).reshape(Nt, N, -1, self.num_max_spk + 1).sigmoid()
        return x
    
class MixDecoder(nn.Module):
    def __init__(self, 
                 out_channels: int,
                 conv_channels: int,
                 aux_dim: int,
                 num_dsp: int):
        super().__init__()
        self.out_channels = out_channels
        self.conv_channels = conv_channels
        self.aux_dim = aux_dim
        self.num_dsp = num_dsp
        for i in range(num_dsp):
            setattr(self, f'upsample{i}', UpSample(out_channels + aux_dim, out_channels,
                                                   kernel_size = 3, stride = 2))
    
    def forward(self, 
                x: torch.Tensor,
                cache: torch.Tensor):
        """
        x - torch.Tensor, shape (N*K, C, T')
        cache - list, [(N*K, C, T1), ...]
        """
        for i in range(self.num_dsp):
            #x = torch.cat([x, cache[-(i+1)]], dim = 1)
            x = getattr(self, f"upsample{i}")(x, cache[-(i+1)])
        return x
    
class MaskModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super().__init__()
        for i in range(3):
            setattr(self, f'conv{i}', nn.Conv1d(in_channels = in_channels,
                                                out_channels = out_channels,
                                                kernel_size = 1))
        self.relu = nn.ReLU(inplace = True)

    def forward(self, X):
        """
        X: (N, C, T0)
        """
        out = []
        for i in range(3):
            out.append(getattr(self, f'conv{i}')(X))
        return self.relu(torch.concat(out, dim = 1))
    
class SpeechDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_sizes: list):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        stride = kernel_sizes[0]/2
        paddings = [int((kernel_size - stride)/2) for kernel_size in kernel_sizes]
        for i, (kernel_size, padding) in enumerate(zip(kernel_sizes, paddings)):
            setattr(self, f'conv{i}', nn.ConvTranspose1d(in_channels = in_channels,
                                                         out_channels = out_channels,
                                                         kernel_size = kernel_size,
                                                         padding = padding,
                                                         stride = int(stride)))

    def forward(self, X):
        """
        X: (N, C, T0)
        """
        out = []
        for i in range(len(self.kernel_sizes)):
            out.append(getattr(self, f'conv{i}')(X))
        return torch.concat(out, dim = 1)

class InteractionModule(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int):
        super().__init__()
        self.seq = nn.Sequential(nn.Conv1d(in_channels, out_channels, 
                                           kernel_size = kernel_size,
                                           padding = (kernel_size - 1)//2),
                                 nn.ReLU(inplace = True))
    
    def forward(self, 
                x: torch.Tensor,
                s: torch.Tensor):
        """
        x, Tensor. (N, k+1, T')
        s, Tensor. (N, (k+1), 3, T)
        """
        T = s.shape[-1]
        N, K = x.shape[:2]
        x = nn.functional.interpolate(x, size = (T), mode = 'linear')
        x = self.seq(x.reshape(N*K, -1, T)).reshape(N, K, -1, T)
        return x*s
    
class ExtractionDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_max_spk: int,
                 int_kernel_size: int,
                 spe_kernel_sizes: list,
                 ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_max_spk = num_max_spk
        self.msk_module = MaskModule(in_channels, out_channels)
        self.int_module = InteractionModule(1, 1, int_kernel_size)
        self.spe_decoder = SpeechDecoder(3*out_channels, 1, kernel_sizes = spe_kernel_sizes)
        
    def forward(self, 
                R: torch.Tensor,
                X: torch.Tensor,
                D: torch.Tensor):
        """
        R, Tensor. (N*(k+1), C, T)
        X, Tensor. (N, 3*C, T)
        D, Tensor. (N, k+1, T')
        """
        NK, C, T = R.shape
        N, K = D.shape[:2]
        M = self.msk_module(R).reshape(N, -1, 3*C, T)  # (N, k+1, 3C, T)
        M = X.unsqueeze(1)*M    # (N, k+1, 3C, T)
        S_tilde = self.spe_decoder(M.reshape(-1, 3*C, T))   # (N*(k+1), 3, T)
        S_hat = self.int_module(D, S_tilde.reshape(N, K, 3, -1))
        return S_hat    

class SUDx(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 spk_dim: int,
                 spe_kernel_sizes: int,
                 diar_kernel_size: int,
                 int_kernel_size: int,
                 num_tcn_blks: int,
                 num_tcn_layers: int,
                 num_max_spk: int,
                 ):
        super().__init__()
        # 1. Speech Encoder
        self.encoder = Encoder(p_em = 0.3,
                               num_max_spk = num_max_spk,
                               in_channels = in_channels,
                               out_channels = out_channels,
                               spe_kernel_sizes = spe_kernel_sizes,
                               num_layers = 3)
        # 4. Separator
        self.separator = Separator(out_channels,
                                   spk_dim = spk_dim,
                                   kernel_size = 3,
                                   num_blks = num_tcn_blks,
                                   num_tcn_layers = num_tcn_layers,
                                   num_max_spk = num_max_spk)
        # 5. Mix Decoder
        self.mix_dec = MixDecoder(out_channels, out_channels*2, out_channels, 3)
        # 5. Diarization Decoder
        self.diar_dec = DiarizationDecoder(out_channels, out_channels,
                                           kernel_size = diar_kernel_size,
                                           num_max_spk = num_max_spk)
        # 6. Extraction Decoder
        self.ext_dec = ExtractionDecoder(out_channels, out_channels, 
                                         num_max_spk = num_max_spk,
                                         int_kernel_size = int_kernel_size,
                                         spe_kernel_sizes = spe_kernel_sizes)
    
    def forward(self, 
                inputs: dict, 
                targets: Optional[dict] = None):
        """
        Args:
            inputs (dict):
                - mixs (Tensor): Mixture waveform tensor of shape (batch_size, 1, seq_samples).
                - refs (list): A list of length N,
                Each item is a tensor of shape (num_speakers_i, max_len_i) representing the
                reference sources for that sample.
                - lens (list): A list of length N, where each element is a list containing the
                valid sequence lengths of each reference source, e.g.,
                [[seq_len_1, seq_len_2, ...], ...].
            targets (Optional[dict]):
                - Note: id of zero speaker is -1
        """
        E = []
        Vs = []
        if self.training:
            targets_ = {'spks': [],
                        'diars': [],
                        'srcs': []}
        else:
            targets_ = targets
        X, X_, cache, E = self.encoder(inputs['mixs'], inputs['refs'], targets['spks'] if targets is not None else None, inputs['lens']) 
        # X (N, 3*C, T)
        # X_ (N*(k+1), C, T)
        # E (N, k+1, C)
        # shuffle
        #print(X.shape, X_.shape, E.shape)
        if self.training:
            NK, C, T = X_.shape
            n = list(range(self.encoder.num_max_spk))
            random.shuffle(n)
            randperm = torch.tensor(n + [self.encoder.num_max_spk]).to(X.device)
            for i in range(len(E)):
                targets_['spks'].append(targets['spks'][i][randperm[:-1]])
                targets_['diars'].append(targets['diars'][i][randperm])
                targets_['srcs'].append(targets['srcs'][i][randperm])
                ### need shuffle X and E
            X_ = X_.reshape(len(E), -1, C, T)[:, randperm].reshape(NK, C, T)
            E = E[:, randperm]
        R, X_ = self.separator(X_, E)    # (Nt, N, C, T)
        X_ = self.mix_dec(X_, cache)
        D = self.diar_dec(R)        # (Nt, N, T', k+1)
        S = self.ext_dec(X_, X, D[-1].detach().transpose(1, 2))
        return {"spk_embs": E[:, :-1, :],
                "diar_probs": D,
                "ext_preds": S}, targets_