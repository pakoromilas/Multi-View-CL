import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import diffdist
import torch.distributed as dist
import itertools
from einops import rearrange
from typing import Optional, Literal


from utils import utils


def gather(z):
    gather_z = [torch.zeros_like(z) for _ in range(torch.distributed.get_world_size())]
    gather_z = diffdist.functional.all_gather(gather_z, z)
    gather_z = torch.cat(gather_z)

    return gather_z


def accuracy(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    acc = (topk == labels).all(1).float()
    return acc


def mean_cumulative_gain(logits, labels, k):
    topk = torch.sort(logits.topk(k, dim=1)[1], 1)[0]
    labels = torch.sort(labels, 1)[0]
    mcg = (topk == labels).float().mean(1)
    return mcg


def mean_average_precision(logits, labels, k):
    # TODO: not the fastest solution but looks fine
    argsort = torch.argsort(logits, dim=1, descending=True)
    labels_to_sorted_idx = torch.sort(torch.gather(torch.argsort(argsort, dim=1), 1, labels), dim=1)[0] + 1
    precision = (1 + torch.arange(k, device=logits.device).float()) / labels_to_sorted_idx
    return precision.sum(1) / k


class NTXent(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., multiplier=2, pwe=False, average=False, mv=False, distributed=False):
        super().__init__()
        self.tau = tau
        self.pwe = pwe
        self.average = average
        self.k_views = multiplier
        self.multiplier = 2
        self.mv = mv
        self.distributed = distributed
        self.norm = 1.

    def forward(self, z):
        n = z.shape[0]
        assert n % self.k_views == 0

        z = F.normalize(z, p=2, dim=1) / np.sqrt(self.tau)
        
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.k_views)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            
            for m in range(self.k_views):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.k_views + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        if self.pwe:
            n_real = int(n / self.k_views)
            views = [z[i * n_real : (i + 1) * n_real] for i in range(self.k_views)]           
            pairs = list(itertools.combinations(views, 2))
            loss = 0 
            acc = 0
            for pair in pairs:
                z = torch.concat(pair)
                cur_loss, cur_acc = self.pairwise_loss(z)
                loss += cur_loss
                acc += cur_acc
            loss = loss / len(pairs)
            acc = acc / len(pairs)
        elif self.average:
            n_real = int(n / self.k_views)
            views = [z[i * n_real : (i + 1) * n_real] for i in range(self.k_views)]            
            pairs = [(view, sum(views[:i] + views[i+1:]) / (len(views) - 1)) for i, view in enumerate(views)]
            
            loss = 0 
            acc = 0
            for pair in pairs:
                z = torch.concat(pair)
                cur_loss, cur_acc = self.pairwise_loss(z)
                loss += cur_loss
                acc += cur_acc
            loss = loss / len(pairs)
            acc = acc / len(pairs)
        else:
            if self.mv:
                self.multiplier = self.k_views
            loss, acc = self.pairwise_loss(z)

        return loss, acc

    
    def pairwise_loss(self, z):
                                
        n = z.shape[0]

        logits = z @ z.t()
        logits[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER

        logprob = F.log_softmax(logits, dim=1)

        # choose all positive objects for an example, for i it would be (i + k * n/m), where k=0...(m-1)
        m = self.multiplier
        labels = (np.repeat(np.arange(n), m) + np.tile(np.arange(m) * n//m, n)) % n
        # remove labels pointet to itself, i.e. (i, i)
        labels = labels.reshape(n, m)[:, 1:].reshape(-1)

        # TODO: maybe different terms for each process should only be computed here...
        loss = -logprob[np.repeat(np.arange(n), m-1), labels].sum() / n / (m-1) / self.norm

        # zero the probability of identical pairs
        pred = logprob.data.clone()
        pred[np.arange(n), np.arange(n)] = -self.LARGE_NUMBER
        acc = accuracy(pred, torch.LongTensor(labels.reshape(n, m-1)).to(logprob.device), m-1)

        return loss, acc
    

class NTXentHPS(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    def __init__(self, tau=1., gpu=None, multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.distributed = distributed
        self.multiplier = multiplier


    def forward(self, z, get_map=False):
        n = z.shape[0]
        z = F.normalize(z, p=2, dim=1)

        if self.distributed:
            z = self.gather_distributed(z)
            n = z.shape[0]
        
        z_real = z[:int(n / 2)]
        z_aug = z[int(n / 2):]
        
        sim_matrix_real = torch.exp(torch.mm(z_real, z_real.t().contiguous()) / self.tau)
        mask = (torch.ones_like(sim_matrix_real) - torch.eye(int(n / 2), device=sim_matrix_real.device)).bool()
        # [B, B-1]
        sim_matrix_real = sim_matrix_real.masked_select(mask).view(int(n / 2), -1)
        #energy_real = torch.log(sim_matrix_real.exp().sum(1)).mean()
        
        sim_matrix_aug = torch.exp(torch.mm(z_aug, z_aug.t().contiguous()) / self.tau)
        mask = (torch.ones_like(sim_matrix_aug) - torch.eye(int(n / 2), device=sim_matrix_aug.device)).bool()
        # [B, B-1]
        sim_matrix_aug = sim_matrix_aug.masked_select(mask).view(int(n / 2), -1)
        
        pos_sim = torch.exp(torch.sum(z_real * z_aug, dim=-1) / self.tau)
        loss = - (torch.log(pos_sim / (sim_matrix_real.sum(dim=-1) * sim_matrix_aug.sum(dim=-1)))).mean()
    
        return loss, 0
    
    def gather_distributed(self, z):
        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        # TODO: try to rewrite it with pytorch official tools
        z_list = diffdist.functional.all_gather(z_list, z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
        # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        z_sorted = []
        
        for m in range(self.multiplier):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * self.multiplier + m])
        z = torch.cat(z_sorted, dim=0)
        return z


class MVINFONCE(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.k_views = multiplier
        self.distributed = distributed

    def forward(self, z):
        n, d = z.shape
        assert n % self.k_views == 0

        z = F.normalize(z, p=2, dim=1)
        
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            # all_gather fills the list as [<proc0>, <proc1>, ...]
            # TODO: try to rewrite it with pytorch official tools
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.k_views)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            
            for m in range(self.k_views):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.k_views + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        n_real = int(n / self.k_views)

        # Energy
        views = [z[i * n_real : (i + 1) * n_real] for i in range(self.k_views)]
        neg_sim = torch.exp(torch.mm(z, z.t().contiguous()) / self.tau) 
        mask = (torch.ones_like(neg_sim) - torch.eye(n, device=neg_sim.device)).bool()
        neg_sim = neg_sim.masked_select(mask).view(n, -1)
        neg_sims = torch.split(neg_sim, n_real)
        neg_sims = torch.cat(neg_sims, dim=1)
        neg_sim = neg_sims.sum(dim=-1) 

        # Alignment
        views = torch.stack(views)
        all_products = torch.einsum('ind,jnd->ijnd', views, views)
    
        # Create a mask to select only the upper triangular part (excluding diagonal)
        mask = torch.triu(torch.ones(self.k_views, self.k_views), diagonal=1).bool()
        
        # Apply the mask and reshape
        pos_prods = all_products[mask].reshape(-1, n_real, d)
        # -- sum
        pos_sim = torch.exp(torch.sum(pos_prods, dim=-1) / self.tau)
        pos_sim = 2 * torch.sum(pos_sim, dim=0)

        loss = - (torch.log(pos_sim / neg_sim)).mean()    
        return loss, 0



class CL_1(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """
    LARGE_NUMBER = 1e9

    def __init__(self, tau=1., multiplier=2, distributed=False):
        super().__init__()
        self.tau = tau
        self.k_views = multiplier
        self.distributed = distributed

    def forward(self, z):
        n, d = z.shape
        assert n % self.k_views == 0

        z = F.normalize(z, p=2, dim=1)
        
        if self.distributed:
            z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
            
            z_list = diffdist.functional.all_gather(z_list, z)
            # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
            z_list = [chunk for x in z_list for chunk in x.chunk(self.k_views)]
            # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
            z_sorted = []
            
            for m in range(self.k_views):
                for i in range(dist.get_world_size()):
                    z_sorted.append(z_list[i * self.k_views + m])
            z = torch.cat(z_sorted, dim=0)
            n = z.shape[0]

        n_real = int(n / self.k_views)

        # Energy
        views = [z[i * n_real : (i + 1) * n_real] for i in range(self.k_views)]
        neg_sims = []
        for idx, z_current in enumerate(views):
            z_rest = torch.stack(views[:idx] + views[idx+1:])
            z_rest = z_rest.view(-1, z_rest.size(-1))
            # Calculate the similarity matrix
            sim_matrix = torch.exp(torch.mm(z_current, z_rest.t().contiguous()) / self.tau)
            
            neg_sims.append(sim_matrix.sum(dim=-1))

        neg_sims = torch.stack(neg_sims)
        
        # Alignment
        views = torch.stack(views)
        all_products = torch.einsum('ind,jnd->ijnd', views, views)

        pos_sim = []
        for k_1 in range(self.k_views):
            cur_sim = []
            for k_2 in range(self.k_views):
                if k_1 == k_2:
                    continue
                cur_sim.append(all_products[k_1, k_2])
            cur_sim = torch.stack(cur_sim)
            cur_sim = torch.exp(torch.sum(cur_sim, dim=-1) / self.tau)
            pos_sim.append(cur_sim.sum(dim=0))

        
        pos_sim = torch.stack(pos_sim)
        # Overall
        loss = (torch.mean(-torch.log(pos_sim)) + torch.log(neg_sims)).mean()
        return loss, 0




class MVDHEL(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    def __init__(self, tau=0.2, multiplier=2, distributed=False, pwe=False, average=False,):
        super().__init__()
        self.tau = tau
        self.k_views = multiplier
        self.distributed = distributed
        self.pwe = pwe
        self.average = average
        if self.pwe or self.average:
            self.pairwise_loss = NTXentHPS(tau=self.tau, multiplier=2, distributed=False)

    def forward(self, z):
        
        z = F.normalize(z, p=2, dim=1)
        if self.distributed:
            z = self.gather_distributed(z)

        n, d = z.shape        
        assert n % self.k_views == 0

        n_real = int(n / self.k_views)
        # Reshape the tensor to (k_views, n_real, d)
        views = [z[i * n_real : (i + 1) * n_real] for i in range(self.k_views)]
        
        if self.pwe:
            pairs = list(itertools.combinations(views, 2))
            loss = 0 
            for pair in pairs:
                z = torch.concat(pair)
                cur_loss, _ = self.pairwise_loss(z)
                loss += cur_loss
            loss = loss / len(pairs)
        elif self.average:
            pairs = [(view, sum(views[:i] + views[i+1:]) / (len(views) - 1)) for i, view in enumerate(views)]
            
            loss = 0 
            for pair in pairs:
                z = torch.concat(pair)
                cur_loss, _ = self.pairwise_loss(z)
                loss += cur_loss
            loss = loss / len(pairs)
        else:
            # Uniformity
            neg_sims = []
            for z_current in views:
                
                # Calculate the similarity matrix
                sim_matrix = torch.exp(torch.mm(z_current, z_current.t().contiguous()) / self.tau)
                
                # Create a mask to remove self-similarities
                mask = (torch.ones_like(sim_matrix) - torch.eye(n_real, device=sim_matrix.device)).bool()
                
                # Apply the mask and reshape the similarity matrix
                sim_matrix = sim_matrix.masked_select(mask).view(n_real, -1)
                
                # Sum the similarities for negative samples
                neg_sims.append(sim_matrix.sum(dim=-1))

            neg_sims = torch.stack(neg_sims)

            # Alignment

            views = torch.stack(views)
            all_products = torch.einsum('ind,jnd->ijnd', views, views)
        
            # Create a mask to select only the upper triangular part (excluding diagonal)
            mask = torch.triu(torch.ones(self.k_views, self.k_views), diagonal=1).bool()
            
            # Apply the mask and reshape
            pos_prods = all_products[mask].reshape(-1, n_real, d)
            # -- sum
            pos_prods = torch.exp(torch.sum(pos_prods, dim=-1) / self.tau)
            pos_sim = torch.sum(pos_prods, dim=0)
           
            # Overall
            loss = - (torch.log(pos_sim / torch.prod(neg_sims, dim=0))).mean()
            
        return loss, 0

    def gather_distributed(self, z):
        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        # TODO: try to rewrite it with pytorch official tools
        z_list = diffdist.functional.all_gather(z_list, z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(self.k_views)]
        # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        z_sorted = []
        
        for m in range(self.k_views):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * self.k_views + m])
        z = torch.cat(z_sorted, dim=0)
        return z



class CL_2(nn.Module):
    """
    Contrastive loss with distributed data parallel support
    """

    def __init__(self, tau=0.2, multiplier=2, distributed=False, pwe=False, average=False,):
        super().__init__()
        self.tau = tau
        self.k_views = multiplier
        self.distributed = distributed
        self.pwe = pwe
        self.average = average
        if self.pwe or self.average:
            self.pairwise_loss = NTXentHPS(tau=self.tau, multiplier=2, distributed=False)

    def forward(self, z):
        
        z = F.normalize(z, p=2, dim=1)
        if self.distributed:
            z = self.gather_distributed(z)

        n, d = z.shape        
        assert n % self.k_views == 0

        n_real = int(n / self.k_views)
        # Reshape the tensor to (k_views, n_real, d)
        views = [z[i * n_real : (i + 1) * n_real] for i in range(self.k_views)]
        
        if self.pwe:
            pairs = list(itertools.combinations(views, 2))
            loss = 0 
            for pair in pairs:
                z = torch.concat(pair)
                cur_loss, _ = self.pairwise_loss(z)
                loss += cur_loss
            loss = loss / len(pairs)
        elif self.average:
            pairs = [(view, sum(views[:i] + views[i+1:]) / (len(views) - 1)) for i, view in enumerate(views)]
            
            loss = 0 
            for pair in pairs:
                z = torch.concat(pair)
                cur_loss, _ = self.pairwise_loss(z)
                loss += cur_loss
            loss = loss / len(pairs)
        else:
            # Uniformity
            neg_sims = []
            for z_current in views:
                
                # Calculate the similarity matrix
                sim_matrix = torch.exp(torch.mm(z_current, z_current.t().contiguous()) / self.tau)
                
                # Create a mask to remove self-similarities
                mask = (torch.ones_like(sim_matrix) - torch.eye(n_real, device=sim_matrix.device)).bool()
                
                # Apply the mask and reshape the similarity matrix
                sim_matrix = sim_matrix.masked_select(mask).view(n_real, -1)
                
                # Sum the similarities for negative samples
                neg_sims.append(sim_matrix.sum(dim=-1))

            neg_sims = torch.stack(neg_sims)

            # Alignment

            views = torch.stack(views)
            all_products = torch.einsum('ind,jnd->ijnd', views, views)

            pos_sim = []
            for k_1 in range(self.k_views):
                cur_sim = []
                for k_2 in range(self.k_views):
                    if k_1 == k_2:
                        continue
                    cur_sim.append(all_products[k_1, k_2])
                cur_sim = torch.stack(cur_sim)
                cur_sim = torch.exp(torch.sum(cur_sim, dim=-1) / self.tau)
                pos_sim.append(cur_sim.sum(dim=0))

            
            pos_sim = torch.stack(pos_sim)
            # Overall
            loss = torch.mean(-torch.log(pos_sim) + torch.sum(torch.log(neg_sims), dim=0))
            #loss = - (torch.log(pos_sim / torch.prod(neg_sims, dim=0))).mean()
            
        return loss, 0

    def gather_distributed(self, z):
        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        # TODO: try to rewrite it with pytorch official tools
        z_list = diffdist.functional.all_gather(z_list, z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(self.k_views)]
        # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        z_sorted = []
        
        for m in range(self.k_views):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * self.k_views + m])
        z = torch.cat(z_sorted, dim=0)
        return z


class PVCLoss(nn.Module):
    def __init__(
        self,
        tau: float = 0.2,
        multiplier: int = 2,
        distributed: bool =False,
        method: Literal["arithmetic", "geometric"] = "geometric"
    ):
        """
        Poly-View Contrastive Loss implementation following the paper.
        
        Args:
            tau: Temperature parameter
            method: Method for aggregating losses ("arithmetic" or "geometric")
        """
        super().__init__()
        self.tau = tau
        self.multiplier = multiplier
        self.distributed = distributed
        self.method = method
        
    
    def forward(
        self,
        z: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the loss.
        
        Args:
            features: Tensor of shape [k*m, d] containing L2-normalized features
                     for k samples and m views
            m: Number of views. If None, will try to infer from tensor shape
            
        Returns:
            Scalar loss value
        """
        
        z = F.normalize(z, p=2, dim=1)
        if self.distributed:
            z = self.gather_distributed(z)
        
        k = z.shape[0] // self.multiplier # batch size
        
        views = [z[i * k : (i + 1) * k] for i in range(self.multiplier)]
        views = torch.stack(views, dim=1)
        
        # build score matrix [k, m, m, k]
        scores = torch.einsum("jmd,knd->jmnk", views, views) / self.tau
        
        # track the losses for each alpha
        losses_alpha = []
        
        # iterate over alpha and beta
        for alpha in range(self.multiplier):
            losses_alpha_beta = []
            for beta in range(self.multiplier):
                # skip on-diagonal terms
                if alpha != beta:
                    logits = scores[:, alpha]  # [k, m, k]
                    labels = torch.arange(k, device=z.device) + beta * k  # [k]
                    mask = self.get_mask(beta, k, z.device)  # [k, m, k]
                    logits = (logits - mask * 1e9)
                    logits = rearrange(logits, 'k m k2 -> k (m k2)') # [k, m * k]
                    loss_alpha_beta = F.cross_entropy(logits,labels) #[k]
                    losses_alpha_beta.append(loss_alpha_beta) # [k, m-1]
            
            losses_alpha_beta = torch.stack(losses_alpha_beta, dim=-1)
            
            # aggregate over betas for each alpha
            if self.method == "arithmetic":
                loss_alpha = torch.logsumexp(losses_alpha_beta, dim=-1) - torch.log(torch.tensor(k, device=z.device))
            else:  # geometric
                loss_alpha = torch.mean(losses_alpha_beta, dim=-1)
                
            losses_alpha.append(loss_alpha)
        
        # build final loss matrix
        losses = torch.stack(losses_alpha, dim=-1)  # [k,m]
        
        # take expectations
        sample_losses = torch.mean(losses, dim=-1)  # [k]
        loss = torch.mean(sample_losses)  # scalar
        
        return loss, 0

    def get_mask(self, beta: int, k: int, device: torch.device) -> torch.Tensor:
        """
        The self-supervised target is j=i, beta=alpha. Produce a mask that 
        removes the contribution of j=i, beta!=alpha.
        
        Args:
            beta: Current beta index
            k: Batch size
            m: Number of views
            device: Device to create tensors on
            
        Returns:
            Tensor of shape [k,m,k] of zeros with ones on:
            - The self-sample index
            - The betas not equal to alpha
        """
        # mask the sample [k, 1, k]
        mask_sample = rearrange(torch.eye(k, device=device), "ka kb -> ka 1 kb")
        
        # mask the beta-th view [k, m, 1]
        mask_beta = rearrange(torch.ones(self.multiplier, device=device), "m -> 1 m 1")
        mask_beta[:, beta] = 0
        
        return mask_beta * mask_sample

    def gather_distributed(self, z):
        z_list = [torch.zeros_like(z) for _ in range(dist.get_world_size())]
        # all_gather fills the list as [<proc0>, <proc1>, ...]
        # TODO: try to rewrite it with pytorch official tools
        z_list = diffdist.functional.all_gather(z_list, z)
        # split it into [<proc0_aug0>, <proc0_aug1>, ..., <proc0_aug(m-1)>, <proc1_aug(m-1)>, ...]
        z_list = [chunk for x in z_list for chunk in x.chunk(self.multiplier)]
        # sort it to [<proc0_aug0>, <proc1_aug0>, ...] that simply means [<batch_aug0>, <batch_aug1>, ...] as expected below
        z_sorted = []
        
        for m in range(self.multiplier):
            for i in range(dist.get_world_size()):
                z_sorted.append(z_list[i * self.multiplier + m])
        z = torch.cat(z_sorted, dim=0)
        return z
