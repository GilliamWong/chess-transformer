import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.optim._muon import _adjust_lr
from torch.optim.optimizer import _to_scalar


#torch's _zeropower_via_newtonschulz on a stack of matrices no taller than wide, each normalised by its own norm
def newton_schulz(stack: Tensor, group: dict) -> Tensor:
    a, b, c = group["ns_coefficients"]
    ortho = stack.bfloat16()
    ortho = ortho / ortho.norm(dim=(1, 2), keepdim=True).clamp(min=group["eps"])
    for _ in range(group["ns_steps"]):
        gram = ortho @ ortho.mT
        gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.baddbmm(ortho, gram_update, ortho, beta=a)
    return ortho

#the rank's equal part of a stack, zero-padded to a multiple of the world size
def share_of(stack: Tensor, rank: int, world_size: int) -> Tensor:
    share = math.ceil(stack.shape[0] / world_size)
    padded = F.pad(stack, (0, 0, 0, 0, 0, share * world_size - stack.shape[0]))
    return padded[rank * share:(rank + 1) * share]

#every rank's shares, (world size, shares size); untraced, as the collective splits the compiled step into two cuda graphs
@torch.compiler.disable
def gather_shares(shares: Tensor) -> Tensor:
    world_size = dist.get_world_size()
    gathered = torch.empty(world_size * shares.numel(), dtype=shares.dtype, device=shares.device)
    dist.all_gather_single(gathered, shares)
    return gathered.view(world_size, -1)


#torch.optim.Muon with each shape's matrices orthogonalised in batched matmuls, same arithmetic and state_dict. Under
#torch.distributed each rank orthogonalises a share and one gather brings the rest: gradients must match on every rank
class BatchedMuon(torch.optim.Muon):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rank, self.world_size = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        #a job: the params of one shape in one group, with their updates stacked (tall ones transposed)
        jobs: list[tuple[dict, list[Tensor], Tensor]] = []
        for group in self.param_groups:
            params: list[Tensor] = []
            grads: list[Tensor] = []
            buffers: list[Tensor] = []
            self._init_group(group, params, grads, buffers)
            momentum = group["momentum"]
            for buffer, grad in zip(buffers, grads):
                buffer.lerp_(grad, 1 - momentum)
            updates = [grad.lerp(buffer, momentum) for buffer, grad in zip(buffers, grads)] if group["nesterov"] else buffers

            by_shape: dict[tuple[int, int], list[int]] = {}
            for index, param in enumerate(params):
                rows, columns = param.shape
                by_shape.setdefault((min(rows, columns), max(rows, columns)), []).append(index)
            for indices in by_shape.values():
                stack = torch.stack([(updates[i].T if params[i].shape[0] > params[i].shape[1] else updates[i]).bfloat16() for i in indices])
                jobs.append((group, [params[i] for i in indices], stack))

        if self.world_size == 1:
            orthos = [newton_schulz(stack, group) for group, _, stack in jobs]
        else:
            shares = [newton_schulz(share_of(stack, self.rank, self.world_size), group) for group, _, stack in jobs]
            gathered = gather_shares(torch.cat([share.flatten() for share in shares]))
            orthos = []
            start = 0
            for (_, params, _), share in zip(jobs, shares):
                #rank by rank, the order of the padded stack
                orthos.append(gathered[:, start:start + share.numel()].reshape(-1, *share.shape[1:])[:len(params)])
                start += share.numel()

        for (group, params, _), ortho in zip(jobs, orthos):
            lr = _to_scalar(group["lr"])
            decay = 1 - lr * group["weight_decay"]
            for param, matrix in zip(params, ortho):
                adjusted_lr = _adjust_lr(lr, group["adjust_lr_fn"], param.shape)
                param.mul_(decay)
                param.add_(matrix.T if param.shape[0] > param.shape[1] else matrix, alpha=-adjusted_lr)
        return loss
