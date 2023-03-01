import torch
from torch import nn
from typing import Tuple
import torch
from mmcv.runner import force_fp32
from torch import nn


def bev_pool(x, geom_feats, B, D, H, W):
    assert x.shape[0] == geom_feats.shape[0]

    ranks = (
        geom_feats[:, 0] * (W * D * B)
        + geom_feats[:, 1] * (D * B)
        + geom_feats[:, 2] * B
        + geom_feats[:, 3]
    )
    _,indices = ranks.sort()
    x, geom_feats, ranks = x[indices], geom_feats[indices], ranks[indices]
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[1:] = ranks[1:] != ranks[:-1]
    interval_starts = torch.where(kept)[0].int()
    interval_lengths = torch.zeros_like(interval_starts)
    interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
    interval_lengths[-1] = x.shape[0] - interval_starts[-1]
    geom_feats = geom_feats.int()
    torch.ops.load_library("/home/ubuntu/workstation/motorai/tensorrt_torchscript_demo/bev_pool/build/lib.linux-x86_64-cpython-38/bev_pool_forward.cpython-38-x86_64-linux-gnu.so")
    x = torch.ops.my_ops.bev_pool_forward(
        x,
        geom_feats.long(),
        interval_lengths.long(),
        interval_starts.long(),
        B,
        D.long(),
        H.long(),
        W.long(),
    )
    # x = QuickCumsumCuda.apply(feats, coords, ranks, B, D, H, W)
    x = x.permute(0, 4, 1, 2, 3).contiguous()
    return x


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]]).cuda()
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]]).cuda()
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    ).cuda()
    return dx, bx, nx

class BEVPool(nn.Module):
    def __init__(
        self,
        xbound: Tuple[float, float, float] = (-54.0, 54.0, 0.3),
        ybound: Tuple[float, float, float] = (-54.0, 54.0, 0.3),
        zbound: Tuple[float, float, float] = (-10.0, 10.0, 20.),
        dbound: Tuple[float, float, float] = (1.0, 60.0, 0.5),
    ) -> None:
        super().__init__()
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

    def forward(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        # B, N, D, H, W, C = B.cuda(), N.cuda(), D.cuda(), H.cuda(), W.cuda(), C.cuda() 
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, torch.Tensor([B])[0].long(), self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final



bevpool = BEVPool()

geom = torch.load("data/geom.pt")
x = torch.load("data/x.pt")

# x = bevpool(geom,x)


bev_trace = torch.jit.trace(bevpool, (geom,x))

bev_trace.save("torchscript_engines/bev_pool.pt")
