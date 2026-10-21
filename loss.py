import torch
import torch.nn.functional as F
import numpy as np
from geomloss import SamplesLoss
from utils import standard_normal_logprob, sample_index
import os


class OTLoss:
    def __init__(self, args, device):
        self.ot_solver = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=args.sinkhorn_blur,
            scaling=args.sinkhorn_scaling,
            debias=True,
        )
        self.device = device

    def __call__(self, a_i, x_i, b_j, y_j, requires_grad=True):
        a_i = a_i.to(self.device)
        x_i = x_i.to(self.device)
        b_j = b_j.to(self.device)
        y_j = y_j.to(self.device)
        if requires_grad:
            x_i.requires_grad_(True)
            y_j.requires_grad_(True)
        return self.ot_solver(a_i, x_i, b_j, y_j)


def _to_device_float(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.from_numpy(x).float().to(device)


# ============================================================
# spatial_regulaization
# ============================================================
def _spatial_consistency_penalty(z, spatial_coords, accel=False, max_pairs=5000):
    
    N = z.shape[0]
    if N < 2:
        return torch.tensor(0.0, device=z.device)

    if accel or N > 2000:
        subset = min(max_pairs, N * (N - 1) // 2)
        idx1 = torch.randint(0, N, (subset,), device=z.device)
        idx2 = torch.randint(0, N, (subset,), device=z.device)
        z1, z2 = z[idx1], z[idx2]
        c1, c2 = spatial_coords[idx1], spatial_coords[idx2]
        z_dists = torch.norm(z1 - z2, dim=1)
        sp_dists = torch.norm(c1 - c2, dim=1)
        n_items = subset
    else:
        z_dists = torch.cdist(z, z).flatten()
        sp_dists = torch.cdist(spatial_coords, spatial_coords).flatten()
        n_items = N * N

    z_dists = z_dists / (z_dists.max() + 1e-8)
    sp_dists = sp_dists / (sp_dists.max() + 1e-8)

    penalty = torch.sum((1.0 - z_dists) * sp_dists) / n_items
    return penalty
# ============================================================


def _compute_direction_loss(
    device,
    args,
    data,
    spatial_coords,       # 【新增】spatial_coords 放在 data 后面
    label,
    model,
    velocity,
    direction="backward",
    ot_loss_fn=None,
):
    
    FIXED_LAMBDA_OT = 5.0
    FIXED_LAMBDA_ENERGY = 1.0
    FIXED_LAMBDA_STD = 10.0
    FIXED_LAMBDA_DENSITY = 0.2
    FIXED_LAMBDA_DISP = 10.0
   
    if not hasattr(args, "top_k_reg") or args.top_k_reg <= 0:
        args.top_k_reg = 1.0

    if direction == "backward":
        itps = args.int_tps[::-1]
        tps = args.timepoints_train[::-1]
    else:
        itps = args.int_tps
        tps = args.timepoints_train

    assert len(itps) == len(tps), "itps and tps must have same length"
    num_steps = len(tps) - 1
    if num_steps <= 0:
        return (torch.tensor(0.0, device=device),) * 11

    ode_func = model.chain[0].odefunc.odefunc

    idx0, _ = sample_index(args.batch_size, data, label, tps[0], w=None)
    x0_np = data[idx0]
    if args.training_noise > 0.0:
        x0_np = x0_np + np.random.randn(*x0_np.shape) * args.training_noise
    z = _to_device_float(x0_np, device)

    deltas = []
    energy_terms = []
    ot_terms = []
    std_terms = []
    density_terms = []
    disp_terms = []
    z_new_list = []

    spatial_reg_terms = []

    
    ratio_stats, cos_stats = [], []
    pred_disp_stats, real_disp_stats = [], []

    for i in range(num_steps):
        t0 = itps[i]
        t1 = itps[i + 1]

        t0_tensor = torch.tensor(t0, dtype=torch.float32, device=device)
        t1_tensor = torch.tensor(t1, dtype=torch.float32, device=device)

        idx_start, _ = sample_index(args.batch_size, data, label, tps[i], w=None)
        x_start_np = data[idx_start]
        if args.training_noise > 0.0:
            x_start_np = x_start_np + np.random.randn(*x_start_np.shape) * args.training_noise
        x_start = _to_device_float(x_start_np, device)

        # ===== saptial =====
        if (i == 0) and (spatial_coords is not None) and (getattr(args, "spatial_reg_strength", 0.0) > 0):
            coords_start_np = spatial_coords[idx_start]
            coords_start = _to_device_float(coords_start_np, device)
            penalty = _spatial_consistency_penalty(
                x_start, coords_start,
                accel=getattr(args, "spatial_reg_accel", False),
            )
            spatial_reg_terms.append(getattr(args, "spatial_reg_strength") * penalty)
        # ==================================

        idx_end, _ = sample_index(args.batch_size, data, label, tps[i + 1], w=None)
        x_end_np = data[idx_end]
        if args.training_noise > 0.0:
            x_end_np = x_end_np + np.random.randn(*x_end_np.shape) * args.training_noise
        x_end = _to_device_float(x_end_np, device)

        centroid_start = torch.mean(x_start, dim=0)
        centroid_end = torch.mean(x_end, dim=0)
        displacement = torch.norm(centroid_end - centroid_start, p=2)

        if i == 0:
            x_input = z
        else:
            x_input = torch.cat((z, x_start), dim=0)

        zero = torch.zeros(x_input.shape[0], 1, device=device)
        vel_start = ode_func.diffeq(t0_tensor, x_start)

        integration_times = torch.tensor([t0, t1], dtype=torch.float32, device=device)
        z_new, delta_logp = model(x_input, zero, integration_times=integration_times)
        deltas.append(delta_logp)

        if i == 0:
            z_new_batch = z_new
        else:
            z_new_batch = z_new[-args.batch_size:]

        if i == 0:
            z = z_new
        else:
            z = torch.cat((z, z_new_batch), dim=0)

        z_new_list.append(z_new_batch)

        # ----energy ----
        vel_end = ode_func.diffeq(t1_tensor, z_new_batch)
        energy_step = 0.5 * (torch.sum(vel_start ** 2, dim=1) + torch.sum(vel_end ** 2, dim=1)) * abs(t1 - t0)
        energy_terms.append(FIXED_LAMBDA_ENERGY * torch.mean(energy_step))

        # ---- OT ----
        if ot_loss_fn is not None:
            w_pred = torch.ones(z_new_batch.shape[0], device=device) / z_new_batch.shape[0]
            w_true = torch.ones(x_end.shape[0], device=device) / x_end.shape[0]
            ot_raw = ot_loss_fn(w_pred, z_new_batch, w_true, x_end, requires_grad=True)
            scale = torch.sqrt(displacement / 50.0)
            ot_terms.append(FIXED_LAMBDA_OT * ot_raw * scale)

        # ---- STD ----
        pred_std = torch.std(z_new_batch, dim=0, unbiased=False)
        true_std = torch.std(x_end, dim=0, unbiased=False)
        std_term = torch.mean((pred_std - true_std) ** 2)
        std_terms.append(FIXED_LAMBDA_STD * std_term)

        # ---- reg ----
        if getattr(args, "top_k_reg", 0) > 0:
            anchor = torch.cat([x_start, x_end], dim=0)
            cdist = torch.cdist(z_new_batch, anchor)
            values, _ = torch.topk(cdist, 5, dim=1, largest=False, sorted=False)
            hinge = 0.1
            values = values - hinge
            values[values < 0] = 0
            density_step = torch.mean(values)
            density_terms.append(FIXED_LAMBDA_DENSITY * density_step)

        # ---- disp ----
        pred_vec = z_new_batch.mean(dim=0) - x_start.mean(dim=0)
        real_vec = x_end.mean(dim=0) - x_start.mean(dim=0)

        pred_disp = torch.norm(pred_vec, p=2)
        real_disp = torch.norm(real_vec, p=2)

        ratio = pred_disp / (real_disp + 1e-6)
        loss_ratio = (ratio - 1.0).pow(2)

        cos = F.cosine_similarity(pred_vec, real_vec, dim=0, eps=1e-8)
        loss_dir = 1.0 - cos
        loss_dir = torch.where(real_disp > 1e-3, loss_dir, torch.zeros_like(loss_dir))

        disp_terms.append(FIXED_LAMBDA_DISP * loss_ratio)

        ratio_stats.append(ratio.detach())
        cos_stats.append(cos.detach())
        pred_disp_stats.append(pred_disp.detach())
        real_disp_stats.append(real_disp.detach())

    if len(z_new_list) == 0:
        return (torch.tensor(0.0, device=device),) * 11

    # ---- NLL ----
    final_z = z_new_list[-1]
    logpz = standard_normal_logprob(final_z)
    logpz = logpz.view(logpz.shape[0], -1).sum(dim=1) if logpz.ndim > 1 else logpz
    logp = logpz
    step_losses = []
    for delta_logp in reversed(deltas):
        if delta_logp.shape[0] > args.batch_size:
            delta_logp_cur = delta_logp[-args.batch_size:]
        else:
            delta_logp_cur = delta_logp
        if delta_logp_cur.ndim > 1:
            delta_logp_cur = delta_logp_cur.view(delta_logp_cur.shape[0], -1).sum(dim=1)
        logp = logp - delta_logp_cur
        step_losses.append(-torch.mean(logp))
    nll = torch.mean(torch.stack(step_losses)) if step_losses else torch.tensor(0.0, device=device)

    
    if len(spatial_reg_terms) > 0:
        spatial_reg = torch.mean(torch.stack(spatial_reg_terms))
        nll = nll + spatial_reg
    else:
        spatial_reg = torch.tensor(0.0, device=device)

   
    ot = torch.mean(torch.stack(ot_terms)) if ot_terms else torch.tensor(0.0, device=device)
    energy = torch.mean(torch.stack(energy_terms)) if energy_terms else torch.tensor(0.0, device=device)
    std_loss = torch.mean(torch.stack(std_terms)) if std_terms else torch.tensor(0.0, device=device)
    density = torch.mean(torch.stack(density_terms)) if density_terms else torch.tensor(0.0, device=device)
    velocity_loss = torch.tensor(0.0, device=device)
    disp_loss = torch.mean(torch.stack(disp_terms)) if disp_terms else torch.tensor(0.0, device=device)

    ratio_mean = torch.mean(torch.stack(ratio_stats)) if ratio_stats else torch.tensor(0.0, device=device)
    cos_mean = torch.mean(torch.stack(cos_stats)) if cos_stats else torch.tensor(0.0, device=device)
    pdisp_mean = torch.mean(torch.stack(pred_disp_stats)) if pred_disp_stats else torch.tensor(0.0, device=device)
    tdisp_mean = torch.mean(torch.stack(real_disp_stats)) if real_disp_stats else torch.tensor(0.0, device=device)

    setattr(args, "_last_spatial_reg", float(spatial_reg.item()))

    return nll, ot, energy, std_loss, density, velocity_loss, disp_loss, ratio_mean, cos_mean, pdisp_mean, tdisp_mean


def compute_loss(device, args, data, spatial_coords, graph, label, velocity, model, logger, full_data):
    
    use_ot = getattr(args, "use_ot", True)
    ot_loss_fn = OTLoss(args, device) if use_ot else None

    (nll_b, ot_b, en_b, st_b, den_b, vel_b, disp_b,
     ratio_b, cos_b, pdisp_b, tdisp_b) = _compute_direction_loss(
        device, args, data, spatial_coords, label, model, velocity, "backward", ot_loss_fn
    )
    spatial_reg_b = getattr(args, "_last_spatial_reg", 0.0)

    (nll_f, ot_f, en_f, st_f, den_f, vel_f, disp_f,
     ratio_f, cos_f, pdisp_f, tdisp_f) = _compute_direction_loss(
        device, args, data, spatial_coords, label, model, velocity, "forward", ot_loss_fn
    )
    spatial_reg_f = getattr(args, "_last_spatial_reg", 0.0)

    total_nll = 0.5 * (nll_b + nll_f)
    total_ot = 0.5 * (ot_b + ot_f)
    total_energy = 0.5 * (en_b + en_f)
    total_std = 0.5 * (st_b + st_f)
    total_density = 0.5 * (den_b + den_f)
    total_disp = 0.5 * (disp_b + disp_f)
    total_spatial_reg = 0.5 * (spatial_reg_b + spatial_reg_f)

    total_ratio = 0.5 * (ratio_b + ratio_f)
    total_cos = 0.5 * (cos_b + cos_f)
    total_pdisp = 0.5 * (pdisp_b + pdisp_f)
    total_tdisp = 0.5 * (tdisp_b + tdisp_f)

    total_loss = total_nll + total_ot + total_energy + total_std + total_density + total_disp

    
    if graph is not None:
        l1 = torch.norm(graph, p=1) / (graph.numel() + 1e-8)
        l2 = torch.norm(graph, p=2) / (graph.numel() + 1e-8)
        total_loss += getattr(args, "graph_l1_weight", 0.01) * l1 + getattr(args, "graph_l2_weight", 0.01) * l2

   
    if not hasattr(args, "_loss_print_step"):
        args._loss_print_step = 0
    args._loss_print_step += 1
    if args._loss_print_step % 20 == 0 or args._loss_print_step == 1:
        logger.info(
            "Step {:5d} | NLL={:.4f} OT={:.4f} ENG={:.4f} STD={:.4f} DEN={:.4f} DISP={:.4f} SREG={:.4f} | "
            "PDISP={:.4f} TDISP={:.4f} RATIO={:.4f} COS={:.4f}".format(
                args._loss_print_step,
                total_nll.item(), total_ot.item(), total_energy.item(),
                total_std.item(), total_density.item(), total_disp.item(),
                total_spatial_reg,
                total_pdisp.item(), total_tdisp.item(), total_ratio.item(), total_cos.item()
            )
        )

    return total_loss
