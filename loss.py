import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
import math
from geomloss import SamplesLoss
from utils import standard_normal_logprob,sample_index
matplotlib.use("Agg")


class OTLoss():

    def __init__(self, args, device):
        self.ot_solver = SamplesLoss("sinkhorn", p=2, blur=args.sinkhorn_blur,
                                     scaling=args.sinkhorn_scaling, debias=True)
        self.device = device

    def __call__(self, a_i, x_i, b_j, y_j, requires_grad=True):
        a_i = a_i.to(self.device)
        x_i = x_i.to(self.device)
        b_j = b_j.to(self.device)
        y_j = y_j.to(self.device)

        if requires_grad:
            a_i.requires_grad_()
            x_i.requires_grad_()
            b_j.requires_grad_()

        loss_xy = self.ot_solver(a_i, x_i, b_j, y_j)
        return loss_xy

def compute_loss(device, args, data, graph, label, velocity, model, logger, full_data):
    """
    Compute loss by integrating backwards from the last time step
    At each time step integrate back one time step, and concatenate that
    to samples of the empirical distribution at that previous timestep
    repeating over and over to calculate the likelihood of samples in
    later timepoints iteratively, making sure that the ODE is evaluated
    at every time step to calculate those later points.

    The growth model is a single model of time independent cell growth /
    death rate defined as a variation from uniform.
    """

    # Backward pass accumulating losses, previous state and deltas
    deltas = []
    zs = []
    losses_xy=[]
    similarity_losses=[]
    z = None
    interp_loss = 0.0
    loss_OT = OTLoss(args, device)
    for i, (itp, tp) in enumerate(zip(args.int_tps[::-1], args.timepoints_train[::-1])):
        # tp counts down from last
        integration_times = torch.tensor([itp - args.time_scale, itp])
        integration_times = integration_times.type(torch.float32).to(device)

        # load data and add noise
        idx,a_i = sample_index(args.batch_size, data, label, tp, w=None)
        x = data[idx]
        if args.training_noise > 0.0:
            x += np.random.randn(*x.shape) * args.training_noise
        x = torch.from_numpy(x).type(torch.float32).to(device)
        if i == 0:
            b_i = a_i

        if i > 0:
            x = torch.cat((z, x))
            zs.append(z)
            b_i = torch.cat((b_i,a_i),dim=0)
        zero = torch.zeros(x.shape[0], 1).to(x)

        # transform to previous timepoint
        z, delta_logp = model(x, zero, integration_times=integration_times)
        b_j = b_i
        loss_xy = loss_OT(b_i, z, b_j, x)
        losses_xy.append(loss_xy.item())
        deltas.append(delta_logp)
    logpz = standard_normal_logprob(z)
    logpz_np = z.detach().cpu().numpy()
    np.save(os.path.join(args.save, f"logpz_tp_{tp}_batch_{i}.npy"), logpz_np)
    logger.info(f"Saved logpz for timepoint {tp}, batch {i}")
        # Accumulate losses
    losses = []
    logps = [logpz]
    for i, delta_logp in enumerate(deltas[::-1]):
        logpx = logps[-1] - delta_logp
        logps.append(logpx[: -args.batch_size])
        losses.append(-torch.mean(logpx[-args.batch_size:]))
    losses = torch.stack(losses)
    weights = torch.ones_like(losses).to(logpx)
    if args.leaveout_timepoint >= 0:
        weights[args.leaveout_timepoint] = 0
    #losses = torch.mean(losses * weights)+losses_xy * ot_weight
    losses = torch.mean(losses * weights)+torch.mean(F.mse_loss(z, x))
    # losses =  np.mean(losses_xy)
    logger.info(np.mean(losses_xy))
        # print(losses_xy)
        # Straightline regularization
        # Integrate to random point at time t and assert close to (1 - t) * end + t * start
    if args.interp_reg:
        t = np.random.rand()
        int_t = torch.tensor([itp - t * args.time_scale, itp])
        int_t = int_t.type(torch.float32).to(device)
        int_x = model(x,integration_times=int_t) 
        int_x = int_x.detach()
        actual_int_x = x * (1 - t) + z * t
        interp_loss += F.mse_loss(int_x, actual_int_x)
    # losses = np.mean(losses_xy)+losses
    if args.interp_reg:
        print("interp_loss", interp_loss)
    
    # Direction regularization
    if args.vecint:
        direction_list = []
        similarity_loss = 0
        for i, (itp, tp) in enumerate(zip(args.int_tps, args.timepoints_train)):
            itp = torch.tensor(itp).type(torch.float32).to(device)
            idx, w_ = sample_index(args.batch_size, data, label, tp, w=None)
            x = data[idx]
            v = velocity[idx]
            x = torch.from_numpy(x).type(torch.float32).to(device)
            v = torch.from_numpy(v).type(torch.float32).to(device)
            x += torch.randn_like(x) * 0.1
            # Only penalizes at the time / place of visible samples
            direction = model.chain[0].odefunc.odefunc.diffeq(itp, x)
            # if args.use_magnitude:
            #     similarity_loss += torch.mean(F.mse_loss(direction, v))
            # else:
            #     similarity_loss -= torch.mean(F.cosine_similarity(direction, v))
            # similarity_loss -= torch.mean(F.cosine_similarity(direction, v))
            # similarity_loss = torch.mean(F.cosine_similarity(direction, v))
            direction_np = direction.detach().cpu().numpy()
            direction_list.append(direction_np)
            similarity_loss_itp = torch.mean(F.mse_loss(direction, v))
            similarity_loss += similarity_loss_itp
            logger.info(similarity_loss_itp)
        direction_array = np.array(direction_list)
        np.save(os.path.join(args.save, "predicted_directions.npy"), direction_array)
        logger.info(f"预测方向数据已保存至 {args.save}/predicted_directions.npy")
        logger.info(similarity_loss)
            # similarity_losses.append(similarity_loss.item())

        # similarity_loss = similarity_loss/(args.timepoints[::-1][0]+1)
        logger.info(similarity_loss) #to 1 better
        loss_sparsity = torch.norm(graph, p=1) / (graph.shape[0] * graph.shape[1])
        # losses += similarity_loss * args.vecint + loss_sparsity * 0.3
        losses += similarity_loss
        # losses +=  loss_sparsity * 0.3

    # Density regularization
    if args.top_k_reg > 0:
        density_loss = 0
        tp_z_map = dict(zip(args.timepoints[:-1], zs[::-1]))
        predicted_expression_list = []
        if args.leaveout_timepoint not in tp_z_map:
            idx = args.data.sample_index(args.batch_size, tp)
            x = args.data.get_data()[idx]
            if args.training_noise > 0.0:
                x += np.random.randn(*x.shape) * args.training_noise
            x = torch.from_numpy(x).type(torch.float32).to(device)
            t = np.random.rand()
            int_t = torch.tensor([itp - t * args.time_scale, itp])
            int_t = int_t.type(torch.float32).to(device)
            int_x = model(x, integration_times=int_t)
            samples_05 = int_x


        else:

            # If we are leaving out a timepoint the regularize there
            samples_05 = tp_z_map[args.leaveout_timepoint]


        # Calculate distance to 5 closest neighbors
        # WARNING: This currently fails in the backward pass with cuda on pytorch < 1.4.0
        #          works on CPU. Fixed in pytorch 1.5.0
        # RuntimeError: CUDA error: invalid configuration argument
        # The workaround is to run on cpu on pytorch <= 1.4.0 or upgrade
        cdist = torch.cdist(samples_05, full_data)
        values, _ = torch.topk(cdist, 5, dim=1, largest=False, sorted=False)
        # Hinge loss
        hinge_value = 0.1
        values -= hinge_value
        values[values < 0] = 0
        density_loss = torch.mean(values)
        print("Density Loss", density_loss.item())
        losses += density_loss * args.top_k_reg
    losses += interp_loss
    return losses

# def loss(device, args, x):
#     losses_xy = []
#     losses_r = []
#     train_epoch = args.niters
#
#     dat_prev = x[args.start_t]
#     x_i, a_i = p_samp(dat_prev, int(dat_prev.shape[0] * args.train_batch))
#     r_i = torch.zeros(int(dat_prev.shape[0] * args.train_batch)).unsqueeze(1)
#     x_r_i = torch.cat([x_i, r_i], dim=1)
#     x_r_i = x_r_i.to(device)
#     ts = [0] + args.train_t
#     y_ts = [np.float64(y[ts_i]) for ts_i in ts]
#     x_r_s = model(y_ts, x_r_i)
#
#     for j in config.train_t:
#         t_cur = j
#
#         dat_cur = x[t_cur]
#         y_j, b_j = p_samp(dat_cur, int(dat_cur.shape[0] * args.train_batch))
#
#         position = config.train_t.index(j)
#         loss_xy = loss(a_i, x_r_s[position + 1][:, 0:-1], b_j, y_j)
#
#         losses_xy.append(loss_xy.item())
#
#         if (config.train_lambda > 0) & (j == config.train_t[-1]):
#             loss_r = torch.mean(x_r_s[-1][:, -1] * config.train_lambda)
#             losses_r.append(loss_r.item())
#             loss_all = loss_xy + loss_r
#         else:
#             loss_all = loss_xy
#
#         loss_all.backward(retain_graph=True)
#
#     train_loss_xy = np.mean(losses_xy)
#     train_loss_r = np.mean(losses_r)
#
#
#
