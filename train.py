import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import RunningAverageMeter, count_total_time, count_nfe, save_checkpoint
from loss import compute_loss, OTLoss
from lib.regularization import get_regularization, append_regularization_to_log
from visualize import visualize
from utils import compute_jacobian_30d

def train_eval(device, args, data,spatial_coords, graph, label, velocity, model, itr, best_loss, logger, full_data):
   
    model.eval()
    with torch.no_grad():
        current_loss = compute_loss(device, args, data,spatial_coords, graph, label, velocity, model, logger, full_data)

    test_nfe = count_nfe(model)
    log_message = "[TEST] Iter {:04d} | Test Loss {:.6f} | NFE {:.0f}".format(
        itr, current_loss.item(), test_nfe
    )
    logger.info(log_message)

    with open(os.path.join(args.save, "train_eval.csv"), "a") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow((itr, current_loss.item()))

    if current_loss.item() < best_loss:
        best_loss = current_loss.item()
        chkpt = {"state_dict": model.state_dict()}
        torch.save(chkpt, os.path.join(args.save, "checkpt.pth"))

    return current_loss.item(), best_loss


def pred_train(device, args, data,spatial_coords, graph, label, velocity, model,
               regularization_coeffs, regularization_fns, logger):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    time_meter = RunningAverageMeter(0.93)
    loss_meter = RunningAverageMeter(0.93)
    nfef_meter = RunningAverageMeter(0.93)
    nfeb_meter = RunningAverageMeter(0.93)

    full_data = torch.from_numpy(data).float().to(device)
    best_loss = float("inf")
    end = time.time()

    for itr in range(1, args.niters + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        loss = compute_loss(device, args, data,spatial_coords, graph, label, velocity, model, logger, full_data)

        if len(regularization_coeffs) > 0:
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states, regularization_coeffs)
                if coeff != 0
            )
            loss = loss + reg_loss
        else:
            reg_states = []

        nfe_forward = count_nfe(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_meter.update(loss.item())

        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)

        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) | "
            "NFE Forward {:.0f}({:.1f}) | NFE Backward {:.0f}({:.1f})".format(
                itr,
                time_meter.val, time_meter.avg,
                loss_meter.val, loss_meter.avg,
                nfef_meter.val, nfef_meter.avg,
                nfeb_meter.val, nfeb_meter.avg,
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
        logger.info(log_message)

        if itr % args.viz_freq == 0:
            with torch.no_grad():
                visualize(device, args, data, label, model, itr)

        if itr % args.save_freq == 0:
            chkpt = {"state_dict": model.state_dict()}
            save_checkpoint(chkpt, args.save, epoch=itr)

        end = time.time()

   
    current_val_loss, best_val_loss = train_eval(
        device, args, data,spatial_coords, graph, label, velocity, model,
        args.niters, best_loss, logger, full_data
    )
    logger.info("Training has finished.")
    return model, current_val_loss, best_val_loss


def graph_discov(
    device, args, data,spatial_coords, graph, label, velocity, model,
    optimizer_graph, regularization_coeffs, regularization_fns, logger,
    P_tensor=None
):
    
    print("########################### graph_discov ###############################")

    full_data = torch.from_numpy(data).float().to(device)

    jac_reg_weight = getattr(args, "jac_reg_weight", 0.1)
    jac_reg_freq = getattr(args, "jac_reg_freq", 10)
    jac_batch = getattr(args, "jac_batch", 64)
    logits_clip = getattr(args, "graph_logits_clip", 10.0)

    for itr in range(1, args.graph_niters + 1):
        optimizer_graph.zero_grad(set_to_none=True)
        g0 = graph.detach().clone()

        
        loss = compute_loss(device, args, data, spatial_coords,graph, label, velocity, model, logger, full_data)

        
        if len(regularization_coeffs) > 0:
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states, regularization_coeffs)
                if coeff != 0
            )
            loss = loss + reg_loss
        else:
            reg_states = []

        # =============== Jacobian consistency REG (differentiable) ===============
        loss_consistency = None
        if (P_tensor is not None) and (jac_reg_weight > 0) and (itr % jac_reg_freq == 0):
            try:
                bs = min(jac_batch, data.shape[0])
                idx = np.random.randint(data.shape[0], size=bs)
                z_batch = torch.from_numpy(data[idx]).float().to(device)

                
                if hasattr(args, "int_tps"):
                    t_scalar = float(np.random.choice(args.int_tps))
                else:
                    t_scalar = float(np.random.choice(args.timepoints_train))
                t_batch = torch.tensor(t_scalar, dtype=torch.float32, device=device, requires_grad=True)

                diffeq = model.chain[0].odefunc.odefunc.diffeq

                
                J_latent = compute_jacobian_30d(diffeq, t_batch, z_batch, max_dim=50, create_graph=True)  # (bs,d,d)

                
                J_mean = torch.mean(J_latent * J_latent, dim=0)  # (d,d)
                J_mean = J_mean / (J_mean.max() + 1e-8)

              
                A_latent = torch.sigmoid(graph)
                A_latent = A_latent - torch.diag_embed(torch.diagonal(A_latent))

               
                A_module = P_tensor @ A_latent @ P_tensor.T
                A_module = A_module / (A_module.max() + 1e-8)

                J_module = P_tensor @ J_mean @ P_tensor.T
                J_module = J_module / (J_module.max() + 1e-8)

                loss_consistency = F.mse_loss(A_module, J_module)
                loss = loss + jac_reg_weight * loss_consistency
            except Exception as e:
                logger.info(f"[Jac-Reg] skipped due to error: {e}")
        # =======================================================================

        loss.backward()

        g = graph.grad
        logger.info("graph.requires_grad: {}".format(graph.requires_grad))
        logger.info("graph.grad is None: {}".format(g is None))
        if g is not None:
            logger.info("graph.grad abs mean/max: {:.3e}/{:.3e}".format(
                g.abs().mean().item(), g.abs().max().item()
            ))

      
        torch.nn.utils.clip_grad_norm_([graph], max_norm=1.0)
        optimizer_graph.step()

        
        with torch.no_grad():
            graph.data.clamp_(-logits_clip, logits_clip)

    
        logger.info(f"graph_delta_mean={(graph.detach() - g0).abs().mean().item():.6e}")
        if loss_consistency is not None:
            logger.info(f"[Jac-Reg] itr={itr} loss_consistency={loss_consistency.item():.6f} (w={jac_reg_weight})")

        log_message = f"[graph_discov] Iter {itr:04d} | Loss {loss.item():.6f}"
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(log_message, regularization_fns, reg_states)
        logger.info(log_message)

    return graph
