import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

from utils import RunningAverageMeter,count_total_time, count_nfe,save_checkpoint
from loss import compute_loss,OTLoss
from lib.regularization import  get_regularization,append_regularization_to_log
from visualize import visualize

def train_eval(device, args, data, graph, label, velocity, model,  itr, best_loss, logger, full_data):
    model.eval()
    test_loss = compute_loss(device, args, data, graph, label, velocity, model, logger, full_data)

    test_nfe = count_nfe(model)
    log_message = "[TEST] Iter {:04d} | Test Loss {:.6f} |" " NFE {:.0f}".format(
        itr, test_loss, test_nfe
    )
    logger.info(log_message)
    # utils.makedirs(args.save)
    with open(os.path.join(args.save, "train_eval.csv"), "a") as f:
        import csv

        writer = csv.writer(f)
        writer.writerow((itr, test_loss))

    if test_loss.item() < best_loss:
        best_loss = test_loss.item()
        chkpt = {
            "state_dict": model.state_dict(),
        }
        torch.save(
            chkpt,
            os.path.join(args.save, "checkpt.pth"),
        )

def pred_train(device, args, data, graph, label, velocity, model, regularization_coeffs, regularization_fns, logger):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    time_meter = RunningAverageMeter(0.93)
    loss_meter = RunningAverageMeter(0.93)
    nfef_meter = RunningAverageMeter(0.93)
    nfeb_meter = RunningAverageMeter(0.93)
    tt_meter = RunningAverageMeter(0.93)

    full_data = (
        torch.from_numpy(data).type(torch.float32).to(device))
    best_loss = float("inf")
    end = time.time()
    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        model.train()

        loss = compute_loss(device, args, data, graph, label, velocity, model,  logger, full_data)
        loss_meter.update(loss.item())

        if len(regularization_coeffs) > 0:
            # Only regularize on the last timepoint
            reg_states = get_regularization(model, regularization_coeffs)
            reg_loss = sum(
                reg_state * coeff
                for reg_state, coeff in zip(reg_states, regularization_coeffs)
                if coeff != 0
            )
            loss = loss + reg_loss

        total_time = count_total_time(model)
        nfe_forward = count_nfe(model)
        loss.backward()
        optimizer.step()

        # Eval
        nfe_total = count_nfe(model)
        nfe_backward = nfe_total - nfe_forward
        nfef_meter.update(nfe_forward)
        nfeb_meter.update(nfe_backward)
        time_meter.update(time.time() - end)
        tt_meter.update(total_time)

        log_message = (
            "Iter {:04d} | Time {:.4f}({:.4f}) | Loss {:.6f}({:.6f}) |"
            " NFE Forward {:.0f}({:.1f})"
            " | NFE Backward {:.0f}({:.1f})".format(
                itr,
                time_meter.val,
                time_meter.avg,
                loss_meter.val,
                loss_meter.avg,
                nfef_meter.val,
                nfef_meter.avg,
                nfeb_meter.val,
                nfeb_meter.avg,
            )
        )
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns, reg_states
            )
        logger.info(log_message)

        if itr == args.niters:
            with torch.no_grad():
                train_eval(device, args, data, graph, label, velocity, model, itr, best_loss, logger, full_data)
        if itr % args.viz_freq == 0:

                with torch.no_grad():
                    visualize(device, args, data, label, model, itr)
        if itr % args.save_freq == 0:
            chkpt = {
                "state_dict": model.state_dict(),

            }
            save_checkpoint(
                chkpt,
                args.save,
                epoch=itr,
            )
        end = time.time()
        # visualize(device, args, model, itr)
    logger.info("Training has finished.")


def graph_discov(device, args, data, graph, label, velocity, graph_causal_fit, model, regularization_coeffs, regularization_fns, logger):
    print("###########################graph_causal_fit###############################")
    gamma = 0.97
    graph_optimizer = torch.optim.Adam([graph_causal_fit], lr=0.00001)
    graph_scheduler = torch.optim.lr_scheduler.StepLR(graph_optimizer, step_size=1, gamma=gamma)
    # test_loss = compute_loss(device, args, model,  logger)
    # loss, loss_sparsity, loss_data = graph_discov(x, y, mask)
    full_data = (
        torch.from_numpy(data).type(torch.float32).to(device))
    gumbel_tau_gamma = 0.97

    for itr in range(1, args.graph_niters + 1):
        # graph_causal_fit = sigmoid_gumbel_sample(graph_causal_fit, tau=1)

        gumbel_tau = 1.0
        model.train()
        # loss = compute_loss(device, args, model, logger, full_data)
        loss = compute_loss(device, args, data, graph, label, velocity, model, logger, full_data)
        # graph_optimizer.zero_grad()
        loss.backward()
        graph_optimizer.step()
        # current_graph_disconv_lr = graph_optimizer.param_groups[0]['lr']
        gumbel_tau *= gumbel_tau_gamma
        log_message = (
            "Iter {:04d} | Loss {:.6f}|".format(
                itr,
                loss,
            )
        )
        reg_states = get_regularization(model, regularization_coeffs)
        if len(regularization_coeffs) > 0:
            log_message = append_regularization_to_log(
                log_message, regularization_fns, reg_states
            )
        logger.info(log_message)
    graph_scheduler.step()
    # chkpt = {"state_dict": model.state_dict(),}
    # torch.save(chkpt, os.path.join(args.save, "graph_checkpt.pth"),)
    graph_causal_fit = nn.Parameter(graph_causal_fit)
    print("iter", iter, ":", graph_causal_fit)
    return graph_causal_fit


