import os
import scipy.sparse as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import scvelo as scv
from sklearn.decomposition import PCA
#from data_simulate import load_data_lorenz_96, load_data_simulation, plot_integrate
from utils import seed_everything, load_data, sigmoid_gumbel_sample,get_logger,makedirs,count_parameters
from parse import parser
from model import build_model_tabular,set_cnf_options
from lib.regularization import create_regularization_fns
from train import pred_train,graph_discov
from visualize import plot_output, plot_causal_matrix,visualize
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import os

# 强制移除导致冲突的显存配置变量
if "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
    del os.environ["PYTORCH_CUDA_ALLOC_CONF"]

import torch
# 下面是你原来的代码...
def integrate_backwards(end_samples, model, savedir, ntimes=100, memory=0.1, device="cpu"):
    """Integrate some samples backwards and save the results."""
    with torch.no_grad():
        z = torch.from_numpy(end_samples).type(torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1).to(z)
        cnf = model.chain[0]

        zs = [z]
        deltas = []
        int_tps = np.linspace(args.int_tps[1], args.int_tps[2], ntimes)
        # for i, itp in enumerate(int_tps[::-1][:-1]):
        for i, itp in enumerate(int_tps[:-1]):
            # tp counts down from last
            timescale = int_tps[1] - int_tps[0]
            integration_times = torch.tensor([itp - timescale, itp])
            # integration_times = torch.tensor([np.linspace(itp - timescale, itp, ntimes)])
            integration_times = integration_times.type(torch.float32).to(device)

            # transform to previous timepoint
            z, delta_logp = cnf(zs[-1], zero, integration_times=integration_times)
            zs.append(z)
            deltas.append(delta_logp)
        zs = torch.stack(zs, 0)
        z_pred = zs[-1]
        zs = zs.cpu().numpy()
        np.save(os.path.join(savedir, "backward_trajectories.npy"), zs)
        return z_pred

def main(args,data_floder):
    seed_everything(1)
    makedirs(args.save)
    logger = get_logger(
        logpath=os.path.join(args.save, "logs"),filepath=os.path.abspath(__file__),displaying=~args.no_display_loss,)
    logger.info(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adata, data, labels, velocity = load_data(args,datapath)
    all_timepoints = np.sort(np.unique(adata.obs['time']))
    print("所有时间点:", all_timepoints)

# 确保至少有8个时间点
    

# 分割时间点：前4个为测试，后4个为训练
    #test_timepoints = all_timepoints[:5]  # 前4个时间点
    train_timepoints = all_timepoints[0:4]  # 后4个时间点

    
    print("训练时间点:", train_timepoints)

# 创建训练/测试索引
    train_indices = np.isin(labels, train_timepoints)
    #test_indices = np.isin(labels, test_timepoints)

# 分割数据
    data_train = data[train_indices]
    labels_train = labels[train_indices]
   
    args.timepoints = all_timepoints
    args.timepoints_train = train_timepoints
 
    max_time = max(all_timepoints)
  
    # as some timepoints may be left out for validation etc.
    #args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale
    #data, labels,GC = load_data_simulation(args, data_floder)
    #data=adata.X
    #graph_causal_fit = pd.read_csv(graphpath)
    #graph_causal_fit=graph_causal_fit.values
    print(adata)
    #args.timepoints = np.unique(labels)
    #args.timepoints = np.unique(adata.obs['time_point'])
    #args.timepoints = args.timepoints.astype(int)
    args.max=max(args.timepoints)
    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.
    args.time_scale = float(args.time_scale)
    args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale
    graph_causal_fit = np.corrcoef(data.T)
    #graph_causal_fit=GC
    graph_causal_fit = nn.Parameter(torch.tensor(graph_causal_fit))
    #idx = np.arange(data.shape[0])[labels == np.unique(labels)[-1]]
    #data_train = np.delete(data, idx, axis=0)
   
    #labels_train = np.delete(labels, idx, axis=0)
    regularization_fns, regularization_coeffs = create_regularization_fns(args)

    graph_causal_fit = nn.Parameter(torch.tensor(graph_causal_fit))

    for itr in range(1, args.total_niters + 1):
        graph_causal_fit =sigmoid_gumbel_sample(graph_causal_fit, tau=1) 
        graph_causal_fit =graph_causal_fit.detach()
        graph_causal_fit.requires_grad = True
        model = build_model_tabular(args, data.shape[1], graph_causal_fit, regularization_fns, ).to(device)
    # growth_model = None
        set_cnf_options(args, model)
        logger.info(model)
        n_param = count_parameters(model)
        logger.info("Number of trainable parameters: {}".format(n_param))
    
        pred_train(device, args, data_train, graph_causal_fit, labels_train, velocity, model, regularization_coeffs, regularization_fns, logger)
        #visualize(device, args, data, labels, model, itr)
        # graph_discov(device, args, data, graph_causal_fit, model, regularization_coeffs, regularization_fns, logger,)
        causal_fit = graph_discov(device, args, data, graph_causal_fit, labels,  velocity, graph_causal_fit,model, regularization_coeffs,regularization_fns, logger)
        #device, args, data, graph, label, velocity, graph_causal_fit, model, regularization_coeffs, regularization_fns, logger
        torch.save(model, "/home/lenovo/jora/data/Rgraph_checkpt_pca.pth",)
        torch.save(causal_fit , "/home/lenovo/jora/data/Rcausal_fit_pca.pth", )
        #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')S
        #idx = np.arange(data.shape[0])[labels == np.unique(labels)[-1]]
        #data_train = np.delete(data, idx, axis=0)
        #labels_train = np.delete(labels, idx, axis=0)
        #regularization_fns, regularization_coeffs = create_regularization_fns(args)
        #end_time_data = data_train[labels_train == np.max(labels_train)]
        #z_pred=integrate_backwards(end_time_data, model, args.save, ntimes=10, device=device)
        #if data.shape[1] == 500:
        #data_original = data.copy()  # 保留原始数据供模型使用
        #data_viz = PCA(n_components=2).fit_transform(data_original)  # (n_samples, 2)
        #plot_output(
            #device,
            #args,
            #data,  # 模型仍使用原始数据
            #labels,
            #model
                        #)

        return data, labels, causal_fit

def eval(args, data, labels, graph_causal_fit):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx = np.arange(data.shape[0])[labels == np.unique(labels)[-1]]
    data_train = np.delete(data, idx, axis=0)
    labels_train = np.delete(labels, idx, axis=0)
    regularization_fns, regularization_coeffs = create_regularization_fns(args)
    model = build_model_tabular(args, data.shape[1], graph_causal_fit, regularization_fns, ).to(device)

    # Use maximum timepoint to establish integration_times
    # as some timepoints may be left out for validation etc.

    # state_dict = torch.load("/media/lenovo/6ED3FFE79A41910F/Lu/causal_1105/results/tmp/embry/dim10/graph_checkpt.pth", map_location=device)
    # model.load_state_dict(state_dict["state_dict"])
    #model = torch.load("/media/lenovo/A06B2FA1620B6FCB/causal_1105/results/tmp/embry4/graph_checkpt.pth")
    # direction = model.chain[0].odefunc.odefunc.diffeq(itp, data)

    set_cnf_options(args, model)
    args.data = data
    # args.timepoints = args.data.get_unique_times()
    # args.int_tps = (np.arange(max(args.timepoints) + 1) + 1.0) * args.time_scale

    # end_time_data = data.data_dict[args.embedding_name]
    # end_time_data = data.get_data()[
    #     args.data.get_times() == np.max(args.data.get_times())
    # ]
    # np.random.permutation(end_time_data)
    # rand_idx = np.random.randint(end_time_data.shape[0], size=5000)
    # end_time_data = end_time_data[rand_idx,:]
    end_time_data = data_train[labels_train == np.max(labels_train)]
    integrate_backwards(end_time_data, model, args.save, ntimes=100, device=device)
    #print("integrating backwards")
    
    
    # end_time_data = data[labels == np.min(labels)]
    # end_time_data = data_train[labels_train == np.max(labels_train)]
    #z_pred = integrate_backwards(end_time_data, model, args.save, ntimes=10, device=device)
    #np.save(os.path.join(args.save, "predicted_data.npy"), z_pred.cpu().numpy())
    # data[idx] = z_pred.cpu()
    # savedir = args.save
    #return z_pred

if __name__ == "__main__":
    args = parser.parse_args()
    # main(args,datapath='/media/lenovo/6ED3FFE79A41910F/Lu/causal_1105/results/adata.h5ad')
    datapath='/home/lenovo/jora/data/R5_filtered_latent.h5ad'
    data, labels, causal_fit= main(args,datapath)
    #np.save(os.path.join(args.save, "predicted_data.npy"), z_pred.cpu().numpy())
    # scv.pl.velocity_embedding_stream(adata, basis="umap", color="celltype")
    # direction = eval(args, data, labels, causal_fit)
    eval(args, data, labels, causal_fit)
    plot_causal_matrix(causal_fit.cpu().detach().numpy(), figsize=[4, 3], vmin=0, vmax=1)

    # plt.savefig(pred_cg, os.path.join(args.save, "pred_cm.jpg"), dpi=300)