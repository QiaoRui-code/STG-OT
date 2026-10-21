import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import itertools

# from Trajectory_GraphODE.utils import base_density
from utils import sample_index, base_density, base_sample,known_base_density

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

LOW = -4
HIGH = 4


def get_transforms(device, args, model, integration_times):
    """
    Given a list of integration points,
    returns a function giving integration times
    """

    def sample_fn(z, logpz=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times
        ]
        if logpz is not None:
            # TODO this works right?
            for it in int_list:
                z, logpz = model(z, logpz, integration_times=it, reverse=True)
            return z, logpz
        else:
            for it in int_list:
                z = model(z, integration_times=it, reverse=True)
            return z

    def density_fn(x, logpx=None):
        int_list = [
            torch.tensor([it - args.time_scale, it]).type(torch.float32).to(device)
            for it in integration_times[::-1]
        ]
        if logpx is not None:
            for it in int_list:
                x, logpx = model(x, logpx, integration_times=it, reverse=False)
            return x, logpx
        else:
            for it in int_list:
                x = model(x, integration_times=it, reverse=False)
            return x

    return sample_fn, density_fn

def plt_flow(prior_logdensity, transform, ax, npts=100, title="$q(x)$", device="cpu"):
    """
    Args:
        transform: computes z_k and log(q_k) given z_0
    """
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.tensor(z, requires_grad=True).type(torch.float32).to(device)
    logqz = prior_logdensity(z)
    z, logqz = transform(z, logqz)
    logqz = torch.sum(logqz, dim=1)[:, None]

    xx = z[:, 0].cpu().numpy().reshape(npts, npts)
    yy = z[:, 1].cpu().numpy().reshape(npts, npts)
    qz = np.exp(logqz.cpu().numpy()).reshape(npts, npts)

    plt.pcolormesh(xx, yy, qz)
    ax.set_xlim(LOW, HIGH)
    ax.set_ylim(LOW, HIGH)
    cmap = matplotlib.cm.get_cmap(None)
    ax.set_facecolor(cmap(0.))
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_density(prior_logdensity, inverse_transform, ax, npts=100, memory=100, title="$q(x)$", device="cpu"):
    side = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(side, side)
    x = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    x = torch.from_numpy(x).type(torch.float32).to(device)
    zeros = torch.zeros(x.shape[0], 1).to(x)

    z, delta_logp = [], []
    inds = torch.arange(0, x.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        z_, delta_logp_ = inverse_transform(x[ii], zeros[ii])
        z.append(z_)
        delta_logp.append(delta_logp_)
    z = torch.cat(z, 0)
    delta_logp = torch.cat(delta_logp, 0)

    logpz = prior_logdensity(z)
    logpx = logpz - delta_logp

    px = np.exp(logpx.cpu().numpy()).reshape(npts, npts)

    ax.imshow(px)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_flow_samples(prior_sample, transform, ax, npts=100, memory=100, title="$x ~ q(x)$", device="cpu"):
    z = prior_sample(npts * npts, 2).type(torch.float32).to(device)
    zk = []
    inds = torch.arange(0, z.shape[0]).to(torch.int64)
    for ii in torch.split(inds, int(memory**2)):
        zk.append(transform(z[ii]))
    zk = torch.cat(zk, 0).cpu().numpy()
    ax.hist2d(zk[:, 0], zk[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def plt_potential_func(potential, ax, npts=100, title="$p(x)$"):
    """
    Args:
        potential: computes U(z_k) given z_k
    """
    xside = np.linspace(LOW, HIGH, npts)
    yside = np.linspace(LOW, HIGH, npts)
    xx, yy = np.meshgrid(xside, yside)
    z = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])

    z = torch.Tensor(z)
    u = potential(z).cpu().numpy()
    p = np.exp(-u).reshape(npts, npts)

    plt.pcolormesh(xx, yy, p)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)


def plt_samples(samples, ax, npts=100, title="$x ~ p(x)$"):
    ax.hist2d(samples[:, 0], samples[:, 1], range=[[LOW, HIGH], [LOW, HIGH]], bins=npts)
    ax.invert_yaxis()
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title)

def visualize_transform(potential_or_samples, prior_sample, prior_density, transform=None, inverse_transform=None, samples=True, npts=100,
    memory=100, device="cpu"):
    """Produces visualization for the model density and samples from the model."""
    plt.clf()
    ax = plt.subplot(1, 3, 1, aspect="equal")
    if samples:
        plt_samples(potential_or_samples, ax, npts=npts)
    else:
        plt_potential_func(potential_or_samples, ax, npts=npts)

    ax = plt.subplot(1, 3, 2, aspect="equal")
    if inverse_transform is None:
        plt_flow(prior_density, transform, ax, npts=npts, device=device)
    else:
        plt_flow_density(prior_density, inverse_transform, ax, npts=npts, memory=memory, device=device)

    ax = plt.subplot(1, 3, 3, aspect="equal")
    if transform is not None:
        plt_flow_samples(prior_sample, transform, ax, npts=npts, memory=memory, device=device)


def visualize(device, args, data, label, model, itr):
    model.eval()
    # for i, tp in enumerate(args.timepoints_train):
    for i, tp in enumerate(args.timepoints):
        idx, w_ = sample_index(args.batch_size, data, label, tp, w=None)
        # idx = args.data.sample_index(args.viz_batch_size, tp)
        p_samples = data[idx]
        sample_fn, density_fn = get_transforms(
            device, args, model, args.int_tps[: i + 1]
        )
        plt.figure(figsize=(9, 3))
        visualize_transform(
            p_samples,
            base_sample(),
            base_density(),
            transform=sample_fn,
            inverse_transform=density_fn,
            samples=True,
            npts=100,
            device=device,
        )
        fig_filename = os.path.join(
            args.save, "figs", "{:04d}_{:01d}.jpg".format(itr, i)
        )
        makedirs(os.path.dirname(fig_filename))
        plt.savefig(fig_filename)
        plt.close()

def save_vectors(prior_logdensity, model, data_samples, full_data, labels, savedir, skip_first=False, ntimes=101, end_times=None, memory=0.01, device='cpu', lim=4):
    model.eval()

    #  Sample from prior
    z_samples = data_samples.to(device)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = prior_logdensity(z_samples)
        t = 0
        for cnf in model.chain:
            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            # integration_list = []
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]

            # Start integration at first end_time
            for i, et in enumerate(end_times[1:]):
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            # if len(end_times) == 1:
            #     integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            # print(integration_list)


            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(z_samples, logp_samples, integration_times=integration_list[0], reverse=True)
            full_traj = [(z_traj, logpz)]
            for int_times in integration_list[1:]:
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True)
                full_traj.append((z_traj, logpz))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            z_traj = z_traj.cpu().numpy()

            # mask out stray negative points
            pos_mask = full_data[:,1] >=0
            full_data = full_data[pos_mask]
            labels = labels[pos_mask]
            print(np.unique(labels))

            plt.figure(figsize=(8, 8))
            ax = plt.subplot(aspect="equal")
            ax.scatter(full_data[:,0], full_data[:,1], c=labels.astype(np.int32), cmap='tab10', s=0.5, alpha=1)
            # If we do not have a known base density then skip vectors for the first integration.

            z_traj = np.swapaxes(z_traj, 0, 1)
            if skip_first:
                z_traj = z_traj[:, ntimes:, :]
            ax.scatter(z_traj[:,0,0], z_traj[:,0,1], s=20, c='k')
            for zk in z_traj:
            #for zk in z_traj[:,ntimes:,:]:
                ax.scatter(zk[:,0], zk[:,1], s=1, c = np.linspace(0,1,zk.shape[0]), cmap='Spectral')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            #ax.set_ylim(4, -4)
            makedirs(savedir)
            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(savedir, f"vectors.jpg"), dpi=300)
            t += 1


def save_trajectory(prior_logdensity, prior_sampler, model, data_samples, savedir, ntimes=101, end_times=None,
                    memory=0.01, device='cpu'):
    model.eval()

    #  Sample from prior
    z_samples = prior_sampler(2000, 500).to(device)

    # sample from a grid
    npts = 800
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_samples = prior_logdensity(z_samples)
        logp_grid = prior_logdensity(z_grid)
        t = 0
        for cnf in model.chain:

            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            for i, et in enumerate(end_times[1:]):
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            full_times = torch.cat(integration_list, 0)
            print(full_times.shape)

            # Integrate over evenly spaced samples
            z_traj, logpz = cnf(z_samples, logp_samples, integration_times=integration_list[0], reverse=True)
            full_traj = [(z_traj, logpz)]
            for int_times in integration_list[1:]:
                prev_z, prev_logp = full_traj[-1]
                z_traj, logpz = cnf(prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True)
                full_traj.append((z_traj, logpz))
            full_zip = list(zip(*full_traj))
            z_traj = torch.cat(full_zip[0], 0)
            # z_logp = torch.cat(full_zip[1], 0)
            z_traj = z_traj.cpu().numpy()

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii], logp_grid[ii], integration_times=integration_list[0], reverse=True
                )
                full_traj = [(_grid_z_traj, _grid_logpz_traj)]
                for int_times in integration_list[1:]:
                    prev_z, prev_logp = full_traj[-1]
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True
                    )
                    full_traj.append((_grid_z_traj, _grid_logpz_traj))
                full_zip = list(zip(*full_traj))
                _grid_z_traj = torch.cat(full_zip[0], 0).cpu().numpy()
                _grid_logpz_traj = torch.cat(full_zip[1], 0).cpu().numpy()
                print(_grid_z_traj.shape)
                grid_z_traj.append(_grid_z_traj)
                grid_logpz_traj.append(_grid_logpz_traj)

            grid_z_traj = np.concatenate(grid_z_traj, axis=1)
            grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)

            plt.figure(figsize=(8, 8))
            for _ in range(z_traj.shape[0]):
                plt.clf()

                # plot target potential function
                ax = plt.subplot(2, 2, 1, aspect="equal")

                ax.hist2d(data_samples[:, 0], data_samples[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Target", fontsize=32)

                # plot the density
                ax = plt.subplot(2, 2, 2, aspect="equal")

                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                plt.pcolormesh(xx, yy, qz)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(cmap(0.))
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Density", fontsize=32)

                # plot the samples
                ax = plt.subplot(2, 2, 3, aspect="equal")

                zk = z_traj[t]
                ax.hist2d(zk[:, 0], zk[:, 1], range=[[-4, 4], [-4, 4]], bins=200)
                ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Samples", fontsize=32)

                # plot vector field
                ax = plt.subplot(2, 2, 4, aspect="equal")

                K = 13j
                y, x = np.mgrid[-4:4:K, -4:4:K]
                K = int(K.imag)
                zs = torch.from_numpy(np.stack([x, y], -1).reshape(K * K, 2)).to(device, torch.float32)
                logps = torch.zeros(zs.shape[0], 1).to(device, torch.float32)
                dydt = cnf.odefunc(full_times[t], (zs, logps))[0]
                dydt = -dydt.cpu().detach().numpy()
                dydt = dydt.reshape(K, K, 2)

                logmag = 2 * np.log(np.hypot(dydt[:, :, 0], dydt[:, :, 1]))
                ax.quiver(
                    x, y, dydt[:, :, 0], -dydt[:, :, 1],
                    # x, y, dydt[:, :, 0], dydt[:, :, 1],
                    np.exp(logmag), cmap="coolwarm", scale=20., width=0.015, pivot="mid"
                )
                ax.set_xlim(-4, 4)
                ax.set_ylim(4, -4)
                # ax.set_ylim(-4, 4)
                ax.axis("off")
                ax.set_title("Vector Field", fontsize=32)

                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
                t += 1


def save_trajectory_density(prior_logdensity, model, data_samples, savedir, ntimes=101, end_times=None, memory=0.01,
                            device='cpu'):
    model.eval()

    # sample from a grid
    # Jnpts = 100
    npts = 800
    side = np.linspace(-4, 4, npts)
    xx, yy = np.meshgrid(side, side)
    xx = torch.from_numpy(xx).type(torch.float32).to(device)
    yy = torch.from_numpy(yy).type(torch.float32).to(device)
    z_grid = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], 1)

    with torch.no_grad():
        # We expect the model is a chain of CNF layers wrapped in a SequentialFlow container.
        logp_grid = prior_logdensity(z_grid)
        t = 0
        for cnf in model.chain:
            # Construct integration_list
            if end_times is None:
                end_times = [(cnf.sqrt_end_time * cnf.sqrt_end_time)]
            integration_list = [torch.linspace(0, end_times[0], ntimes).to(device)]
            for i, et in enumerate(end_times[1:]):
                integration_list.append(torch.linspace(end_times[i], et, ntimes).to(device))
            full_times = torch.cat(integration_list, 0)

            grid_z_traj, grid_logpz_traj = [], []
            inds = torch.arange(0, z_grid.shape[0]).to(torch.int64)
            for ii in torch.split(inds, int(z_grid.shape[0] * memory)):
                _grid_z_traj, _grid_logpz_traj = cnf(
                    z_grid[ii], logp_grid[ii], integration_times=integration_list[0], reverse=True
                )
                full_traj = [(_grid_z_traj, _grid_logpz_traj)]
                for int_times in integration_list[1:]:
                    prev_z, prev_logp = full_traj[-1]
                    _grid_z_traj, _grid_logpz_traj = cnf(
                        prev_z[-1], prev_logp[-1], integration_times=int_times, reverse=True
                    )
                    full_traj.append((_grid_z_traj, _grid_logpz_traj))
                full_zip = list(zip(*full_traj))
                _grid_z_traj = torch.cat(full_zip[0], 0).cpu().numpy()
                _grid_logpz_traj = torch.cat(full_zip[1], 0).cpu().numpy()
                print(_grid_z_traj.shape)
                grid_z_traj.append(_grid_z_traj)
                grid_logpz_traj.append(_grid_logpz_traj)

            grid_z_traj = np.concatenate(grid_z_traj, axis=1)[ntimes:]
            grid_logpz_traj = np.concatenate(grid_logpz_traj, axis=1)[ntimes:]

            # plt.figure(figsize=(8, 8))
            # fig, axes = plt.subplots(2,1, gridspec_kw={'height_ratios': [7, 1]}, figsize=(5,7))
            for _ in range(grid_z_traj.shape[0]):
                fig, axes = plt.subplots(2, 1, gridspec_kw={'height_ratios': [7, 1]}, figsize=(8, 8))
                # plt.clf()
                ax = axes[0]
                # Density
                z, logqz = grid_z_traj[t], grid_logpz_traj[t]

                xx = z[:, 0].reshape(npts, npts)
                yy = z[:, 1].reshape(npts, npts)
                qz = np.exp(logqz).reshape(npts, npts)

                ax.pcolormesh(xx, yy, qz)
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)
                cmap = matplotlib.cm.get_cmap(None)
                ax.set_facecolor(cmap(0.))
                # ax.invert_yaxis()
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.set_title("Density", fontsize=32)

                ax = axes[1]

                # Colorbar
                cb = matplotlib.colorbar.ColorbarBase(ax,
                                                      # cmap='Spectral',
                                                      cmap=plt.cm.Spectral,
                                                      orientation='horizontal')
                # cb.set_ticks(np.linspace(0,1,4))
                # cb.set_ticklabels(['48HR', 'Day 12', 'Day 18', 'Day 30'])
                # cb.set_ticklabels(['E12.5', 'E14.5', 'E16.0', 'E17.5'])
                # cb.set_ticks(np.linspace(0,1,5))
                # cb.set_ticklabels(np.arange(len(end_times)))
                ax.axvline(t / grid_z_traj.shape[0], c='k', linewidth=15)
                ax.set_title('Time')

                print('making dir: %s' % savedir)
                makedirs(savedir)
                plt.savefig(os.path.join(savedir, f"viz-{t:05d}.jpg"))
                plt.close()
                t += 1


def trajectory_to_video(savedir):
    import subprocess
    bashCommand = 'ffmpeg -y -i {} {}'.format(os.path.join(savedir, 'viz-%05d.jpg'), os.path.join(savedir, 'traj.mp4'))
    process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()

def plot_output(device, args, data, labels, model):
    save_traj_dir = os.path.join(args.save, "trajectory")
    # logger.info('Plotting trajectory to {}'.format(save_traj_dir))
    # idx, w_ = sample_index(2000, data, labels, 0, w=None)
    idx, w_ = sample_index(2000, data, labels, 1, w=None)
    data_samples = data[idx]
    np.random.seed(42)
    start_points = base_sample()(1000,500)
    # idx = args.data.sample_index(50, 0)
    # start_points = args.data.get_data()[idx]
    # start_points = torch.from_numpy(start_points).type(torch.float32)
    save_vectors(
        base_density(),
        model,
        start_points,
        data,
        labels,
        args.save,
        skip_first=(not known_base_density()),
        device=device,
        end_times=args.int_tps,
        ntimes=100,
    )
    save_trajectory(
        base_density(),
        base_sample(),
        model,
        data_samples,
        save_traj_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
    )
    trajectory_to_video(save_traj_dir)

    density_dir = os.path.join(args.save, "density2")
    save_trajectory_density(
        base_density(),
        model,
        data_samples,
        density_dir,
        device=device,
        end_times=args.int_tps,
        ntimes=25,
        memory=0.1,
    )
    trajectory_to_video(density_dir)

def plot_scatter_train(args,sorted_array,t):
    plt.clf()
    plt.scatter(sorted_array[:, 0], sorted_array[:, 1], c=t, cmap='coolwarm')
    savedir = args.save
    plt.savefig(os.path.join(savedir, f"scatter_train.jpg"), dpi=300)


def plot_scatter(args,sorted_array,t,savename):
    plt.clf()
    plt.scatter(sorted_array[:, 0], sorted_array[:, 1], c=t, cmap='coolwarm')
    plt.xlim(4,5.5)
    plt.ylim(4.5, 5.5)
    savedir = args.save
    plt.savefig(os.path.join(savedir, savename), dpi=300)


def plot_causal_matrix(cmtx, class_names=None, figsize=None, vmin=None, vmax=None, show_text=True, cmap="magma"):
    """
    A function to create a colored and labeled causal matrix matplotlib figure
    given true labels and preds.
    Args:
        cmtx (ndarray): causal matrix.
        num_classes (int): total number of nodes.
        class_names (Optional[list of strs]): a list of node names.
        figsize (Optional[float, float]): the figure size of the causal matrix.
            If None, default to [6.4, 4.8].

    Returns:
        img (figure): matplotlib figure.
    """
    num_classes = cmtx.shape[0]
    if class_names is None or type(class_names) != list:
        class_names = [str(i) for i in range(num_classes)]
    figsize[0] = 30 if figsize[0] > 30 else figsize[0]
    figsize[1] = 20 if figsize[1] > 20 else figsize[1]
    plt.clf()
    plt.close("all")
    figure = plt.figure(figsize=figsize)
    plt.imshow(cmtx, interpolation="nearest",
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title("Causal matrix")
    plt.colorbar()
    # Use white text if squares are dark; otherwise black.
    threshold = cmtx.max() / 2.0
    for i, j in itertools.product(range(cmtx.shape[0]), range(cmtx.shape[1])):
        color = "white" if cmtx[i, j] < threshold else "black"
        if cmtx.shape[0] < 20 and show_text:
            # plt.text(j, i, format(cmtx[i, j], ".2e") if cmtx[i, j] != 0 else ".",
            #          horizontalalignment="center", color=color, )
            plt.text(j, i,np.around(cmtx[i, j],0), horizontalalignment="center", color=color, )
    plt.tight_layout()
    plt.savefig( "./sim_datset/sim_pred_cm_2.jpg", dpi=300)
    # return figure
