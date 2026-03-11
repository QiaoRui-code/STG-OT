import argparse
import layers
SOLVERS = ["dopri5", "bdf", "rk4", "midpoint", "adams", "explicit_adams", "fixed_adams"]

parser = argparse.ArgumentParser("Continuous Normalizing Flow")
parser.add_argument(
    "--embedding_name",
    type=str,
    default='pca',
    help="choose embedding name to perform TrajectoryNet on",
)
parser.add_argument("--use_velocity", type=eval, default=True)
parser.add_argument("--no_display_loss", action="store_false")

######moon
# parser.add_argument("--save", type=str, default="../results/tmp/moon")
# parser.add_argument("--max_dim", type=int, default=2)
# parser.add_argument("--total_niters", type=int, default=2)
# parser.add_argument("--graph_niters", type=int, default=100)
# parser.add_argument("--niters", type=int, default=1000)
# parser.add_argument("--dims", type=str, default="2-2-2")
# parser.add_argument("--batch_size", type=int, default=10)
# parser.add_argument("--time_scale", type=float, default=0.5)


# ######lorenz96
# parser.add_argument("--save", type=str, default="../results/tmp/lorenz")
# parser.add_argument("--max_dim", type=int, default=10)
# parser.add_argument("--total_niters", type=int, default=3)
# parser.add_argument("--graph_niters", type=int, default=100)
# parser.add_argument("--niters", type=int, default=200)
# parser.add_argument("--batch_size", type=int, default=10)
# parser.add_argument("--time_scale", type=float, default=0.5)
# parser.add_argument("--dims", type=str, default="10-10-10")
# parser.add_argument("--interp_reg", type=float, default=TRUE, help="regularize interpolation")
######EmbryoidBody plot
# parser.add_argument("--save", type=str, default="../results/tmp/embry/dim10")
# parser.add_argument("--max_dim", type=int, default=10)
# parser.add_argument("--total_niters", type=int, default=1)
# parser.add_argument("--graph_niters", type=int, default=500)
# parser.add_argument("--niters", type=int, default=500)
# parser.add_argument("--dims", type=str, default="10-10-10")
# parser.add_argument("--time_scale", type=float, default=0.5)
# parser.add_argument("--batch_size", type=int, default=10)
######EmbryoidBody plot
parser.add_argument("--save", type=str, default="/home/lenovo/jora/")
parser.add_argument("--max_dim", type=int, default=10)
parser.add_argument("--total_niters", type=int, default=1)
parser.add_argument("--graph_niters", type=int, default=100)
parser.add_argument("--niters", type=int, default=200)
parser.add_argument("--dims", type=str, default="30-30-30")
parser.add_argument("--time_scale", type=float, default=0.5)
parser.add_argument("--batch_size", type=int, default=128)


parser.add_argument("--atol", type=float, default=1e-5)
parser.add_argument("--rtol", type=float, default=1e-5)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=1e-5)

parser.add_argument("--training_noise", type=float, default=0.1)
parser.add_argument('--sinkhorn_scaling', default=0.7, type=float)
parser.add_argument('--sinkhorn_blur', default=0.05, type=float)
# parser.add_argument('--ot_weight', default=0.5, type=float)

parser.add_argument("--viz_freq", type=int, default=1000)
parser.add_argument("--save_freq", type=int, default=1000)
parser.add_argument("--viz_batch_size", type=int, default=1000)
# parser.add_argument("--num_blocks", type=int, default=1, help="Number of stacked CNFs.")
# parser.add_argument("--nonlinearity", type=str, default="tanh", choices=layers.NONLINEARITIES)
# parser.add_argument("--residual", action="store_true")
# parser.add_argument("--rademacher", action="store_true")
# parser.add_argument("--train_T", type=eval, default=True)
parser.add_argument("--leaveout_timepoint", type=int, default=-1)

parser.add_argument("--vecint", type=float, default=0, help="regularize direction")
parser.add_argument("--top_k_reg", type=float, default=0.0, help="density following regularization")


# Track quantities
parser.add_argument("--l1int", type=float, default=None, help="int_t ||f||_1")
parser.add_argument("--l2int", type=float, default=None, help="int_t ||f||_2")
parser.add_argument("--sl2int", type=float, default=None, help="int_t ||f||_2^2")


parser.add_argument(
    "--divergence_fn",
    type=str,
    default="brute_force",
    choices=["brute_force", "approximate"],
)

parser.add_argument("--solver", type=str, default="rk4", choices=SOLVERS)
parser.add_argument("--test_solver", type=str, default=None, choices=SOLVERS + [None])
parser.add_argument("--interp_reg", type=float, default=None, help="regularize interpolation")