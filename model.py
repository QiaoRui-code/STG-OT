from layers import ODEnet, ODEfunc, CNF, SequentialFlow

def set_cnf_options(args, model):
    def _set(module):
        if isinstance(module, CNF):
            # Set training settings
            module.solver = args.solver
            module.atol = args.atol
            module.rtol = args.rtol
            # if args.step_size is not None:
            #     module.solver_options["step_size"] = args.step_size

            # If using fixed-grid adams, restrict order to not be too high.
            if args.solver in ["fixed_adams", "explicit_adams"]:
                module.solver_options["max_order"] = 4

            # Set the test settings
            module.test_solver = args.test_solver if args.test_solver else args.solver
            module.test_atol = args.atol
            module.test_rtol = args.rtol

        if isinstance(module, ODEfunc):
            module.rademacher = False
            module.residual = False

    model.apply(_set)


def build_model_tabular(args, dims, graph, regularization_fns=None):

    hidden_dims = tuple(map(int, args.dims.split("-")))

    def build_cnf():
        diffeq = ODEnet(
            hidden_dims=hidden_dims,
            input_shape=(dims,),
            graph=graph,
            strides=None,
            conv=False,
            layer_type="GraphConvolution",
            nonlinearity="tanh",
        )
        odefunc = ODEfunc(
            diffeq=diffeq,
            divergence_fn=args.divergence_fn,
            residual=False,
            rademacher=False,
        )
        cnf = CNF(
            odefunc=odefunc,
            T=args.time_scale,
            train_T=True,
            regularization_fns=regularization_fns,
            solver=args.solver,
        )
        return cnf

    chain = [build_cnf() for _ in range(1)]
    model = SequentialFlow(chain)

    set_cnf_options(args, model)

    return model
