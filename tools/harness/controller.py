import argparse
import functools
import pickle
import zipfile

import librl.agent.pg
import librl.agent.mdp
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import numpy as np
import pytest
import torch
from tabulate import tabulate

import librl.agent.pg
import librl.nn.core, librl.nn.actor, librl.nn.critic, librl.nn.pg_loss
import librl.reward, librl.replay.episodic
import librl.task, librl.task.cc
import librl.train.train_loop, librl.train.cc
from torch.distributions import categorical

import graphity.agent.core, graphity.strategy.grad, graphity.strategy.base
import graphity.environment.sim, graphity.environment.reward
import graphity.task.task, graphity.task.sampler
import graphity.grad.predictor
import graphity.train.cc

######################
#    Init Agents     #
######################

agents = {}
def register_agent(agent_fn):
    @functools.wraps(agent_fn)
    def wrapper(agent_fn):
        return agent_fn
    global agents
    if not agent_fn.__name__ in agents:
        agents[agent_fn.__name__] = agent_fn
    return wrapper

@register_agent   
def random_helper(hypers, env, *args):
    agent = librl.agent.mdp.RandomAgent(env.observation_space, env.action_space)
    return agent

@register_agent
def metropolis_helper(hypers, env, *args):
    agent = graphity.agent.core.MetropolisAgent(graphity.strategy.base.random_sampling_strategy(1))
    return agent

@register_agent
def grad_descent_helper(hypers, env, *args):
    grad_fn = graphity.strategy.grad.TrueGrad(env.H)
    agent = graphity.agent.core.ForwardAgent(graphity.strategy.grad.gd_sampling_strategy(grad_fn))
    return agent

@register_agent
def metropolis_grad_descent_helper(hypers, env, *args):
    grad_fn = graphity.strategy.grad.TrueGrad(env.H)
    agent = graphity.agent.core.MetropolisAgent(graphity.strategy.grad.beta_sampling_strategy(grad_fn))
    return agent

@register_agent
def stochastic_grad_descent_helper(hypers, env, *args):
    grad_fn = graphity.strategy.grad.TrueGrad(env.H)
    agent = graphity.agent.core.ForwardAgent(graphity.strategy.grad.beta_sampling_strategy(grad_fn))
    return agent

@register_agent
def neural_grad_descent_helper(hypers, env, *args):
    assert 'pretrained' in hypers and hypers['pretrained'], "When specifiying nn-based models, \"--pretrained-model\" is required."
    grad_fn = graphity.strategy.grad.NeuralGrad(torch.load(hypers['pretrained']))
    agent = graphity.agent.core.ForwardAgent(graphity.strategy.grad.gd_sampling_strategy(grad_fn))
    return agent

@register_agent
def simulated_annealing_helper(hypers, env, *args):
    grad_fn = graphity.strategy.grad.TrueGrad(env.H)
    # Don't give simulated annealing grad descent, that isn't a fair comparison for us!
    # We proposed grad descent as a search strategy.
    agent = graphity.agent.core.SimulatedAnnealingAgent(graphity.strategy.base.random_sampling_strategy(),
        .75, 10, .5
    )
    return agent
"""
Until errors in RL algorithms have been resolved, it is not advised to use this code.
"""
@register_agent
def vpg_helper(hypers, env, critic_net, policy_net):
    agent = librl.agent.pg.REINFORCEAgent(policy_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent.train()
    return agent

@register_agent   
def pgb_helper(hypers, env, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PGB(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

@register_agent
def ppo_helper(hypers, env, critic_net, policy_net):
    actor_loss = librl.nn.pg_loss.PPO(critic_net, explore_bonus_fn=librl.reward.basic_entropy_bonus())
    agent = librl.agent.pg.ActorCriticAgent(critic_net, policy_net, actor_loss=actor_loss)
    agent.train()
    return agent

#######################
#  Init Environments  #
#######################
def build_actor_critic(hypers, env):
    flatten = graphity.FlattenInput(env.observation_space.shape)
    policy_kernel = librl.nn.core.RecurrentKernel(flatten.output_dimension, 512, 10)
    policy_linked = librl.nn.core.SequentialKernel([flatten, policy_kernel])
    policy_net = librl.nn.actor.BiCategoricalActor(policy_linked, 
        env.action_space, env.observation_space
    )
    critic_kernel = librl.nn.core.RecurrentKernel(flatten.output_dimension, 512, 10)
    critic_net= librl.nn.critic.ValueCritic(critic_kernel)
    return critic_net, policy_net 

#######################
#     Entry point     #
#######################
def run_shared(hypers={}):
    if not hypers['seed']:
        hypers['seed'] = np.random.randint(0, 2**32, (1,))
    dist = graphity.task.task.TaskDistribution()
    sampler = graphity.task.sampler.CachedSampler(graph_size=hypers['n'], seed=hypers['seed'])
    hypers['env'] = graphity.environment.sim.Simulator(hypers['H'], 
        hypers['n'], sampler=sampler
    )

    global agents
    for idx, alg in enumerate(hypers['alg']):
        critic, actor = build_actor_critic(hypers, hypers['env'])
        critic, actor = critic.to(hypers['device']), actor.to(hypers['device'])
        agent = agents[alg](hypers, hypers['env'], critic, actor)
        alg_name = "_".join(alg.split("_")[:-1])

        dist.add_task(librl.task.Task.Definition(graphity.task.task.GraphTask, 
            agent=agent, name=alg_name, number = idx, env=hypers['env'], 
            sampler=hypers['env'].sampler,
            episode_length=hypers['episode_length'],
            trajectories=hypers['trajectories'])
        )

        hypers[f"model_{idx:02d}"] = alg_name
    logger = graphity.train.cc.DirLogger(args.H, args.log_file)
    logger.log_main(__file__)
    graphity.train.cc.group_trainer(hypers, dist, librl.train.cc.policy_gradient_step, 
        logger=logger
    )

def run_entry_point(args):
    hypers = {}
    hypers['H'] = args.H
    hypers['n'] = args.n
    hypers['pretrained'] = args.pretrained
    hypers['device'] = args.device
    hypers['epochs'] = args.epochs
    hypers['trajectories'] = args.trajectories
    hypers['episode_length'] = args.episode_length
    hypers['alg'] = args.alg
    hypers['seed'] = args.seed
    run_shared(hypers)

def rerun_entry_point(args):
    hypers = {}
    with zipfile.ZipFile(args.old_log_file) as old_results:
        with old_results.open("config.pkl") as file:
            hypers = pickle.load(file)
    if args.H: hypers['H'] = args.H
    if args.pretrained: hypers['pretrained'] = args.pretrained
    hypers['device'] = args.device     
    run_shared(hypers)

def compute_stats(hypers, results, models):
    experiments = {}
    for idx, alg in models.items():
        # Stored as min, max, avg, std
        experiments[idx] = {'overall':{}}
        overall_min, overall_max = float("inf"), -float("inf")
        overall_rewards = []
        for epoch in range(hypers['epochs']):
            epoch_rewards = []
            epoch_min, epoch_max = float("inf"), -float("inf")
            for trajectory in range(hypers['trajectories']):
                with results.open(f"models/{idx:02d}/{epoch:04d}/traj{trajectory}.pkl", "r") as file:
                    traj = pickle.load(file)
                    # Trajectory's reward buffer may not be a tensor.
                    # So, accumulate all rewards, then stack them.
                    rewards = [traj.reward_buffer[idx]for idx in range(min(traj.done, len(traj.reward_buffer)))]
                    rewards = torch.stack(rewards).exp()
                    epoch_rewards.append(rewards)
            epoch_rewards = torch.cat(epoch_rewards).view(-1)
            overall_rewards.append(epoch_rewards)
            epoch_min = torch.min(epoch_rewards)
            epoch_max = torch.max(epoch_rewards)
            epoch_avg = epoch_rewards.sum().item() / len(epoch_rewards)
            epoch_std = torch.std(epoch_rewards).item()
            
            # Log all relevant statistics on a per-epoch basis. 
            experiments[idx][epoch] = {}
            experiments[idx][epoch]['min'] = int(epoch_min)
            experiments[idx][epoch]['%min'] = (epoch_rewards == epoch_min).sum().item() / len(epoch_rewards)
            experiments[idx][epoch]['max'] = int(epoch_max)
            experiments[idx][epoch]['avg'] = epoch_avg
            experiments[idx][epoch]['std'] = epoch_std

        overall_rewards = torch.cat(overall_rewards).view(-1)
        overall_min = torch.min(overall_rewards)
        overall_max = torch.max(overall_rewards)
        overall_avg = overall_rewards.sum().item() / len(overall_rewards)
        overall_std = torch.std(overall_rewards).item()

        # Log relevant statisics on a per-model basis.
        experiments[idx]["overall"]['min'] = int(overall_min)
        experiments[idx]["overall"]['%min'] = (overall_rewards == overall_min).sum().item() / len(overall_rewards)
        experiments[idx]["overall"]['max'] = int(overall_max)
        experiments[idx]["overall"]['avg'] = overall_avg
        experiments[idx]["overall"]['std'] = overall_std
    return experiments

def graph_entry_point(args):
    import matplotlib.pyplot as plt
    hypers, models = {}, {}
    columns = []
    with zipfile.ZipFile(args.log_file) as results:
        with results.open("config.pkl") as file: hypers = pickle.load(file)
        with results.open("models.pkl") as file: models = pickle.load(file)
        stats = compute_stats(hypers, results, models)
        for _, values in stats.items():
            local_column = [values[epoch]['min'] for epoch in range(hypers['epochs'])]
            columns.append(local_column)
        
    model_strs = ["_".join(alg.split("_")[:-1]) for _, alg in models.items()]
    for idx in range(len(columns)):
        plt.plot(columns[idx])
    plt.ylabel('Energy Level'), plt.xlabel('Epoch')
    plt.ylim(bottom=0)
    plt.legend([alg for alg in model_strs], bbox_to_anchor=(0.5, 1.05), loc='upper center', fontsize='xx-small',
        fancybox=True, ncol=3, shadow=True
    )
    plt.show()

def table_entry_point(args):
    with zipfile.ZipFile(args.log_file) as results:
        hypers, models = {}, {}
        with results.open("config.pkl") as file: hypers = pickle.load(file)
        with results.open("models.pkl") as file: models = pickle.load(file)
        stats = compute_stats(hypers, results, models)
        rows = None
        if args.table_mode == "summary":
            rows = [("_".join(alg.split("_")[:-1]) for _, alg in models.items())]
            columns = []
            for _, values in stats.items():
                local_column = [f"{values['overall']['min']} ({values['overall']['%min']})"]
                columns.append(local_column)
            rows.extend(zip(*columns))
            print("\Global Minimums (% minimum graphs)\n", tabulate(rows, headers="firstrow", showindex="False"))
        elif args.table_mode == "details":
            headers = ["epoch"]
            headers.extend(("_".join(alg.split("_")[:-1]) for _, alg in models.items()))
            rows = [headers]
            columns = []
            for _, values in stats.items():
                local_column = [values[epoch]['min'] for epoch in range(hypers['epochs'])]
                columns.append(local_column)
            rows.extend(zip(*columns))
            print("\tPer-Epoch Minimums\n", tabulate(rows, headers="firstrow", showindex=True))
        else: raise NotImplementedError()
        
def seed_entry_point(args):
    with zipfile.ZipFile(args.log_file) as results:
        with results.open(f"seed/{args.epoch:04d}.csv") as file: 
            x = np.genfromtxt(file, delimiter=',')
            print(x)

def config_entry_point(args):
    rows, hypers = [["Hyperparameter", "value"]], {}
    with zipfile.ZipFile(args.log_file) as old_results:
        with old_results.open("config.pkl") as file:
            hypers = pickle.load(file)
    for key, value in hypers.items():
        if key == "H": continue
        elif "model" in key: continue
        elif "env" in key: continue
        elif "alg" == key: continue
        elif value == None: continue
        else: rows.append([key, value])
    for alg in hypers['alg']:
        rows.append(['using algorithm', "_".join(alg.split("_")[:-1])])
    print(tabulate(rows, headers="firstrow", showindex="never"))
    
def ls_entry_point(args):
    hypers = {}
    with zipfile.ZipFile(args.log_file) as old_results:
        with old_results.open("models.pkl") as file:
            hypers = pickle.load(file)
            rows = [("Exp. Index", "Exp. Name")]
            rows.extend([(idx, "_".join(alg.split("_")[:-1])) for idx, alg in  hypers.items()])
            print(tabulate(rows, headers="firstrow", showindex="never"))
 
def train_entry_point(args):
    nn = graphity.grad.predictor.GradPredictor(args.n, [500, 400, 300, 200])
    graphity.grad.predictor.grad_train_loop(args.__dict__, nn)
    torch.save(nn, args.out_file)

def main(args):
    opts = {
        "run": run_entry_point, "rerun": rerun_entry_point,
        "table": table_entry_point, "graph": graph_entry_point,
        "config": config_entry_point, "seed": seed_entry_point,
        "ls": ls_entry_point, "train": train_entry_point
    }
    if args.subcommand in opts:  opts[args.subcommand](args)
    else: raise NotImplementedError("Don't use that command!")

##################
# Parsing Config #
##################
def create_alg_options(parser, set_default=True):
    learn_alg_group = parser.add_argument_group()
    learn_alg_group.add_argument("--random", action='append_const', const=random_helper.__name__, dest='alg', help="Perform a random search of the state space.")
    learn_alg_group.add_argument("--metropolis", action='append_const', const=metropolis_helper.__name__, dest='alg', help="Use a MCMC method to back out bad actions.")
    learn_alg_group.add_argument("--gd", action='append_const', const=grad_descent_helper.__name__, dest='alg', help="Search state space using grad descent.")
    learn_alg_group.add_argument("--mgd", action='append_const', const=metropolis_grad_descent_helper.__name__, dest='alg', help="Search state space using metropolis grad descent.")
    learn_alg_group.add_argument("--sgd", action='append_const', const=stochastic_grad_descent_helper.__name__, dest='alg', help="Search state space using stochastic grad descent.")
    learn_alg_group.add_argument("--nngd", action='append_const', const=neural_grad_descent_helper.__name__, dest='alg', help="Search state space using stochastic grad descent.")
    learn_alg_group.add_argument("--sa", action='append_const', const=simulated_annealing_helper.__name__, dest='alg', help="Search state space using simulated annealing.")
    learn_alg_group.add_argument("--vpg", action='append_const', const=vpg_helper.__name__, dest='alg', help="Train a RL agent using VPG.")
    learn_alg_group.add_argument("--pgb", action='append_const', const=pgb_helper.__name__, dest='alg', help="Train a RL agent using PGB.")
    learn_alg_group.add_argument("--ppo", action='append_const', const=ppo_helper.__name__, dest='alg', help="Train a RL agent using PPO.")
    if set_default: learn_alg_group.set_defaults(alg=[])
    parser.add_argument("--pretrained-model", dest="pretrained", help="Pretrained neural net model. Required if \"--nngd\" is selected.")

def create_H_options(parser, set_default=True):
    # Hamiltonian choices
    hamiltonian_group = parser.add_mutually_exclusive_group()
    # TODO: Add helper functions that instantiate reward fn's, don't create them here.
    hamiltonian_group.add_argument("--masked-a2d", action='store_const', const=graphity.environment.reward.LogASquaredD(2), dest='H', help="Mask out diagonal (default) when computing H.")
    hamiltonian_group.add_argument("--unmasked-a2d", action='store_const', const=graphity.environment.reward.LogASquaredD(2, keep_diag=True), dest='H', help="Keep diagonal when computing H.")
    hamiltonian_group.add_argument("--ising", action='store_const', const=graphity.environment.reward.IsingHamiltonian(), dest='H', help="Keep diagonal when computing H.")
    hamiltonian_group.add_argument("--spin-glass", action='store_const', const=graphity.environment.reward.SpinGlassHamiltonian(categorical=True), dest='H', help="Keep diagonal when computing H.")
    if set_default: hamiltonian_group.set_defaults(H=graphity.environment.reward.LogASquaredD(2))
    

# Invoke main, construct CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train generator network on a task for multiple epochs and record results.")
    subparser = parser.add_subparsers(title="subcommands", description="Valid subcommands", dest="subcommand", required=True)
   
    # Create "run" section of experiment
    run_parser = subparser.add_parser(name="run")
    run_parser.add_argument("-f", dest="log_file", help="File in which to store logs. Should end in \".zip\"")
    run_parser.add_argument("-n", default=10, type=int, help="Number of nodes in graph.")
    run_parser.add_argument("--epochs", default=1000, type=int, help="Number of epochs for which to train / evaluate agents.")
    run_parser.add_argument("--trajectories", dest="trajectories", default=1, type=int, help="Numbers of episodes (trial runs) in each epoch. Within each epoch, each task is presented with the same seed graph")
    run_parser.add_argument("--timesteps", dest="episode_length", default=100, type=int, help="Number of evolution steps for each task.")
    run_parser.add_argument("--device", dest="device", help="Which device to run torch on. Pick (cuda) or (cpu).", default="cpu")
    run_parser.add_argument("--seed", dest="seed", type=int, help="String to use as np, torch seeds.")
    create_alg_options(run_parser)
    create_H_options(run_parser)
    
    # Create "re-run" section of experiment
    rerun_parser = subparser.add_parser(name="rerun")
    rerun_parser.add_argument("-f", dest="log_file", help="File in which to store logs. Should end in \".zip\"", required=True)
    rerun_parser.add_argument(dest="old_log_file", help="File from which to read old configs. Should end in \".zip\"")
    rerun_parser.add_argument("--device", dest="device", help="Which device to run torch on. Pick (cuda) or (cpu).", default="cpu")
    create_alg_options(rerun_parser)
    create_H_options(rerun_parser)

    # Inspection subcommands   
    table_parser = subparser.add_parser(name="table")
    table_parser.add_argument("-f", dest="log_file", help="File from which to read logs. Should end in \".zip\"", required=True)
    # Table mode choices
    table_mode_group = table_parser.add_mutually_exclusive_group()
    table_mode_group.add_argument("--summary", action='store_const', const="summary", dest='table_mode', help="Mask out diagonal (default) when computing H.")
    table_mode_group.add_argument("--details", action='store_const', const="details", dest='table_mode', help="Keep diagonal when computing H.")
    table_mode_group.set_defaults(table_mode="summary")
    
    graph_parser = subparser.add_parser(name="graph")
    graph_parser.add_argument("-f", dest="log_file", help="File from which to read logs. Should end in \".zip\"", required=True)
    seed_parser = subparser.add_parser(name="seed")
    seed_parser.add_argument("-f", dest="log_file", help="File from which to read logs. Should end in \".zip\"", required=True)
    seed_parser.add_argument(dest="epoch", type=int, help="Which epochs' seed to print")

    # Create "config" section of experiment
    config_parser = subparser.add_parser(name="config", description="Display the configuration for a particular experiment")
    config_parser.add_argument("-f", dest="log_file", help="File in which to store logs. Should end in \".zip\"", required=True)

    # Create "ls" section of experiment
    ls_parser = subparser.add_parser(name="ls")
    ls_parser.add_argument("-f", dest="log_file", help="File in which to store logs. Should end in \".zip\"", required=True)

    # Create "train" section of experiment
    train_parser = subparser.add_parser(name="train")
    train_parser.add_argument("--out-file", dest="out_file", help="Pytorch model file.", required=True)
    train_parser.add_argument("-n", default=10, type=int, help="Number of nodes in graph.")
    train_parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for which to train / evaluate agents.")
    train_parser.add_argument("--batch-size", dest="batch_size", default=10, type=int, help="Numbers of episodes (trial runs) in each epoch. Within each epoch, each task is presented with the same seed graph")
    train_parser.add_argument("--train-batches", dest="batch_train", default=10, type=int, help="Number of evolution steps for each task.")
    train_parser.add_argument("--test-batches", dest="batch_test", default=10, type=int, help="Number of evolution steps for each task.")
    create_H_options(train_parser)

    args = parser.parse_args()
    main(args)

#######################
#  Integration Tests  #
#######################

# Test that train generates a pretrained nn for various sizes.
def test_train():
    pass

# Tests that "run" can generate a ziparchive, which is needed for all other tests.
def test_run():
    pass

# Create a dummy ziparchive, and check that ls generates output.
def test_ls():
    pass

# Create a dummy ziparchive, and check that config generates output.
def test_config():
    pass

# Create a dummy ziparchive, and check that seed generates output.
def test_seed():
    pass
# Check that seed fails when given insane epoch #'s.
def test_seed_fail():
    pass

# Create a dummy ziparchive, and check that grapg generates images.
def test_graph():
    # TODO: Must set pyplot to be non-interactive.
    pass
# Check that graph fails when given insane epoch #'s.
def test_seed_graph():
    pass

# Test that nngd fails when not provided pretrained model.
def test_nn_requires_pretrained_fail():
    pass
# Test that nngd can be used as an agent.
def test_nn_requires_pretrained():
    pass