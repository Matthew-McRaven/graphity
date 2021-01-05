import argparse
import torch
import os, os.path
import ray, ray.tune

import graphity.grad.predictor

def train(hypers, checkpoint_dir=None):
    nn = graphity.grad.predictor.GradPredictor(hypers['n'], hypers['layers'], alpha=hypers['alpha'],
        l2=hypers['l2'], dropout=hypers['dropout']
        )
    criterion = torch.nn.MSELoss(reduction='sum')
    print(checkpoint_dir)
    for epoch in range(hypers['epochs']):
        nn.train()
        train_losses, test_losses = [], []
        for batch in range(hypers['batch_train']):
            graphs, labels = graphity.grad.predictor.grad_generate_batch(hypers['H'], hypers['n'])
            loss = graphity.grad.predictor.grad_eval_batch(nn, graphs, labels, criterion)
            loss.backward()
            train_losses.append(loss.item())
            nn.optim.step(), nn.zero_grad()

        nn.eval()
        for batch in range(hypers['batch_test']):
            graphs, labels = graphity.grad.predictor.grad_generate_batch(hypers['H'], hypers['n'])
            loss = graphity.grad.predictor.grad_eval_batch(nn, graphs, labels, criterion)
            test_losses.append(loss.item())
        test_loss = sum(test_losses)/len(test_losses)
        with ray.tune.checkpoint_dir(step=epoch) as checkpoint_dir:
            torch.save(nn, os.path.join(checkpoint_dir,f"model.pth"))
        ray.tune.report(test_loss=test_loss)
        print(f"Train loss is {sum(train_losses) / len(train_losses):.4E}.\nTest loss is {test_loss:.4E}.\n")
        #graph, grad = grad_generate_batch(hypers['H'], hypers['n'], 1)
        #print(f"{graph}\n{grad.view(args.n, args.n).detach().numpy()}\n {nn(graph).view(args.n, args.n).detach().numpy()}\n")
    torch.save(nn, hypers['out_file'])

if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Train generator network on a task for multiple epochs and record results.")
    parser.add_argument("--out-file", dest="out_file", help="Pytorch model file.", required=True)
    parser.add_argument("-n", default=10, type=int, help="Number of nodes in graph.")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs for which to train / evaluate agents.")
    parser.add_argument("--batch-size", dest="batch_size", default=10, type=int, help="Numbers of episodes (trial runs) in each epoch. Within each epoch, each task is presented with the same seed graph")
    parser.add_argument("--train-batches", dest="batch_train", default=10, type=int, help="Number of evolution steps for each task.")
    parser.add_argument("--test-batches", dest="batch_test", default=10, type=int, help="Number of evolution steps for each task.")


    # Hamiltonian choices
    hamiltonian_group = parser.add_mutually_exclusive_group()
    hamiltonian_group.add_argument("--masked-a2d", action='store_const', const=graphity.environment.reward.LogASquaredD(2), dest='H', help="Mask out diagonal (default) when computing H.")
    hamiltonian_group.add_argument("--unmasked-a2d", action='store_const', const=graphity.environment.reward.LogASquaredD(2, keep_diag=True), dest='H', help="Keep diagonal when computing H.")
    hamiltonian_group.set_defaults(H=graphity.environment.reward.LogASquaredD(2))
    


    args = parser.parse_args()
    search_space = {
        "alpha": ray.tune.loguniform(0.0001, 0.1),
        "l2": ray.tune.uniform(0.0, .2),
        "dropout": ray.tune.uniform(0.0, .2),
        "layers": ray.tune.grid_search([
            [500, 400, 300, 200],
            [500, 500],
            [100, 500, 250],
        ]),
        **args.__dict__
    }

    ray.shutdown()  # Restart Ray defensively in case the ray connection is lost. 
    ray.init(log_to_driver=False)
    #ray.init()

    v = lambda trial_id, result: result['test_loss'] < 10
    analysis = ray.tune.run(
        train, 
        config=search_space, 
        num_samples=5,
        verbose=1,
        #name="train_mnist",  # This is used to specify the logging directory.
        stop=v # This will stop the trial 
    )
    print(analysis.get_best_trial("test_loss", "min", "last-5-avg").last_result)