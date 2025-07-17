import optuna
import argparse
from main import run_training
from config import Config

def parse_args_from_trial(trial):
    suggested_new_lr = trial.suggest_float("new_lr", 1e-5, 1e-3, log=True)
    suggested_pretrained_lr = trial.suggest_float("pretrained_lr", 1e-5, 1e-3, log=True)
    suggested_graph_layers = trial.suggest_int("graph_layers", 2, 5, log=True)

    args = argparse.Namespace()

    args.dataset = "cdr"
    args.save_path = "best.pt"
    args.log_path = "log.txt"
    args.seed = 2004
    args.tqdm = False

    args.num_epoch = 30
    args.batch_size = 4
    args.update_freq = 1
    args.warmup_ratio = 0.06
    args.max_grad_norm = 1.0

    # Suggest hyperparameters with optuna
    args.new_lr = suggested_new_lr
    args.pretrained_lr = suggested_pretrained_lr
    args.adam_epsilon = 1e-6

    args.device = "cuda:0"
    args.transformer = "bert-base-cased"
    args.type_dim = 20
    args.graph_layers = suggested_graph_layers

    args.lower_temp = 2.0
    args.upper_temp = 20.0
    args.loss_tradeoff = 1.0

    return args


def objective(trial):
    args = parse_args_from_trial(trial)
    cfg = Config(args)
    val_score = run_training(cfg, no_tqdm=True)
    return val_score

if __name__ == '__main__':
    study = optuna.create_study(
        direction="maximize",
        study_name="tuning",
    )
    study.optimize(objective, n_trials=10)

    print("Best trial:")
    print(study.best_trial)

