import json
import argparse
from trainer import train

def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    cli_args = vars(args)

    # CLI arguments override JSON settings only when explicitly provided (not None)
    for key, value in cli_args.items():
        if value is not None:
            param[key] = value

    param.setdefault("use_wandb", False)
    train(param)

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--use_wandb', action='store_true', default=None,
                        help='Enable Weights & Biases experiment tracking.')
    parser.add_argument('--no_wandb', action='store_false', dest='use_wandb',
                        help='Disable Weights & Biases experiment tracking.')
    parser.add_argument('--wandb_project', type=str, default=None,
                        help='Weights & Biases project name.')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases entity (username or team).')
    parser.add_argument('--wandb_group', type=str, default=None,
                        help='Weights & Biases run group.')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=None,
                        help='Additional tags for Weights & Biases run.')
    parser.add_argument('--wandb_mode', type=str, default=None, choices=['online', 'offline', 'disabled'],
                        help='Weights & Biases running mode (online, offline, disabled).')
    return parser

if __name__ == '__main__':
    main()
