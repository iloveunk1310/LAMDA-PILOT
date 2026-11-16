import json
import argparse
import logging
import os

BENCHMARKS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'exps', 'benchmarks.json')
NON_PARAM_CLI_KEYS = {'config', 'method', 'benchmark', 'dry_run'}
NATURAL_224_DATASETS = {'cub', 'imagenetr', 'imageneta', 'omnibenchmark', 'vtab'}


def main():
    args = setup_parser().parse_args()
    runs = resolve_runs(args)

    if args.dry_run:
        for idx, run in enumerate(runs, 1):
            param = run['param']
            print(
                f"[{idx}/{len(runs)}] preset={run['preset'] or 'custom'} "
                f"template={run['config_path']} "
                f"model={param.get('model_name')} "
                f"backbone={param.get('backbone_type')} "
                f"dataset={param.get('dataset')} "
                f"init_cls={param.get('init_cls')} "
                f"increment={param.get('increment')}"
            )
        return

    from trainer import train
    for idx, run in enumerate(runs, 1):
        param = run['param']
        if len(runs) > 1:
            print(
                f"\n=== Running [{idx}/{len(runs)}] preset={run['preset']} "
                f"dataset={param.get('dataset')} (b0 inc{param.get('increment')}) ==="
            )
        # Reset root logger handlers between sequential cluster runs so each run writes to its own log file
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            handler.close()
        train(param)


def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param


def load_benchmarks(benchmarks_path=BENCHMARKS_PATH):
    return load_json(benchmarks_path)


def _select_template_group(dataset_name, preset=None):
    if dataset_name is not None:
        ds = dataset_name.lower()
        if ds == 'cifar224':
            return 'cifar'
        if ds in NATURAL_224_DATASETS:
            return 'inr'
    if preset and preset.get('template_group') in ('cifar', 'inr'):
        return preset['template_group']
    return 'cifar'


def resolve_runs(args, benchmarks_path=BENCHMARKS_PATH):
    benchmarks = None
    if args.benchmark is not None or args.method is not None:
        benchmarks = load_benchmarks(benchmarks_path)

    if args.benchmark is not None:
        clusters = benchmarks.get('clusters', {})
        presets = benchmarks.get('presets', {})
        if args.benchmark in clusters:
            preset_names = list(clusters[args.benchmark])
        elif args.benchmark in presets:
            preset_names = [args.benchmark]
        else:
            valid = sorted(list(presets.keys()) + list(clusters.keys()))
            raise ValueError(
                f"Unknown benchmark '{args.benchmark}'. Available presets/clusters: {', '.join(valid)}"
            )
    else:
        preset_names = [None]

    cli_args = vars(args)
    resolved = []
    for preset_name in preset_names:
        preset = benchmarks['presets'][preset_name] if preset_name is not None else {}
        target_dataset = args.dataset if args.dataset is not None else preset.get('dataset')

        if args.method is not None:
            methods = benchmarks.get('methods', {})
            if args.method not in methods:
                valid_methods = sorted(methods.keys())
                raise ValueError(
                    f"Unknown method '{args.method}'. Available methods: {', '.join(valid_methods)}"
                )
            template_group = _select_template_group(target_dataset, preset)
            config_path = methods[args.method][template_group]
        else:
            config_path = args.config

        param = load_json(config_path)

        if preset:
            param['dataset'] = preset['dataset']
            param['init_cls'] = preset['init_cls']
            param['increment'] = preset['increment']

        # CLI arguments override JSON/preset settings only when explicitly provided (not None)
        for key, value in cli_args.items():
            if key in NON_PARAM_CLI_KEYS:
                continue
            if value is not None:
                param[key] = value

        param.setdefault('use_wandb', False)
        resolved.append({
            'preset': preset_name,
            'config_path': config_path,
            'param': param,
        })

    return resolved


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='./exps/simplecil.json',
                        help='Json file of settings.')
    parser.add_argument('--method', type=str, default=None,
                        help='Method name from exps/benchmarks.json (auto-selects cifar or *_inr template).')
    parser.add_argument('--benchmark', type=str, default=None,
                        help='Benchmark preset or cluster name from exps/benchmarks.json (e.g., cifar_b0inc5, fast, all_6).')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Override dataset name (e.g., cifar224, cub, imagenetr, imageneta, omnibenchmark, vtab).')
    parser.add_argument('--init_cls', type=int, default=None,
                        help='Override initial number of classes in task 0.')
    parser.add_argument('--increment', type=int, default=None,
                        help='Override number of classes added per incremental task.')
    parser.add_argument('--backbone_type', type=str, default=None,
                        help='Override backbone type (e.g., pretrained_vit_b16_224_adapter_in21k).')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training and evaluation.')
    parser.add_argument('--init_lr', type=float, default=None,
                        help='Initial learning rate.')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='Resolve benchmark/method configuration(s) and print summary without training.')
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
