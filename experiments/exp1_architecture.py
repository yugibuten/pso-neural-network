"""Experiment 1: architecture. Run explicitly with python -m."""
import json
from .configs import architecture_configs
from .runner import common_parser, run_suite


def main():
    parser = common_parser(__doc__, 'exp1.csv', needs_base=False)
    args = parser.parse_args()
    configs = architecture_configs()
    if args.list:
        print(json.dumps(configs, indent=2))
        print(f'{len(configs)} settings; {len(configs)*10} runs')
        return
    run_suite(configs, args.output, args.data, resume=args.resume)


if __name__ == '__main__':
    main()
