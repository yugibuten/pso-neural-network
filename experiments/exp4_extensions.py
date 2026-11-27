"""Experiment 4: extensions. Run explicitly with python -m."""
import json
from .configs import extension_configs
from .runner import common_parser, run_suite


def main():
    parser = common_parser(__doc__, 'exp4.csv', needs_base=True)
    args = parser.parse_args()
    configs = extension_configs(json.loads(args.base.read_text()))
    if args.list:
        print(json.dumps(configs, indent=2))
        print(f'{len(configs)} settings; {len(configs)*10} runs')
        return
    run_suite(configs, args.output, args.data, resume=args.resume)


if __name__ == '__main__':
    main()
