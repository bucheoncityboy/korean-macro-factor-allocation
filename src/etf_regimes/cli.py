from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description='Paper-aligned global ETF regime research')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('acquire')
    audit_parser = sub.add_parser('audit')
    audit_parser.add_argument('--freeze', action='store_true')
    run_parser = sub.add_parser('run')
    run_parser.add_argument('--workers', type=int, default=4)
    verify_parser = sub.add_parser('verify')
    verify_parser.add_argument('--artifacts-only', action='store_true')
    sub.add_parser('report')
    args = parser.parse_args()
    if args.command == 'acquire':
        from .data import acquire
        from .snapshots import restore_raw_snapshot
        restore_raw_snapshot()
        result = acquire()
        print(f"Acquired {len(result['objects'])} immutable public-source objects")
    elif args.command == 'audit':
        from .data import audit
        result = audit()
        print(json.dumps(result, indent=2, default=str))
        if args.freeze:
            from .study import freeze
            freeze()
            print('Execution contract saved; commit before run')
    elif args.command == 'run':
        from .study import run
        run(args.workers)
    elif args.command == 'verify':
        from .verification import verify
        print(json.dumps(verify(args.artifacts_only), indent=2, default=str))
    elif args.command == 'report':
        from .reporting import report
        report()


if __name__ == '__main__':
    main()
