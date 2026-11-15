"""Choose a revised-manuscript study; forward remaining options to its runner."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True, choices=("parameters", "quench"))
    args, remaining = parser.parse_known_args()
    if args.study == "parameters":
        from experiments.parameter_study.run import main as run
    else:
        from experiments.local_quench.run import main as run
    run(remaining)


if __name__ == "__main__":
    main()
