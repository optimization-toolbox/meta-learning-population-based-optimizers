
import argparse

from mlo import Experiment

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path") 
    args = parser.parse_args()
    config_path = args.config_path

    exp = Experiment(config_path)
    exp.run()