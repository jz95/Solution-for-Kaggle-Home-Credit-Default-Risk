import argparse
import datetime
from src.model import run


def main():
    parser = argparse.ArgumentParser(
        description="Train a model, perform random parameter search \
        or make prediction.")
    parser.add_argument(
        "-m", "--mode",
        choices=['all', 'train', 'predict', 'search', 'extract', 'stacking'],
        default='all',
        help="choose a mode to start the program,\
        mode - all: train the model and \
        make prediction as submission for competition.\
        mode - train: train the model.\
        mode - predict: make prediction using the trained model.\
        mode - search: perform random parameter search.\
        mode - extract: only extract features but not train the model.\
        mode - stacking: run the model in stacking mode.")
    parser.add_argument(
        "--train-data",
        help="specify the train dataset file path,\
        the model would skip feature extracting and run\
        directly on it.")
    parser.add_argument(
        "--test-data",
        help="specify the test dataset file path,\
        only useful in ALL mode and when you would like\
        to skip feature extraction.")
    parser.add_argument(
        "-n", "--name",
        help="name and start this running, the result\
        would be saved in the workspace.")
    parser.add_argument(
        "-d", "--debug", action='store_true',
        help="running the model in debug mode,\
        i.e. only training a few samples helps \
        you locate the possible bugs quickly.")
    parser.add_argument(
        "-t", "--type",
        default='lgb',
        choices=['lgb', 'xgb', 'lgb-forest'],
        help="choose wether to use lightGBM or XGBoost Libary.")

    args = parser.parse_args()
    if args.name:
        name = args.name
    else:
        time = datetime.datetime.now()
        name = 'MODEL_' + time.strftime('%Y_%m_%d_%H_%M_%S')

    if args.mode == 'search' and not args.name:
        name = 'PARAM_SEARCH_' + name

    if args.mode == 'stacking' and not args.name:
        name = 'STACKING_' + name

    if args.debug and not args.name:
        name += '_DEBUG'

    run(
        name=name,
        debug=args.debug,
        mode=args.mode,
        type_=args.type,
        train_path=args.train_data,
        test_path=args.test_data
    )


if __name__ == '__main__':
    main()
