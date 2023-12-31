# -*- coding: utf-8 -*-
import argparse
from src.clean import clean_raw_data
from src.preprocess import preprocess_data
from src.train import build_and_train_model
from src.utils import initialize_experiment_directory, write_hyperparameters, create_dir_if_not_exists

# build argument parser
parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", help="name of experiment")
parser.add_argument("--tsv_path", help="path to raw .tsv dataset (defaults to raw_data/spotify_playlists.tsv)", default="raw_data/spotify_playlists.tsv")
parser.add_argument("-nr", "--num_rows", type=int, help="number of rows to extract (-1 for whole dataset) (defaults to 10000)", default=-1)
parser.add_argument("-wsz", "--window_size", type=int, help="(half) window size for skip-gram contexts (defaults to 2)", default=2)
parser.add_argument("-emb", "--embedding_dim", type=int, help="word2vec embedding dimension size (defaults to 100)", default=100)
parser.add_argument("-e", "--epochs", type=int, help="number of training epochs (defaults to 20)", default=20)
parser.add_argument("-lr", "--learning_rate", type=float, help="learning rate for Adam optimizer (defaults to 0.001)", default=0.001)
parser.add_argument("-rs", "--random_seed", type=int, help="random seed for skip-gram generation and training (defaults to 42)", default=42)
args = parser.parse_args()

# initialize experiment directory
experiment_root_dir = "experiments"
create_dir_if_not_exists(experiment_root_dir)
experiment_dir_name = experiment_root_dir + "/" + args.experiment_name
initialize_experiment_directory(experiment_dir_name)
write_hyperparameters(args, experiment_dir_name, verbose=True)

# clean raw data
clean_raw_data(experiment_dir_name, args.tsv_path, args.num_rows)

# preprocess dataset
preprocess_data(experiment_dir_name, args.window_size, args.random_seed)

# train data
build_and_train_model(experiment_dir_name, args.embedding_dim, args.learning_rate, args.epochs, args.random_seed)