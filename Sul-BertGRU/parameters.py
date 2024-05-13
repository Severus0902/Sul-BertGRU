"""
@Project : Multi-LSTM (1)
@File    : parameters.py
@Author  : endeavor
@Brief   : Stores parameters
"""
import argparse


def get_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
    parser.add_argument("--input_size", type=int, default=768, help="size of each data dimension")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="the dimension or size of the hidden state in GRU cells")
    parser.add_argument("--num_layers", type=int, default=3, help="the number of stacked layers in GRU")
    parser.add_argument("--num_epochs", type=int, default=50,help="a complete iteration through a dataset")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="the number of target variables in a classification task")
    parser.add_argument("--learning_rate", type=float, default=0.002,
                        help="the step size or rate during each iteration")
    parser.add_argument("--input_channels", type=int, default=61, help="the number of channels in the input data")
    #parser.add_argument("--input_channels", type=int, default=2304, help="the number of channels in the input data")
    parser.add_argument("--output_size", type=int, default=2, help="the size of the output layer")
    parser.add_argument("--k_fold", type=int, default=10, help="number of cross-validation folds")
    obj = parser.parse_args()

    return obj
