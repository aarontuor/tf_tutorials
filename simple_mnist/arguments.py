#docs: https://docs.python.org/3/library/argparse.html
#tutorial: https://docs.python.org/2/howto/argparse.html

import argparse

parser = argparse.ArgumentParser(description="Demonstrate basic argparse functionality")
parser.add_argument("-train_feat",  metavar="TRAIN_FEAT_FN", type=str, required=True,
                    help="The name of the training set feature file. The file should contain N lines"
                         "(where N is the number of data points), and each line should contains D space-delimited"
                         "floating point values (where D is the feature dimension).")
parser.add_argument("-epochs",  metavar="EPOCHS", type=int, required=True,
                    help="The total number of epochs (i.e. passes through the data) to train for. If "
                         "minibatch training is supported, there will be multiple updates per epoch (see section"
                         "on Minibatch Training later).")
parser.add_argument("-learnrate", metavar="LEARNRATE", type=float, required=True,
                    help="The step size to use for training.")
parser.add_argument("-mb",  metavar="MINIBATCH_SIZE", type=int, default=0,
                    help="If minibatching is implemented, this specifies the number"
                         "of data points to be included in each minibatch. Set this value to 0 to do full batch"
                         "training when minibatching is supported.")


args = parser.parse_args()

print("args.train_feat \t type: %s \t value: %s" % (type(args.train_feat), args.train_feat))
print("args.learnrate \t type: %s \t value: %s" % (type(args.learnrate), args.learnrate))
print("args.epochs \t type: %s \t value: %s" % (type(args.epochs), args.epochs))
print("args.mb \t type: %s \t value: %s" % (type(args.mb), args.mb))
