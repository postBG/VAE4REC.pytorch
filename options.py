import argparse

parser = argparse.ArgumentParser(description='PyTorch Variational Autoencoders for Collaborative Filtering')
parser.add_argument('--data', type=str, default='/data/sm/ml-1m', help='Movielens-20m dataset location')
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('--wd', type=float, default=0.00, help='weight decay coefficient')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='upper epoch limit')
parser.add_argument('--total_anneal_steps', type=int, default=200000,
                    help='the total number of gradient updates for annealing')
parser.add_argument('--anneal_cap', type=float, default=0.2, help='largest annealing parameter')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', default=True, help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--save', type=str, default='model.pt', help='path to save the final model')
args = parser.parse_args()
