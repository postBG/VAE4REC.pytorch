import torch
import torch.optim as optim
import wandb as wandb
from tensorboardX import SummaryWriter

import losses
import models
from data import create_datasets
from options import args
from trainer import Trainer


def main(parsed_args):
    torch.manual_seed(parsed_args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    wandb.init(config=parsed_args, name=parsed_args.exp, project="vae4rec")

    device = torch.device("cuda" if args.cuda else "cpu")
    datasets, n_items = create_datasets(parsed_args)

    p_dims = [200, 600, n_items]
    model = models.MultiVAE(p_dims).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=args.wd)
    criterion = losses.vae_loss

    writer = SummaryWriter()
    trainer = Trainer(model, datasets, optimizer, criterion, writer, args, device)
    trainer.train()


if __name__ == '__main__':
    main(args)
