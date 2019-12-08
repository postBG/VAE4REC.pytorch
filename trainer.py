import time

import numpy as np
import torch
import wandb

import metric
from options import args
from utils import naive_sparse2tensor


def print_epoch_train_info(epoch, epoch_start_time, n100, r20, r50, val_loss):
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:4.2f}s | valid loss {:4.2f} | '
          'n100 {:5.3f} | r20 {:5.3f} | r50 {:5.3f}'.format(
        epoch, time.time() - epoch_start_time, val_loss,
        n100, r20, r50))
    print('-' * 89)


class Trainer(object):
    def __init__(self, model, datasets, optimizer, criterion, writer, args, device):
        self.model = model
        self.datasets = datasets
        self.optimizer = optimizer
        self.criterion = criterion

        self.batch_size = args.batch_size
        self.total_anneal_steps = args.total_anneal_steps
        self.anneal_cap = args.anneal_cap

        self.writer = writer
        self.args = args
        self.logging_interval = args.log_interval
        self.device = device

    def train(self):
        best_n100 = -np.inf
        n_iter = 0

        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            n_iter = self.train_one_epoch(self.model, self.optimizer, self.criterion,
                                          self.datasets['train_data'], n_iter, epoch)
            val_loss, n100, r20, r50 = self.evaluate(self.model, self.criterion, self.datasets['val_data_tr'],
                                                     self.datasets['val_data_te'], n_iter)
            print_epoch_train_info(epoch, epoch_start_time, n100, r20, r50, val_loss)

            self.writer.add_scalars('data/loss', {'valid': val_loss}, n_iter)
            self.writer.add_scalar('data/n100', n100, n_iter)
            self.writer.add_scalar('data/r20', r20, n_iter)
            self.writer.add_scalar('data/r50', r50, n_iter)
            wandb.log({
                'val/loss': val_loss,
                'val/n100': n100,
                'val/r20': r20,
                'val/r50': r50
            }, n_iter)

            # Save the model if the n100 is the best we've seen so far.
            if n100 > best_n100:
                with open(args.save, 'wb') as f:
                    torch.save(self.model, f)
                best_n100 = n100

    def train_one_epoch(self, model, optimizer, criterion, train_data, total_iters, epoch):
        # Turn on training mode
        model.train()
        train_loss = 0.0
        start_time = time.time()

        train_data_size = train_data.shape[0]
        idxlist = list(range(train_data_size))
        np.random.shuffle(idxlist)

        for batch_idx, start_idx in enumerate(range(0, train_data_size, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, train_data_size)
            data = train_data[idxlist[start_idx:end_idx]]
            data = naive_sparse2tensor(data).to(self.device)

            if args.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 1. * total_iters / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)

            loss = criterion(recon_batch, data, mu, logvar, anneal)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            total_iters += 1

            if batch_idx % self.logging_interval == 0 and batch_idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:4.2f} | '
                      'loss {:4.2f}'.format(
                    epoch, batch_idx, len(range(0, train_data_size, self.batch_size)),
                    elapsed * 1000 / self.logging_interval,
                    train_loss / self.logging_interval))

                # Log loss to tensorboard
                n_iter = (epoch - 1) * len(range(0, train_data_size, self.batch_size)) + batch_idx
                self.writer.add_scalars('data/loss', {'train': train_loss / self.logging_interval}, n_iter)
                wandb.log({
                    'train/loss': train_loss / self.logging_interval
                }, n_iter)

        return total_iters

    def evaluate(self, model, criterion, data_tr, data_te, n_iter):
        # Turn on evaluation mode
        model.eval()
        total_loss = 0.0

        data_tr_size = data_tr.shape[0]
        e_idxlist = list(range(data_tr_size))
        e_N = data_tr.shape[0]
        n100_list = []
        r20_list = []
        r50_list = []

        with torch.no_grad():
            for start_idx in range(0, e_N, self.batch_size):
                end_idx = min(start_idx + args.batch_size, data_tr_size)
                data = data_tr[e_idxlist[start_idx:end_idx]]
                heldout_data = data_te[e_idxlist[start_idx:end_idx]]

                data_tensor = naive_sparse2tensor(data).to(self.device)

                if self.total_anneal_steps > 0:
                    anneal = min(self.anneal_cap,
                                 1. * n_iter / self.total_anneal_steps)
                else:
                    anneal = self.anneal_cap

                recon_batch, mu, logvar = model(data_tensor)

                loss = criterion(recon_batch, data_tensor, mu, logvar, anneal)
                total_loss += loss.item()

                # Exclude examples from training set
                recon_batch = recon_batch.cpu().numpy()
                recon_batch[data.nonzero()] = -np.inf

                n100 = metric.NDCG_binary_at_k_batch(recon_batch, heldout_data, 100)
                r20 = metric.Recall_at_k_batch(recon_batch, heldout_data, 20)
                r50 = metric.Recall_at_k_batch(recon_batch, heldout_data, 50)

                n100_list.append(n100)
                r20_list.append(r20)
                r50_list.append(r50)

        total_loss /= len(range(0, e_N, self.batch_size))
        n100_list = np.concatenate(n100_list)
        r20_list = np.concatenate(r20_list)
        r50_list = np.concatenate(r50_list)

        return total_loss, np.mean(n100_list), np.mean(r20_list), np.mean(r50_list)
