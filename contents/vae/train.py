import argparse
import numpy as np
import pytorch_lightning as pl
import torch
from tqdm import tqdm
from methods import get_module
from dataset import Obj3DDataset
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, help="The neural network method to be tested")
    parser.add_argument("--sim", type=str, help="The simulation case")
    parser.add_argument("--nstates", type=int, help="The number of latent dimension states")
    parser.add_argument("--ncom", type=int, help="The number of coms")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    parser.add_argument("--max_epochs", type=int, default=500, help="Maximum number of epochs of training")
    parser.add_argument("--ndset", type=int, default=100, help="Number of simulations to be generated")
    parser.add_argument("--tmax", type=float, default=10.0, help="Max sim time for training")
    parser.add_argument("--dt", type=float, default=0.01, help="The dt to capture the speed")
    parser.add_argument("--version", type=int, default=None, help="The experiment version")
    parser.add_argument("--batch_size", type=int, default=256, help="The max size of a batch")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ndset = args.ndset
    ts = np.linspace(0, args.tmax, int(args.tmax * 10))
    max_epochs = args.max_epochs
    method = args.method
    sim = args.sim
    nstates = args.nstates
    ncom = args.ncom
    dt = args.dt
    batch_size = args.batch_size
    ts_list = [ts for _ in range(ndset)]
    train_dset = Obj3DDataset(mode="train")
    val_dset = Obj3DDataset(mode="val")
    # pdb.set_trace()
    # data_shape = [3, 2, 128, 128]
    data_shape = train_dset[0].shape
    data_shape = [data_shape[0], data_shape[2], data_shape[3]]
    module = get_module(method, data_shape=data_shape, ncom=ncom, nlatent_dim=nstates, dt=dt)
    # set up the pytorch lightning modules, dataloader, and do the training
    train_dloader = torch.utils.data.DataLoader(train_dset, shuffle=True, batch_size=batch_size)
    val_dloader = torch.utils.data.DataLoader(val_dset, shuffle=False, batch_size=batch_size)
    logger = pl.loggers.tensorboard.TensorBoardLogger("lightning_logs", name="", version=args.version)
    model_chkpt = pl.callbacks.ModelCheckpoint(monitor="val_loss")
    trainer = pl.Trainer(max_epochs=max_epochs, logger=logger, callbacks=[model_chkpt])
    trainer.fit(module, train_dloader, val_dloader)

if __name__ == "__main__":
    main()
