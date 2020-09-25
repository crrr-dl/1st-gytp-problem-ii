import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import os, glob, tqdm, argparse
import numpy as np
from apex import amp

from data_loader import get_train_dataloader, get_test_dataloader
from model import VoxResNet
from losses import CombinedLoss
from optimizers import Ranger
from solver import Solver


def parse_args():

    parser = argparse.ArgumentParser(description = 'Validating on the CIFAR-10 dataset')
    
    parser.add_argument('--fold', help = 'Which fold to validate the model on', required = True, type = int)
    parser.add_argument('--batch-size', help = 'Batch size', default = 1, type = int)
    parser.add_argument('--model', help = 'Log directory to load model from or path to checkpoint', required = True, type = str)
    parser.add_argument('--log', help = 'Directory to save logs to', default = 'logs', type = str)
    parser.add_argument('--gpu', help = 'Specify which GPU to use (separate with commas). -1 means to use CPU', default = None, type = str)
    parser.add_argument('--amp', help = 'Whether to use mixed precision training', default = None, type = str)
    
    return parser.parse_args()


if __name__ == '__main__':

    start_epoch = 0

    args = parse_args()
    
    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'


    print('==> Prepping data...')
    test_dataloader = get_test_dataloader(args.fold, args.batch_size)

    print('==> Building CNN...')
    model = VoxResNet(1, 4)
    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        cudnn.benchmark = True

    model.to(device)

    optimizer = Ranger(model.parameters())

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level = args.amp)


    if os.path.isdir(args.model):
        checkpoint = torch.load(
            sorted(glob.glob(os.path.join(args.model, 'checkpoint*.pkl'))).pop(),
            map_location = lambda _, __: _
        )
        print(f'Model trained for {checkpoint["epoch"] + 1} epoch(s)...')
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.amp:
            amp.load_state_dict(checkpoint['amp'])
    
    elif os.path.isdir(args.model):
        checkpoint = torch.load(
            sorted(glob.glob(os.path.join(args.log, args.model, 'checkpoint*.pkl'))).pop(),
            map_location = lambda _, __: _
        )
        print(f'Model trained for {checkpoint["epoch"] + 1} epoch(s)...')
        start_epoch = checkpoint["epoch"] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.amp:
            amp.load_state_dict(checkpoint['amp'])
    
    else:
        raise FileNotFoundError(f'Checkpoint not found at {args.model}!')  
        
    print(f'Number of total params: {sum([np.prod(p.shape) for p in model.parameters()])}')

    if not os.path.exists(args.log):
        os.makedirs(args.log)
    index = int(sorted(os.listdir(args.log), key = lambda x: x[:4]).pop()[:4]) + 1
    args.log = os.path.join(args.log, '%.4d-val' % index)
    os.makedirs(args.log)

    print(f'==> Saving logs to {args.log}')

    solver = Solver(
        model = model, optimizer = optimizer, criterion = CombinedLoss(),
        start_epoch = None, num_epochs = None, device = device, 
        log_dir = args.log,  checkpoint_interval = None, amp = amp if args.amp else None
    )
    print(f'==> Validating...')
    solver.validate(test_dataloader)
    