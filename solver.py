import io
import os
import tqdm
import logging
import numpy as np

import torch
import nibabel as nib


def dice(y_hat, y, num_classes):

    y_hat = y_hat.argmax(1)

    dices = []
    for y_i, y_hat_i in zip(y, y_hat):
        dice = np.zeros(num_classes)
        for i in range(1, num_classes):
            m0 = (y_hat_i == i).int()
            m1 = (y_i == i).int()
            dice[i] = (2 * (m0 * m1).sum(dim = (0, 1, 2)).float() / ((m0 + m1).sum(dim = (0, 1, 2))).float()).cpu().numpy()
        dices.append(dice)
    
    return np.mean(dices)


class Solver:

    def __init__(self, model, optimizer, criterion, start_epoch, num_epochs, device, log_dir, checkpoint_interval, amp):
        
        self.model = model
        self.num_classes = 4
        self.optimizer = optimizer
        self.criterion = criterion
        
        self.start_epoch = start_epoch
        self.num_epochs = num_epochs

        self.device = device

        self.log_dir = log_dir
        self.checkpoint_interval = checkpoint_interval

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        self.amp = amp
        
        self.logger.addHandler(logging.StreamHandler())
        file_handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s")) 
        self.logger.addHandler(file_handler)


    def train(self, train_dataloader, test_dataloader):
        
        for i in range(self.start_epoch, self.num_epochs):

            self.logger.info(f'Epoch {i}:')

            self.model.train()
            bar = tqdm.tqdm(train_dataloader, ncols = 125)
            training_loss = []

            for x, y in bar:
                
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                if self.amp:
                    with self.amp.scale_loss(loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                self.optimizer.step()

                training_loss.append(loss.item())
                bar.set_description('Loss: %.3f' % np.mean(training_loss))

            if i % self.checkpoint_interval == 0:
                checkpoint = {
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }
                if self.amp:
                    checkpoint['amp'] = self.amp.state_dict()
                torch.save(checkpoint, os.path.join(self.log_dir, f'checkpoint{i:04d}.pkl'))
                del checkpoint

                self.model.eval()
                test_loss = []; test_dice = []
                with torch.no_grad():
                    for x, y in test_dataloader:
                    
                        x, y = x.to(self.device), y.to(self.device)
                        y_hat = self.model(x)
                        loss = self.criterion(y_hat, y)

                        test_loss.append(loss.item())
                        test_dice.append(dice(y_hat, y, self.num_classes))
                self.logger.info(f'Testing loss: {np.mean(test_loss):.3f} | Dice: {np.mean(test_dice) * 100:.3f}%')
                
        self.logger.info(f'Training finished...')
        

    def validate(self, test_dataloader):
        
        self.model.eval()
        test_loss = []; test_dice = []
        
        with torch.no_grad():     
            for i, (x, y) in enumerate(test_dataloader):            
                x, y = x.to(self.device).float(), y.to(self.device).long()
                y_hat = self.model(x)
                nib.Nifti1Image(y_hat[0].argmax(0).cpu().numpy().astype(np.uint8), np.eye(4)).to_filename(f'TEST_{i + 1}.nii.gz')
                
                loss = self.criterion(y_hat, y)

                test_loss.append(loss.item())
                test_dice.append(dice(y_hat, y, self.num_classes))

        self.logger.info(f'Testing loss: {np.mean(test_loss):.3f} | Dice: {np.mean(test_dice) * 100:.3f}%')
