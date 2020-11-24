import math
import torch
from torch.utils.data import Dataset
from .nn_modules import *
    
def accuracy(logits, targets):
    '''
    :param logits: (batch_size, class_num)
    :param targets: (batch_size, class_num)
    :return: 
    '''
    pred = logits.data > 0.5
    true = targets.data > 0.5
    return (true == pred).sum().item()/(pred.shape[1] * pred.shape[0])

class AverageMeter():
    '''
    batch average acc
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0. #value
        self.avg = 0. #average
        self.sum = 0. #sum
        self.count = 0 #count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n 
        self.count += n 
        self.avg = self.sum / self.count 

def clip_gradient(optimizer, grad_clip):
    """
    Gradient clip
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                # inplace, no new tensors created
                # gradient cliped to (-grad_clip, grad_clip)
                param.grad.data.clamp_(-grad_clip, grad_clip)
                
from torch.utils.data import Dataset
class dataloader(Dataset):
    '''
    dataloader
    '''
    
    def __init__(self, X, X_length, y1, y2, y3):
        '''
        :param output_folder: 
        :param split: 'train', 'dev', or 'test'
        '''
#         self.split = split
#         assert self.split in {'train', 'dev', 'test'}
        
        self.dataset = X
        self.length = X_length
        self.label_1 = y1
        self.label_2 = y2
        self.label_3 = y3

        self.dataset_size = self.dataset.shape[0]
        
    def __getitem__(self, i):

        sentence = self.dataset[i] # sentence shape [max_len]
        sentence_length = self.length[i]
        sentence_label_1 = self.label_1[i]
        sentence_label_2 = self.label_2[i]
        sentence_label_3 = self.label_3[i]
        
        return sentence, sentence_length, sentence_label_1, sentence_label_2, sentence_label_3

    def __len__(self):
        
        return self.dataset_size
    
def train(train_loader, model, criterion, optimizer, epoch, print_freq, device, grad_clip=None):
    '''
    Train one epoch
    '''
    # enable dropout
    model.train()
    
    losses = AverageMeter()  # average loss for a batch
    accs = AverageMeter()  # average acc for a batch
    
    for i, (seqs, seqs_len, labels_1, labels_2, labels_3) in enumerate(train_loader):
        index = torch.from_numpy(np.argsort(seqs_len.data.numpy())[::-1].copy())
        seqs = seqs[index]
        seqs_len = seqs_len[index]
        labels_1 = labels_1[index]
        labels_2 = labels_2[index]
        labels_3 = labels_3[index]
        # move to CPU/GPU
        seqs = seqs.to(device)
        #seqs_len = seqs_len.to(device)
        
        labels_1 = labels_1.to(device)
        labels_2 = labels_2.to(device)
        labels_3 = labels_3.to(device)
        
        # forward
        logits_1, logits_2, logits_3 = model(seqs, seqs_len)
        
        logits_1 = logits_1[labels_1.sum(dim = 1) >= 1]
        logits_2 = logits_2[labels_2.sum(dim = 1) >= 1]
        logits_3 = logits_3[labels_3.sum(dim = 1) >= 1]
        
        labels_1 = labels_1[labels_1.sum(dim = 1) >= 1]
        labels_2 = labels_2[labels_2.sum(dim = 1) >= 1]
        labels_3 = labels_3[labels_3.sum(dim = 1) >= 1]
        
        # loss
        if labels_1.size()[0] == 0:
            loss_1 = 0
            acc1 = 0
        else:
            loss_1 = criterion(logits_1, labels_1)
            acc1 = accuracy(logits_1, labels_1)
        if labels_2.size()[0] == 0:
            loss_2 = 0
            acc2 = 0
        else:
            loss_2 = criterion(logits_2, labels_2)
            acc2 = accuracy(logits_2, labels_2)
        if labels_3.size()[0] == 0:   
            loss_3 = 0
            acc3 = 0
        else:
            loss_3 = criterion(logits_3, labels_3)
            acc3 = accuracy(logits_3, labels_3)
        loss =  (loss_1 + loss_2 + loss_3)/3
        acc = (acc1 + acc2 + acc3)/3
        # backprop
        optimizer.zero_grad()
        loss.backward()

        # grad_clip
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        
        # update optimizer
        optimizer.step()
        
        # update performance
        accs.update(acc)
        losses.update(loss.item())
        
        # print
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, i, len(train_loader),
                                                                          loss=losses,
                                                                          acc=accs))
            
def validate(val_loader, model, criterion, print_freq, device):
    '''
    validate on one epoch
    '''
    
    #disenble drop off
    model = model.eval()

    losses = AverageMeter()  
    accs = AverageMeter()  

    # no gradient calculation
    with torch.no_grad():
        for i, (seqs, seqs_len, labels_1, labels_2, labels_3) in enumerate(val_loader):
            index = torch.from_numpy(np.argsort(seqs_len.data.numpy())[::-1].copy())
            seqs = seqs[index]
            seqs_len = seqs_len[index]
            labels_1 = labels_1[index]
            labels_2 = labels_2[index]
            labels_3 = labels_3[index]
            # move to CPU/GPU
            seqs = seqs.to(device)
            # seqs_len = seqs_len.to(device)

            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_3 = labels_3.to(device)

            # forward
            logits_1, logits_2, logits_3 = model(seqs, seqs_len)

            logits_1 = logits_1[labels_1.sum(dim = 1) >= 1]
            logits_2 = logits_2[labels_2.sum(dim = 1) >= 1]
            logits_3 = logits_3[labels_3.sum(dim = 1) >= 1]

            labels_1 = labels_1[labels_1.sum(dim = 1) >= 1]
            labels_2 = labels_2[labels_2.sum(dim = 1) >= 1]
            labels_3 = labels_3[labels_3.sum(dim = 1) >= 1]
            # loss
            if labels_1.size()[0] == 0:
                loss_1 = 0
                acc1 = 0
            else:
                loss_1 = criterion(logits_1, labels_1)
                acc1 = accuracy(logits_1, labels_1)
            if labels_2.size()[0] == 0:
                loss_2 = 0
                acc2 = 0
            else:
                loss_2 = criterion(logits_2, labels_2)
                acc2 = accuracy(logits_2, labels_2)
            if labels_3.size()[0] == 0:   
                loss_3 = 0
                acc3 = 0
            else:
                loss_3 = criterion(logits_3, labels_3)
                acc3 = accuracy(logits_3, labels_3)

            loss =  (loss_1 + loss_2 + loss_3)/3
            losses.update(loss.item())

            acc = (acc1 + acc2 + acc3)/3
            accs.update(acc)
            

            if i % print_freq  == 0:
                print('Validation: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})\t'.format(i, len(val_loader),
                                                                                loss=losses, acc=accs))
        # overall acc
        print('LOSS - {loss.avg:.3f}, ACCURACY - {acc.avg:.3f}\n'.format(loss=losses, acc=accs))

    return losses.avg


