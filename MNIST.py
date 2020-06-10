from __future__ import print_function

import argparse
import os
import shutil
import time
import random
import dpoptimizer
import pathlib
    
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets


import torch.optim as optim
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from torchvision import datasets, transforms
import threading
import scipy.stats as st


import torch
import torch.multiprocessing as mp



use_cuda= True



import argparse




parser = argparse.ArgumentParser(description='')
parser.add_argument('--noisemodel', type=str, default='johnson')
parser.add_argument('--noiseparams', type=float,  nargs='+')
parser.add_argument('--quantization', action='store_true')
parser.add_argument('--epochs',type=int,default=100)

parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--microbatch',type=int,default=4)
parser.add_argument('--batch',type=int,default=256)
parser.add_argument('--testbatch',type=int,default=1000)
parser.add_argument('--ngpus',type=int,default=1)
parser.add_argument('--nprocs',type=int,default=1)
parser.add_argument('--momentum',type=float,default=0)
parser.add_argument('--clip',type=float,default=1.2)


parser.add_argument('--errcrt', action='store_true')
parser.add_argument('--distancemultiplier',type=float,default=1.0/0.0013)
parser.add_argument('--distancethresh',type=float,default=0.7)


parser.add_argument('--quantclip',type=float,default=0.9)
parser.add_argument('--quantmultiplier',type=float,default=1.5)

parser.add_argument('--adaptivenoise',type=str,default='')



args = parser.parse_args()


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 8, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 1)
        self.fc1 = nn.Linear(3 * 3 * 32, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        y = x.view(-1, 3 * 3 * 32)
        x = F.relu(self.fc1(y))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


    
     

def sync_models(models):
    base = models[0]
    base_parameters = list(base.parameters())
    for model in models[1:]:
        for ind, p in enumerate(model.parameters()):
            if not p.requires_grad:
                continue
            p.data.copy_(base_parameters[ind].data)

    
            
            

def test( model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


    
def make_one_hot(labels, C=10):

    one_hot = torch.FloatTensor(labels.size()[0], C).zero_()
    one_hot=one_hot.to(labels.device)
    
    target = one_hot.scatter_(1, labels.data.view(-1,1), 1)
    
    target = torch.autograd.Variable(target)
        
    return target
        
def train_proc(rank,device_id, model,grad_holder,grad_argsorter,to_master_queue,job_queue,my_queue):
    print("I STARTED",rank)
    device = torch.device("cuda" ,device_id)
    model.train()
    trainloader_list = list(trainloader)
    dummy_out = torch.zeros_like(grad_holder)
    optimizer = optim.SGD(model.parameters(),lr=0.01)
    
    while True:
        indexes = job_queue.get()
        data = torch.cat([trainloader_list[i][0] for i in indexes])
        target = torch.cat([trainloader_list[i][1] for i in indexes])
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        out = model (data)
        #out_l = F.log_softmax(out, dim=1)

        loss = F.cross_entropy(out,target)
        loss.backward()
        
        grad_holder.zero_()
        last_id =  0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            grad = p.grad.data
            grad_holder[last_id:last_id+grad.size().numel()].add_(grad.view(-1,))
            last_id += grad.size().numel()
            
        assert  last_id == len(grad_holder)
        
        torch.sort(grad_holder.abs(),descending=True,out= (dummy_out,grad_argsorter))
        
        to_master_queue.put(rank)
        com = my_queue.get()
        
        if com == 'Q':
            break


            
            
##### LOAD DATA
train_batch = args.microbatch
test_batch = args.testbatch


transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
#      transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#    transforms.ToTensor(),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([

    transforms.ToTensor(),
    #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataloader = datasets.MNIST
num_classes = 10


trainset = dataloader(root='./data',  train=True, download=True, transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)

testset = dataloader(root='./data',  train =False, download=True, transform=transform_train)
testloader = data.DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=0
                            )



batch_per_step = int(args.batch/train_batch)
assert args.batch/train_batch<= batch_per_step

num_processes = args.nprocs

num_gpus = args.ngpus





def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = .2
    if epoch > 10:
        lr = .15
    if epoch > 20:
        lr = 0.1
    if epoch > 40:
        lr = 0.05
    if epoch > 60:
        lr = 0.01 
    
    #if epoch > 40:
    #    lr = 0.02
    
    #if epoch >  50:
    #     lr = 0.005
        
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def adjust_noise(optimizer,epoch,args,noise_dict):
    keys = list(noise_dict.keys())
    keys.sort()
    select = keys[0]
    for k in keys:
        select = k
        if k > epoch:
            break
    
    if args.noisemodel == 'johnson':
        noise_model = st.johnsonsu(*noise_dict[select])
    if args.noisemodel == 'gaussian':
        print ("GAUSSIAN NOISE set")
        noise_model = st.norm(*noise_dict[select])
    if args.noisemodel == 'laplace':
        noise_model = st.laplace(*noise_dict[select])

    if args.noisemodel == 'cauchy':
        noise_model = st.cauchy(*noise_dict[select])

    if args.noisemodel == 'gennorm':
        noise_model = st.gennorm(*noise_dict[select])

    if args.noisemodel == 'studentt':
        noise_model = st.t(*noise_dict[select])
        
    print ('noise model is',args.noisemodel,noise_dict[select])
    optimizer.noise_generator= noise_model

    

if __name__ == '__main__':
    quantizer = None
    import yaml        
    noise_dict = {args.epochs : args.noiseparams}
    if args.adaptivenoise != '':
        
        noise_dict = yaml.load(args.adaptivenoise,Loader=yaml.FullLoader)


    

    
        
    if  args.quantization:
        quantizer = torch.load('quantization')
        quantizer =  torch.stack([q.abs().sort(descending=True)[0] for q in quantizer]).to(torch.device("cuda" ,0))
        quantizer = quantizer[:,:14000]
        quantizer = (quantizer/quantizer.norm(dim=1).reshape((-1,1))) * args.quantclip
    if quantizer == None :
        print ("No quantization given using just clipping")
    print (args)
    mp.set_start_method('spawn')
    
    base_device_id = 0 
    all_devices  = [torch.device("cuda" ,i) for i in range(num_gpus)]
    
    device = torch.device("cuda" ,base_device_id)
    
    models = [Net().to(dv) for dv in all_devices]
    
    base_model  = models[0]
    optimizer = dpoptimizer.JDPSGD(base_model.parameters(),lr=0.2, C= args.clip, batch_size=batch_per_step,momentum=args.momentum,device=device,error_correction=args.errcrt,quantizers=quantizer,quant_multiplier=args.quantmultiplier,distance_multiplier = args.distancemultiplier , distance_threshold = args.distancethresh)                                
    
    
    
    grad_holders = []
    grad_argsorted = []
    import time
    for g in grad_holders:
        g.share_memory_()
    
    for g in grad_argsorted:
        g.share_memory_()
    
    
    whole_index = np.arange(len(trainset))
    np.random.shuffle(whole_index)
    
    job_queue = mp.Queue()
    to_master_queue = mp.Queue()
    
    
    
    process_queues = [] 
    processes= []
    for model in models:
        model.share_memory()
    
    for rank in range(num_processes):
        
        
        pr_queue = mp.Queue()
        dv_id  =(rank%(num_gpus))
        gr_holder = torch.FloatTensor(size=[count_parameters(base_model),]).to(all_devices[dv_id]) 
        gr_sort = torch.LongTensor(size=[count_parameters(base_model),]).to(all_devices[dv_id])
        
        grad_holders.append(gr_holder)
        grad_argsorted.append(gr_sort)
        p = mp.Process(target=train_proc, args=(rank,dv_id, models[dv_id],gr_holder,gr_sort,to_master_queue,job_queue,pr_queue ))
        # We first train the model across `num_processes` processes
        process_queues.append(pr_queue)
        p.start()
        
        processes.append(p)
      
    
    for epoch in range(args.epochs):
        adjust_noise(optimizer,epoch,args,noise_dict)
        steps = len(whole_index)// ( batch_per_step * train_batch)
        np.random.shuffle(whole_index)
        start_ind = 0 
        t1= time.time()
        adjust_learning_rate(optimizer,epoch)
        for  step in range(steps): 
            sync_models(models)
            np.random.shuffle(whole_index)
            start_ind = 0 
            for _ in range(batch_per_step):

                job_queue.put(whole_index[start_ind:start_ind+train_batch])
                start_ind += train_batch
            
            for _ in range(batch_per_step):
                who_did = to_master_queue.get()
                #optimizer.step_with_grad(grad_holders[who_did],grad_argsorted[who_did])
                last_id = 0 
                for p in base_model.parameters():
                    if not p.requires_grad:
                        continue
                    if p.grad is None:
                        p.grad = torch.zeros_like (p.data)
                    p.grad.copy_(grad_holders[who_did][last_id:last_id+p.size().numel()].view(p.size()))
                    last_id += p.grad .size().numel()
                assert  last_id  == grad_holders[who_did].size().numel()
                
                optimizer.step(sorted_grads = grad_argsorted[who_did])

                process_queues[who_did].put("G")
            
            if step %100 ==0 :
                print ("STEP: ",step)

        #torch.save(base_model.state_dict(),'saved_models/mnist_epoch_%d'%epoch)
        torch.save(optimizer.errors,'errors')
        print ('epoch: ',epoch, 'time to train: ', time.time()-t1)
        test( base_model, device, testloader) 
        
    
    for p in processes:
        p.terminate()
    for p in processes:
        p.join()
