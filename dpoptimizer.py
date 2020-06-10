import math
import torch

import uuid
import torch
from torch.optim.optimizer import Optimizer, required
import scipy.stats as st
from scipy.special import gamma, factorial,gammainc,gammaincc,gammainccinv
import numpy as np 
import torch.nn.functional as F

from torchsearchsorted import searchsorted


def ttest(a,b):
    n = a.shape[0]
    df = float(n - 1)

    d = (a - b).float()
    v = d.std()
    dm = d.mean()
    denom = torch.sqrt(v / float(n))

    t = dm/denom
    
    return t



def ks2(data1,data2):
    
    n1 = data1.shape[1]
    n2 = data2.shape[1]

    data1 = data1.sort()[0]
    data2 = data2.sort()[0]
    data_all = torch.cat([data1,data2],dim=1)
    cdf1 = searchsorted(data1,data_all,side='right')/(1.0*n1)
    cdf2 = (searchsorted(data2,data_all,side='right'))/(1.0*n2)
    d = (cdf1-cdf2).abs().max()
    return d


def pearsonr(x, y, batch_first=True):
    r"""Computes Pearson Correlation Coefficient across rows.
    Pearson Correlation Coefficient (also known as Linear Correlation
    Coefficient or Pearson's :math:`\rho`) is computed as:
    .. math::
        \rho = \frac {E[(X-\mu_X)(Y-\mu_Y)]} {\sigma_X\sigma_Y}
    If inputs are matrices, then then we assume that we are given a
    mini-batch of sequences, and the correlation coefficient is
    computed for each sequence independently and returned as a vector. If
    `batch_fist` is `True`, then we assume that every row represents a
    sequence in the mini-batch, otherwise we assume that batch information
    is in the columns.
    Warning:
        We do not account for the multi-dimensional case. This function has
        been tested only for the 2D case, either in `batch_first==True` or in
        `batch_first==False` mode. In the multi-dimensional case,
        it is possible that the values returned will be meaningless.
    Args:
        x (torch.Tensor): input tensor
        y (torch.Tensor): target tensor
        batch_first (bool, optional): controls if batch dimension is first.
            Default: `True`
    Returns:
        torch.Tensor: correlation coefficient between `x` and `y`
    Note:
        :math:`\sigma_X` is computed using **PyTorch** builtin
        **Tensor.std()**, which by default uses Bessel correction:
        .. math::
            \sigma_X=\displaystyle\frac{1}{N-1}\sum_{i=1}^N({x_i}-\bar{x})^2
        We therefore account for this correction in the computation of the
        covariance by multiplying it with :math:`\frac{1}{N-1}`.
    Shape:
        - Input: :math:`(N, M)` for correlation between matrices,
          or :math:`(M)` for correlation between vectors
        - Target: :math:`(N, M)` or :math:`(M)`. Must be identical to input
        - Output: :math:`(N, 1)` for correlation between matrices,
          or :math:`(1)` for correlation between vectors
    Examples:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> input = torch.rand(3, 5)
        >>> target = torch.rand(3, 5)
        >>> output = pearsonr(input, target)
        >>> print('Pearson Correlation between input and target is {0}'.format(output[:, 0]))
        Pearson Correlation between input and target is tensor([ 0.2991, -0.8471,  0.9138])
    """  # noqa: E501
    assert x.shape == y.shape

    if batch_first:
        dim = -1
    else:
        dim = 0

    centered_x = x - x.mean(dim=dim, keepdim=True)
    centered_y = y - y.mean(dim=dim, keepdim=True)

    covariance = (centered_x * centered_y).sum(dim=dim, keepdim=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(dim=dim, keepdim=True)
    y_std = y.std(dim=dim, keepdim=True)

    corr = bessel_corrected_covariance / (x_std * y_std)

    return corr


    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
def hardfix(inp, tresh,fix_val = None):
    out = F.hardshrink(inp,lambd=tresh)
    out = F.hardtanh(out,min_val=-tresh-1e-6, max_val=tresh+1e-6)
    if fix_val: 
        out *= fix_val/tresh 
    return out



def count_parameters(parameters):
    return sum(p.numel() for p in parameters if p.requires_grad)


class JDPSGD(Optimizer):
    

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,C=1, batch_size=256,device=None,error_correction=False, noise_generator=None,quantizers = None, quant_multiplier = 1.5,num_seletion=None,distance_multiplier = 1.0 , distance_threshold = 0.0 ):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.params = list(params)
        
            
        super(JDPSGD, self).__init__(self.params, defaults)
        

        for p in self.params:
            if  hasattr(p,'myid'):
                del p.myid
        self.momentum = momentum
        self.batch_size = batch_size
        self.device = device
        self.C = C
        self.bigger_batch = {}
        self.current = {}

        self.bigger_batch_count = {}
        self.errors = []

        self.num_selection = 500000
        self.saver = 0
        self.rand_nullifier  = {}
        
        self.distance_multiplier = distance_multiplier
        self.distance_threshold = distance_threshold
 
        num_params = count_parameters(self.params)
        print ('num params' ,num_params)

        self.num_params = num_params
        self.grad_vec  = torch.FloatTensor(size=[num_params,]).to(device)
        #self.unprivate_grad_vec  = torch.stack([torch.FloatTensor(size=[num_params,]).to(device)]*2000)
        #self.unprivate_grad_vec.normal_(0,0.11)
        
        self.unprivate_grad  = torch.FloatTensor(size=[num_params,]).to(device)

        
        self.noise_generator= noise_generator
        self.quant_multiplier = quant_multiplier

        self.mean = torch.FloatTensor([0]).to(device)
        self.std = torch.FloatTensor([1]).to(device)
        self.lap = torch.distributions.laplace.Laplace(self.mean[0],self.std[0])
        self.err_crrct = error_correction

        self.quant = quantizers
        
#         self.quant = torch.FloatTensor(st.exponweib(1.5063832694895258,
#  0.7476324067345266,
#  0.007481742361398164,
#  0.0036885972380380364).rvs((1000,200000))).to(device)
#         self.quant.normal_()
#         self.quant =  torch.stack([q.abs().sort(descending=True)[0][:50000] for q in self.quant]).to(torch.device("cuda" ,0))
#         self.quant = (self.quant/self.quant.norm(dim=1).reshape((-1,1))) # * args.quantclip
#         print (self.quant.shape,self.quant.norm(dim=1).max())

      
    def __setstate__(self, state):
        
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, sorted_grads = None):
        
        if self.noise_generator == None:
            raise ValueError("No noise model given")
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        
        loss = None
        if closure is not None:
            loss = closure()

        
        zeros = 0
        
        updated =0 
        al = 0 
        self.grad_vec.zero_()
        last_id = 0 
        for group in self.param_groups:
        
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                
                
                self.grad_vec[last_id:last_id+grad.size().numel()]= grad.view(-1,)
                
                last_id = last_id+grad.size().numel()
        norm = 0
        if self.quant != None:
            num_selection  =self.quant.shape[1]
        elif self.num_selection !=None:
            num_selection = self.num_selection
        else:
            num_selection  = self.num_params

        
        if sorted_grads is None:

            a = torch.argsort(self.grad_vec.abs(),descending=True)
            
        else:
            a = sorted_grads
        
        picked_grads = torch.zeros_like(self.grad_vec)
        
#         self.unprivate_grad_vec[self.saver%2000]= self.grad_vec.clone()
#         self.saver += 1
#         self.saver %= 2000
        
        norm = self.grad_vec[a[:num_selection]].norm()    
        
        self.grad_vec[a[:num_selection]] = (self.grad_vec[a[:num_selection]] / torch.max(norm,torch.ones_like(norm)*self.C))*self.C
        
        if self.quant != None:
            self.grad_vec[a[:num_selection]]  *= self.quant_multiplier
            mid = torch.argmax( F.cosine_similarity(self.quant, self.grad_vec[a[:num_selection]].abs().reshape((1,-1)) ) )

            picked_grads[a[:num_selection] ] = torch.max (torch.min((self.grad_vec[a[:num_selection]]),self.quant[mid]),-self.quant[mid])
        
        else: 
            picked_grads[a[:num_selection] ]  = self.grad_vec[a[:num_selection] ] 

        last_id = 0 

        #print (num_selection,picked_grads.norm())
        
        for group in self.param_groups:
        
            for p in group['params']:
                if p.grad is None:
                    continue      
                grad = p.grad.data
                
                if not hasattr(p,'myid'):
                    p.myid = uuid.uuid4()
                    self.bigger_batch[p.myid] = torch.zeros_like(grad)
                    self.rand_nullifier [p.myid]= torch.zeros_like(grad)
                    self.rand_nullifier [p.myid].uniform_()

                    
                    self.bigger_batch_count[p.myid] =  torch.cuda.LongTensor(size=[1]).zero_() if self.bigger_batch[p.myid].is_cuda else  torch.LongTensor(size=[1]).zero_()
                
                
                
                self.bigger_batch [p.myid].add_(picked_grads[last_id:last_id+grad.size().numel()].view(grad.size()))
                self.bigger_batch_count[p.myid]+=1
                last_id = last_id+grad.size().numel()
        
        


         

        if   self.bigger_batch_count[p.myid] == self.batch_size:
            #self.bool = not self.bool

            self.grad_vec.zero_()
            last_id = 0 
            
            student = torch.distributions.StudentT(10,loc=self.mean[0],scale=0.003)
            correct_1 = 0 
            correct_2 = 0 
            correct_3 = 0 
            
            for group in self.param_groups:
            
                for p in group['params']:
                    
                    

                    
                    base =  self.bigger_batch[p.myid]
                    
                    
                    
                    
                    my_rand_g = torch.from_numpy(self.noise_generator.rvs(base.shape)).float().to(self.device) #lap.rsample(base.shape)
#                     my_rand_g = torch.zeros_like(base) #self.lap.rsample(base.shape) 
#                     my_rand_g.normal_(0,0.9* 0.9)
                    #print(base.size(),my_rand.size())
#                     my_rand_lap.add_(base)
                    my_rand_g.add_(base) 
                    #base = base/float(self.batch_size) 
                    d_p = my_rand_g/float(self.batch_size) 
                    
#                     if self.thresh < 0 :
#                         d_p = my_rand_g/float(self.batch_size) 
#                     else:
#                         d_p = torch.zeros_like(base)
#                         my_rand_g= my_rand_g/float(self.batch_size) 
#                         #print(my_rand_g)
#                         for i in [0.0001,0.001,0.005,0.01,0.02,0.05,0.1,0.2]:
#                             d_p [my_rand_g >(i)]  = i
#                             d_p [my_rand_g < (-i)]  = -i
                        
                        
# #                         d_p_2 = torch.zeros_like(base)
# #                         d_p_2 [my_rand_g >((self.thresh))]  = 0.01
# #                         d_p_2 [my_rand_g < (-(self.thresh))]  = -0.01
                        
#                         d_p_3 = torch.zeros_like(base)
#                         d_p_3 [base >(1.0)]  = 0.01
#                         d_p_3 [base < (-1.0)]  = -0.01
                        
                        
                        
#                         correct_1+=(d_p== d_p_3).sum()
#                         correct_2+=( d_p_3 !=  0).sum()
#                         correct_3 += ( d_p !=  0).sum()
                    self.bigger_batch[p.myid] =d_p
                    self.grad_vec[last_id:last_id+d_p.size().numel()]= d_p.view(-1,)
                    self.unprivate_grad[last_id:last_id+d_p.size().numel()]= base.view(-1,)/float(self.batch_size) 
                    if self.saver<0 :#or (self.saver>4000 and self.saver < 4500):
                        self.unprivate_grad_vec[self.saver%500,last_id:last_id+d_p.size().numel()]= base.view(-1,)/float(self.batch_size) 

            
                    last_id = last_id+d_p.size().numel()
            
#             print(correct_1)
#             print(correct_2)
#             print(correct_3)

            picked_grads = torch.zeros_like(self.grad_vec)
                
          
            picked_grads = self.grad_vec
            
            
            if self.err_crrct :
                
                lap  = torch.distributions.laplace.Laplace(self.mean[0], 0.00014431920863030293)
                

                just_noise = torch.from_numpy(self.noise_generator.rvs(picked_grads.shape)).float().to(self.device).sort(descending=True)[0]/float(self.batch_size) 
                
        
                tt =  ttest(self.unprivate_grad.reshape((-1,)), picked_grads.reshape((-1,)))
                cosine = F.cosine_similarity(self.unprivate_grad.reshape((1,-1)), picked_grads.reshape((1,-1)))
                cosine_k = 1.02-(1+ F.cosine_similarity(just_noise.reshape((1,-1)), picked_grads.reshape((1,-1))))
                ks =ks2(just_noise.reshape((1,-1)),picked_grads.reshape((1,-1)))
                error = {'before':[(picked_grads-self.unprivate_grad).abs().mean(),(picked_grads-self.unprivate_grad).norm(),F.cosine_similarity(self.unprivate_grad.reshape((1,-1)), picked_grads.reshape((1,-1))),pearsonr(self.unprivate_grad, picked_grads),'not private l2',self.unprivate_grad.norm(),'mse to noise',(picked_grads-just_noise).norm(),'cosine',F.cosine_similarity(just_noise.reshape((1,-1)), picked_grads.reshape((1,-1))),'ks',ks2(just_noise.reshape((1,-1)),picked_grads.reshape((1,-1))),'ttest',tt]}
                
                
                a = torch.argsort(picked_grads,descending=True)
                picked_grads_vals = lap.rsample(a.shape) 
                
                picked_grads_corrected = picked_grads.clone().detach() #torch.zeros_like(picked_grads)
                #picked_grads_corrected[a] = picked_grads_vals.sort(descending=True)[0]
                
                start = 0 
                sample_tresh = 1000 
                step_size = min(100000,self.num_params)
                steps =   self.num_params//step_size
                #print(steps,self.num_params)
                for i in range(steps):
                    step = step_size
                    if i == steps-1:
                        step = self.num_params - (i*step_size)

                    #print ('step is',step)
#                 for group in self.param_groups:
#                     p  = 0 
#                     while p < (len(group['params'])):

#                         step = 0 
#                         while True:
#                             step += group['params'][p].shape.numel()
#                             p += 1
#                             if p >= (len(group['params'])): 
#                                 break
#                             if group['params'][p].shape.numel() > sample_tresh and step> sample_tresh:
#                                 break


                        #distance = (ks2(just_noise.reshape((1,-1)),picked_grads[start:start+ p.grad.data.numel()].reshape((1,-1))) )/0.003 CIFAR
                    #cosine = F.cosine_similarity(self.unprivate_grad[start:start+ step].reshape((1,-1)), picked_grads[start:start+ step].reshape((1,-1))) 
                    distance =ks *self.distance_multiplier #0.6 + (ks2(just_noise.reshape((1,-1)),picked_grads[start:start+ step].reshape((1,-1))))/0.01
                    
                     
                    #print (start,step,distance) 
                    picked_grads [start:step+start] =  picked_grads_corrected [start:step+start] * (min(distance ,5.0)) 
                    start = start+ step
                    if distance < self.distance_threshold:
                        picked_grads.zero_()
                        #print ('zeroed')
                
                assert start == self.num_params
                
                #print ('cosine_k',cosine_k,ks)
                
                        #torch.zeros_like(picked_grads)
                #print(self.grad_vec.mean(),self.grad_vec.std())
                #print (picked_grads.std(),picked_grads.mean())
                #picked_grads_vals.normal_(0,0.0002)
                #print ('actual dist',cosine,'estimated',distance)
                         
                #print('after EC',(picked_grads-self.unprivate_grad).abs().mean(),(picked_grads-self.unprivate_grad).norm(),F.cosine_similarity(self.unprivate_grad.reshape((1,-1)), picked_grads.reshape((1,-1))),pearsonr(self.unprivate_grad, picked_grads))
                error['after']=[(picked_grads-self.unprivate_grad).abs().mean(),(picked_grads-self.unprivate_grad).norm(),F.cosine_similarity(self.unprivate_grad.reshape((1,-1)), picked_grads.reshape((1,-1))),pearsonr(self.unprivate_grad, picked_grads)]
                self.errors.append(error)
#                 start= start + step_size




                #print('after EC',(picked_grads-self.unprivate_grad).abs().mean(),(picked_grads-self.unprivate_grad).norm(),F.cosine_similarity(self.unprivate_grad.reshape((1,-1)), picked_grads.reshape((1,-1))),pearsonr(self.unprivate_grad, picked_grads))
                    
#                     picked_grads[a2] = picked_grads_vals[half:].sort(descending=True)[0]
                #raise
                
            
           
                #print (picked_grads)
                   
                                    
            


            updated =0 
            al =0
            
            last_id = 0
            

            for group in self.param_groups:
                weight_decay = group['weight_decay']
                momentum = group['momentum']
                dampening = group['dampening']
                nesterov = group['nesterov']

                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    d_p =  picked_grads[last_id:last_id+self.bigger_batch[p.myid].size().numel()].view(self.bigger_batch[p.myid].size())
                    #print (d_p )
                    
                    updated+=float((d_p!=0).sum())
                    al += float(d_p.size().numel())
                    
                    
                    if weight_decay != 0:
                        d_p.add_(weight_decay, p.data)
                        
                    if momentum != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                            res = buf 
                        else:
                            buf = param_state['momentum_buffer']
                            res = buf 
                            if distance >  1.0:

                                res = buf.clone().mul(momentum).add(d_p, alpha=1 - dampening)

                            if distance > 1.5:
                                #print ("distance,", distance)
                                param_state['momentum_buffer'] = res
                        if nesterov:
                            d_p = d_p.add(res, alpha=momentum)
                        else:
                            d_p = res
                    
                    #self.updates.setdefault(p.myid,[]).append(d_p.clone().detach())
                    p.data.add_(-group['lr'], d_p)
                    
                    self.bigger_batch[p.myid].zero_()
                    self.bigger_batch_count[p.myid].zero_()
                    self.rand_nullifier [p.myid].uniform_()
                    last_id = last_id+self.bigger_batch[p.myid].size().numel()
            
            #print (self.tracker,self.tracker/float(self.batch_size),updated,self.ind)
            self.tracker=0
        return loss

    
#     