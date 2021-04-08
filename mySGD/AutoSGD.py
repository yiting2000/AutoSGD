
'''Implement AutoSGD, significant code segments modified from pytorch.optim source file, 
https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
'''


import torch
from torch.optim.optimizer import Optimizer, required

class AutoSGD(Optimizer):
    
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(MySGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AutoSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
            Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
            """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                        
                   
                # get the norm for the next two steps
                step_size = group['lr']
                internal1 =  p.add_(d_p, alpha=-group['lr'])
                internal2 =  p.add_(d_p, alpha=-group['lr']*2)
                g1 = internal1.grad
                g2 = internal2.grad
                                       
                # use the norm to determine the learning rate of the next step 
                if torch.norm(g1)/torch.norm(g2) < 1/2:  #torch.max(g1[0],g2[0]) == g1[0]:
                    step_size = step_size
                else:
                    step_size = step_size * 2


                p.add_(d_p, alpha=-step_size)
                p.add_(d_p, alpha=-group['lr'])



        return loss

    
    
    


