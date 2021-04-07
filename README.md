# AutoSGD
The aim of this personal project is to create a SGD (stochastic descent) variant which automatically adjusts its learning rate. 

Many efforts and progress has been made on this topic. Broadly two types of approaches exist: one adds a schedule of some sorts to the SGD (examples are cyclic SGD and SGDR), some others, a more popular branch, uses past gradient information to adjusts future steps. 

We try to explore the possibility of using past gradient information on the learning rate alone, other than the overall gradient.

Significant amount of code modified from torch.optim https://pytorch.org/docs/stable/optim.html
pytorch/fairseq
https://github.com/pytorch/fairseq/tree/master/fairseq/optim
and 
adabound https://github.com/Luolc/AdaBound
