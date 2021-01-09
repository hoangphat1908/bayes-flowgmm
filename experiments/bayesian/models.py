import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from oil.model_trainers.classifier import Classifier,Trainer
from oil.utils.losses import softmax_mse_loss, softmax_mse_loss_both
from oil.utils.utils import Eval, izip, icycle,imap, export, FixedPytorchSeed
from oil.utils.mytqdm import tqdm
#from .schedules import sigmoidConsRamp
import flow_ssl
import experiments.train_flows.utils as utils
from flow_ssl import FlowLoss
from flow_ssl.realnvp import RealNVPTabular
from flow_ssl.distributions import SSLGaussMixture
from swag.posteriors import SWAG

from scipy.spatial.distance import cdist
from torchcontrib.optim import SWA
from functools import partial

@export
def RealNVPTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=.8,cov_std=1.,nperlayer=1,acc=0.9):
    #print(f'Instantiating means with dimension {dim_in}.')
    device = torch.device('cuda')
    inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
    model = RealNVPTabular(num_coupling_layers=coupling_layers,in_dim=dim_in,hidden_dim=k,num_layers=1,dropout=True)#*np.sqrt(1000/dim_in)/3
    #dist_scaling = np.sqrt(-8*np.log(1-acc))#np.sqrt(4*np.log(20)/dim_in)#np.sqrt(1000/dim_in)
    if num_classes ==2:
        means = utils.get_means('random',r=means_r,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        #means = torch.zeros(2,dim_in,device=device)
        #means[0,1] = 3.75
        dist = 2*(means[0]**2).sum().sqrt()
        means[0] *= 7.5/dist
        means[1] = -means[0]
        # means[0] /= means[0].norm()
        # means[0] *= dist_scaling/2
        # means[1] = - means[0]
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    else:
        means = utils.get_means('random',r=means_r*.7,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
        model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
        means_np = means.cpu().numpy()
    print("Pairwise dists:", cdist(means_np, means_np))
    return model

def RealNVPTabularSWAG(dim_in,coupling_layers,k,nperlayer=1,subspace='covariance', max_num_models=10):
    swag_model = SWAG(RealNVPTabular,
                      subspace_type=subspace, subspace_kwargs={'max_rank': max_num_models}, 
                      num_coupling_layers=coupling_layers,in_dim=dim_in, hidden_dim=k,num_layers=1,dropout=True)
    return swag_model

# def ResidualTabularWPrior(num_classes,dim_in,coupling_layers,k,means_r=1.,cov_std=1.,nperlayer=1,acc=0.9):
#     #print(f'Instantiating means with dimension {dim_in}.')
#     device = torch.device('cuda')
#     inv_cov_std = torch.ones((num_classes,), device=device) / cov_std
#     model = TabularResidualFlow(in_dim=dim_in,hidden_dim=k,num_per_block=coupling_layers)#*np.sqrt(1000/dim_in)/3
#     dist_scaling = np.sqrt(-8*np.log(1-acc))
#     means = utils.get_means('random',r=means_r*dist_scaling,num_means=num_classes, trainloader=None,shape=(dim_in),device=device)
#     means[0] /= means[0].norm()
#     means[0] *= dist_scaling/2
#     means[1] = - means[0]
#     model.prior = SSLGaussMixture(means, inv_cov_std,device=device)
#     means_np = means.cpu().numpy()
#     #print("Pairwise dists:", cdist(means_np, means_np))
#     return model




@export
class SemiFlow(Trainer):
    def __init__(self, *args, unlab_weight=1.,cons_weight=3., **kwargs):
        super().__init__(*args, **kwargs)
        self.hypers.update({'unlab_weight':unlab_weight,'cons_weight':cons_weight})
        self.dataloaders['train'] = izip(icycle(self.dataloaders['train']),self.dataloaders['_unlab'])

    def loss(self, minibatch):
        (x_lab, y_lab), x_unlab = minibatch
        a = float(self.hypers['unlab_weight'])
        b = float(self.hypers['cons_weight'])
        flow_loss = self.model.nll(x_lab,y_lab).mean() + a*self.model.nll(x_unlab).mean()
        # with torch.no_grad():
        #     unlab_label = self.model.prior.classify(self.model(x_unlab)).detach()
        # cons_loss = self.model.nll(x_unlab,unlab_label).mean()
        return flow_loss#+b*cons_loss
    def step(self, minibatch):
        self.optimizer.zero_grad()
        loss = self.loss(minibatch)
        loss.backward()
        utils.clip_grad_norm(self.optimizer, 100)
        self.optimizer.step()
        
        return loss
        
    def logStuff(self, step, minibatch=None):
        bpd_func = lambda mb: (self.model.nll(mb).mean().cpu().data.numpy()/mb.shape[-1] + np.log(256))/np.log(2)
        acc_func = lambda mb: self.model.prior.classify(self.model(mb[0])).type_as(mb[1]).eq(mb[1]).cpu().data.numpy().mean()
        metrics = {}
        with Eval(self.model), torch.no_grad():
            #metrics['Train_bpd'] = self.evalAverageMetrics(self.dataloaders['unlab'],bpd_func)
            metrics['val_bpd'] = self.evalAverageMetrics(imap(lambda z: z[0],self.dataloaders['val']),bpd_func)
            metrics['Train_Acc'] = self.evalAverageMetrics(self.dataloaders['Train'],acc_func)
            metrics['val_Acc'] = self.evalAverageMetrics(self.dataloaders['val'],acc_func)
            metrics['test_Acc'] = self.evalAverageMetrics(self.dataloaders['test'],acc_func)
            if minibatch:
                metrics['Unlab_loss(mb)']=self.model.nll(minibatch[1]).mean().cpu().data.numpy()
        self.logger.add_scalars('metrics',metrics,step)
        super().logStuff(step, minibatch)
        
    def getTestOutputs(self, loader='test'):
        logits_func = lambda mb: self.model.prior.class_logits(self.model(mb[0])).cpu().numpy()
        probs_func = lambda mb: self.model.prior.class_probs(self.model(mb[0])).cpu().numpy()
        preds_func = lambda mb: self.model.prior.classify(self.model(mb[0])).cpu().numpy()
        targets_func = lambda mb: mb[1].cpu().numpy()
                
        logits = self.getModelOutputs(self.dataloaders[loader], logits_func)
        probs = self.getModelOutputs(self.dataloaders[loader], probs_func)
        preds = self.getModelOutputs(self.dataloaders[loader], preds_func)
        targets = self.getModelOutputs(self.dataloaders[loader], targets_func)
        acc = (preds == targets).mean()
        return logits, probs, preds, targets, acc
        
    def evalAverageMetrics(self, loader, metrics):
        num_total, loss_totals = 0, 0
        with Eval(self.model), torch.no_grad():
            for minibatch in loader:
                try: mb_size = minibatch[0].shape[0]
                except AttributeError: mb_size=1
                loss_totals += mb_size*metrics(minibatch)
                num_total += mb_size
        if num_total==0: raise KeyError("dataloader is empty")
        return loss_totals/num_total
    
    def getModelOutputs(self, loader, output_func):
        output = []
        with Eval(self.model), torch.no_grad():
            for minibatch in loader:
                output.append(output_func(minibatch))
        return np.concatenate(output)
    
class SemiFlowSWAG(SemiFlow):
    def __init__(self, *args,swag_model=None, steps_per_epoch=24, num_epochs=100, 
                 swa_dec_pct=.5, swa_start_pct=.75, swa_freq_pct=.05, swa_lr_factor=.5,
                 **kwargs):
#         dataloaders['train'] = izip(icycle(dataloaders['train']),dataloaders['_unlab'])
#         def swa_wrapper(parameters, opt_constr):
#             optimizer = opt_constr(parameters)
#             return SWA(optimizer)
#         opt_constr=partial(swa_wrapper, opt_constr=opt_constr)
#         lr_sched = partial(lr_sched, steps_per_epoch=len(dataloaders['_unlab']))

#         lr_lambda = swa_learning_rate(num_epochs, swa_dec_pct, swa_start_pct, swa_lr_factor)
        self.swag_model = swag_model
        self.swa_start = int(swa_start_pct * steps_per_epoch * num_epochs)
        self.swa_freq = int(swa_freq_pct * steps_per_epoch * num_epochs)
        print("swa_start is {}, swa_freq is {}".format(self.swa_start, self.swa_freq))
#         print("swa_start is {}, swa_freq is {}".format(self.swa_start, self.swa_freq))
#         self.steps_per_epoch = steps_per_epoch
#         self.num_epochs = num_epochs
        
        super().__init__(*args, **kwargs)
                
        
    def train(self, num_epochs=100):
        """ The main training loop"""
        start_epoch = self.epoch
        steps_per_epoch = len(self.dataloaders['train']); step=0
        for self.epoch in tqdm(range(start_epoch+1, start_epoch + num_epochs+1),desc='train'):
            for i, minibatch in enumerate(self.dataloaders['train']):
                step = i + (self.epoch-1)*steps_per_epoch
                with self.logger as do_log:
                   if do_log: self.logStuff(step, minibatch)
                self.step(minibatch, step)
                [sched.step(step/steps_per_epoch) for sched in self.lr_schedulers]
        self.logStuff(step)
        
    
    def step(self, minibatch, step):   
        loss = super().step(minibatch)
        if (step + 1) > self.swa_start and (step + 1 - self.swa_start) % self.swa_freq == 0:
            print("UPDATE SWAG")
            self.swag_model.collect_model(self.model)
        return loss
    
    def getTestOutputs(self, loader='test'):
        logits_func = lambda mb: self.model.prior.class_logits(self.swag_model(mb[0])).cpu().numpy()
        probs_func = lambda mb: self.model.prior.class_probs(self.swag_model(mb[0])).cpu().numpy()
        preds_func = lambda mb: self.model.prior.classify(self.swag_model(mb[0])).cpu().numpy()
        targets_func = lambda mb: mb[1].cpu().numpy()
                
        logits = self.getModelOutputs(self.dataloaders[loader], logits_func)
        probs = self.getModelOutputs(self.dataloaders[loader], probs_func)
        preds = self.getModelOutputs(self.dataloaders[loader], preds_func)
        targets = self.getModelOutputs(self.dataloaders[loader], targets_func)
        acc = (preds == targets).mean()
        return logits, probs, preds, targets, acc
    
    def getModelOutputs(self, loader, output_func):
        output = []
        with Eval(self.swag_model), torch.no_grad():
            for minibatch in loader:
                output.append(output_func(minibatch))
        return np.concatenate(output)
    

# from oil.tuning.study import Study, train_trial
# import collections
# import os
# import copy
# #from train_semisup_text_baselines import makeTabularTrainer
# #from flowgmm_tabular_new import tabularTrial
# from flow_ssl.data.nlp_datasets import AG_News
# from flow_ssl.data import GAS, HEPMASS, MINIBOONE

# # if __name__=="__main__":
# #     trial(uci_hepmass_flowgmm_cfg)
#     # thestudy = Study(trial,uci_hepmass_flowgmm_cfg,study_name='uci_flowgmm_hypers222_m__m_m')
#     # thestudy.run(1,ordered=False)
#     # covars = thestudy.covariates()
#     # covars['test_Acc'] = thestudy.outcomes['test_Acc'].values
#     # covars['dev_Acc'] = thestudy.outcomes['dev_Acc'].values
#     #print(covars.drop(['log_suffix','saved_at'],axis=1))
#     # print(thestudy.covariates())
#     # print(thestudy.outcomes)

