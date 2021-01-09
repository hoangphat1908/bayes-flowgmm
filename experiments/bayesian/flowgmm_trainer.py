import torch, torchvision
import torch.nn.functional as F
from torch.optim import SGD,Adam,AdamW
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA
from torch import optim

from oil.datasetup.datasets import CIFAR10, split_dataset
from oil.utils.utils import Eval, LoaderTo, cosLr, dmap, FixedNumpySeed, FixedPytorchSeed

from flow_ssl.data.nlp_datasets import AG_News,YAHOO

from utils import swa_learning_rate
from models import RealNVPTabularWPrior, RealNVPTabularSWAG, SemiFlow, SemiFlowSWAG

import numpy as np
import os
import pandas as pd
from functools import partial

import warnings

def make_trainer(train_data, test_data, bs=5000, split={'train':200,'val':5000},
                network=RealNVPTabularWPrior, net_config={}, num_epochs=15,
                optim=AdamW, lr=1e-3, opt_config={'weight_decay':1e-5},
                swag=False, swa_config={'swa_dec_pct':.5, 'swa_start_pct':.75, 'swa_freq_pct':.05, 'swa_lr_factor':.1},
                swag_config={'subspace':'covariance', 'max_num_models':20},
#                 subspace='covariance', max_num_models=20,
                trainer=SemiFlow, trainer_config={'log_dir':os.path.expanduser('~/tb-experiments/UCI/'),'log_args':{'minPeriod':.1, 'timeFrac':3/10}},
                dev='cuda', save=False):
    with FixedNumpySeed(0):
        datasets = split_dataset(train_data,splits=split)
        datasets['_unlab'] = dmap(lambda mb: mb[0],train_data)
        datasets['test'] = test_data
        
    device = torch.device(dev)
    
    dataloaders = {k : LoaderTo(DataLoader(v,
                                         batch_size=min(bs,len(datasets[k])),
                                         shuffle=(k=='train'),
                                         num_workers=0,
                                         pin_memory=False),
                              device) 
                   for k, v in datasets.items()}
    dataloaders['Train'] = dataloaders['train']
    
#     model = network(num_classes=train_data.num_classes, dim_in=train_data.dim, **net_config).to(device)
#     swag_model = SWAG(model_cfg.base, 
#                     subspace_type=args.subspace, subspace_kwargs={'max_rank': args.max_num_models},
#                     *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    
#     swag_model.to(args.device)
    opt_constr = partial(optim, lr=lr, **opt_config)
    model = network(num_classes=train_data.num_classes, dim_in=train_data.dim, **net_config).to(device)
    if swag:
        swag_model = RealNVPTabularSWAG(dim_in=train_data.dim, **net_config, **swag_config)
#         swag_model = SWAG(RealNVPTabular,
#                           subspace_type=subspace, subspace_kwargs={'max_rank': max_num_models}, 
#                           num_classes=train_data.num_classes, dim_in=train_data.dim, 
#                           num_coupling_layers=coupling_layers,in_dim=dim_in,**net_config)
#         swag_model.to(device)
#         swag_model = SWAG(RealNVPTabular, num_classes=train_data.num_classes, dim_in=train_data.dim, 
#                         swag=True, **swag_config, **net_config)
#         model.to(device)
        swag_model.to(device)
        swa_config['steps_per_epoch'] = len(dataloaders['_unlab'])
        swa_config['num_epochs'] = num_epochs
        lr_sched = swa_learning_rate(**swa_config)
#         lr_sched = cosLr(num_epochs)
        return trainer(model,dataloaders, opt_constr,lr_sched, swag_model=swag_model, **swa_config, **trainer_config)
    else:
#         model = network(num_classes=train_data.num_classes, dim_in=train_data.dim, **net_config).to(device)
        lr_sched = cosLr(num_epochs)
    #     lr_sched = lambda e:1
        return trainer(model,dataloaders, opt_constr,lr_sched, **trainer_config)