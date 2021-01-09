from models import RealNVPTabularWPrior, RealNVPTabularSWAG, SemiFlow, SemiFlowSWAG
from ensembles import SWAG_Results, Ensembles
from flowgmm_trainer import make_trainer

from flow_ssl.data.nlp_datasets import AG_News,YAHOO

import argparse
import json
import os
import warnings
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Test accuracies between Deep Ensembles and Flow Ensembles')
    
    parser.add_argument("--dataset", help="Text dataset (YAHOO or AG_News)",
                        choices=["AG_News", "YAHOO"], default="AG_News")
    
    parser.add_argument("--labeled", help="Number of labeled data",
                        default=200, type=int)
    
    parser.add_argument("--net_config", help="Flow configuration",
                        default={'k':1024,'coupling_layers':7,'nperlayer':1})
    
    parser.add_argument("--num_models", help="Number of models in ensembles",
                        default=6, type=int)
    
    parser.add_argument("--num_epochs", help="Number of training epochs",
                        default=100, type=int)
    
    parser.add_argument("--test_epochs", help="Number of training epochs per test phase",
                        default=5, type=int)
    
    parser.add_argument("--lr", help="Learning rate",
                        default=0.0005, type=float)
    
    parser.add_argument("--swa_freq_pct", help="SWA frequency percentage of number of epochs",
                        default=0.01, type=float)
    
    parser.add_argument("--swa_lr_factor", help="SWA learning rate as factor of original learning rate",
                        default=0.2, type=float)
    
    parser.add_argument("--unlab_weight", help="Unlabeled data weight",
                        default=.6, type=float)
    
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    
    dataset = AG_News if args.dataset=="AG_News" else YAHOO
    labeled = args.labeled
    net_config = args.net_config
    num_models = args.num_models
    num_epochs = args.num_epochs
    test_epochs = args.test_epochs
    lr = args.lr
    swa_freq_pct = args.swa_freq_pct
    swa_lr_factor = args.swa_lr_factor
    unlab_weight = args.unlab_weight
    
    train_data, test_data = dataset(), dataset(train=False) 
    trainers=[
        make_trainer(
            train_data=train_data,
            test_data=test_data,
            split={'train':labeled,'val':5000},
            net_config=net_config,
            num_epochs=num_epochs,
            lr=lr,
            swag=True,
            swa_config={'swa_dec_pct':.5, 'swa_start_pct':.75, 'swa_freq_pct':swa_freq_pct, 'swa_lr_factor':swa_lr_factor},
            swag_config={'subspace':'covariance', 'max_num_models':20},
            trainer=SemiFlowSWAG,
            trainer_config={'unlab_weight':unlab_weight}
        ) for i in range(num_models)
    ]
    swag = SWAG_Results(trainers[0], 10, .5)
#     epoch = 0
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainers[0].train(num_epochs)
#         while epoch < num_epochs:
            
#             for trainer in trainers:
#                 trainer.train(test_epochs)
                
#             epoch += test_epochs
#             ensembles.update_results(epoch)
        swag.update_results(num_epochs)
            
    print(swag.results)
            
    os.makedirs('results', exist_ok = True)
    filename = "{}-swag-{:03}.json".format(args.dataset, random.randrange(1, 10**3))
    with open(os.path.join('results',filename), 'w') as outfile:
        json.dump(swag.results, outfile)