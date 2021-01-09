from swag.utils import bn_update, calibration_curve

import torch
import torch.nn.functional as F

import numpy as np
import json

class SWAG_Results:
    def __init__(self, trainer, num_models, scale=.5, num_bins=60):
        self.num_models = num_models
        self.trainer = trainer
        self.scale = scale
        self.num_bins = num_bins
        self.results = self.__init_results()
        
    def __init_results(self):
        results = {}
        results["Number of models"]=self.num_models
        results["Epochs"]=[]
        for index in range(self.num_models):
            results["Model {}".format(index+1)]=[]
        results["SWA"]=[]
        results["SWAG"]=[]
        results["Flow SWAG"]=[]
        return results
    def update_results(self, epoch, calibration=False):
        outputs=[]
        self.results["Epochs"].append(epoch)
        self.trainer.swag_model.set_swa()
        logits, probs, preds, targets, acc = self.trainer.getTestOutputs()
        self.results["SWA"].append(acc)
        if calibration:
            calibration_dict = calibration_curve(probs, targets, num_bins=self.num_bins)
            self.results["SWA calibration"] = calibration_dict
        for index in range(self.num_models):
            self.trainer.swag_model.sample(scale=self.scale)
            bn_update(self.trainer.dataloaders["train"], self.trainer.swag_model)
            logits, probs, preds, targets, acc = self.trainer.getTestOutputs()
            self.results["Model {}".format(index+1)].append(acc)
            outputs.append([logits, probs, preds, targets, acc])
            if calibration:
                calibration_dict = calibration_curve(probs, targets, num_bins=self.num_bins)
                self.results["Model {} calibration".format(index+1)] = calibration_dict
        
        en_probs, en_acc, fl_probs, fl_acc, targets = self.__get_ensembles_accuracy(outputs)
        self.results["SWAG"].append(en_acc)
        self.results["Flow SWAG"].append(fl_acc)
        if calibration:
            calibration_dict = calibration_curve(en_probs, targets, num_bins=self.num_bins)
            self.results["SWAG calibration"] = calibration_dict
            calibration_dict = calibration_curve(fl_probs, targets, num_bins=self.num_bins)
            self.results["Flow SWAG calibration"] = calibration_dict
        
    def __get_ensembles_accuracy(self, outputs):
        targets = outputs[0][3]

        en_probs = np.mean(np.array(outputs)[:,1], axis=0)

        en_logits = np.mean(np.array(outputs)[:,0], axis=0)
        fl_probs = F.softmax(torch.from_numpy(en_logits), dim=1).numpy()

        en_preds = np.argmax(en_probs, axis=1)
        fl_pred = np.argmax(fl_probs, axis=1)

        en_acc = (en_preds == targets).mean()
        fl_acc = (fl_pred == targets).mean()

        return en_probs, en_acc, fl_probs, fl_acc, targets
    
class Ensembles:
    def __init__(self, trainers, num_bins = 60, all_ensembles=False):
        self.num_models = len(trainers)
        self.trainers = trainers
        self.num_bins = num_bins
        self.min_ensembles = 1 if all_ensembles else self.num_models
        self.results = self.__init_results()
    def __init_results(self):
        results = {}
        results["Number of models"]=self.num_models
        results["Epochs"]=[]
        for index in range(self.num_models):
            results["Model {}".format(index+1)]=[]
        for i in range(self.min_ensembles, self.num_models+1):
            results["Deep Ensembles-{}".format(i)]=[]
            results["Flow Ensembles-{}".format(i)]=[]
        return results
    def update_results(self, epoch, calibration=False):
        outputs=[]
        self.results["Epochs"].append(epoch)
        for index, trainer in enumerate(self.trainers):
            logits, probs, preds, targets, acc = trainer.getTestOutputs()
            self.results["Model {}".format(index+1)].append(acc)
            outputs.append([logits, probs, preds, targets, acc])
            if calibration:
                calibration_dict = calibration_curve(probs, targets, num_bins=self.num_bins)
                self.results["Model {} calibration".format(index+1)] = calibration_dict
        
        for i in range(self.min_ensembles, self.num_models+1):
            en_probs, en_acc, fl_probs, fl_acc, targets = self.__get_ensembles_accuracy(outputs[0:i])
            self.results["Deep Ensembles-{}".format(i)].append(en_acc)
            self.results["Flow Ensembles-{}".format(i)].append(fl_acc)
            if calibration:
                calibration_dict = calibration_curve(en_probs, targets, num_bins=self.num_bins)
                self.results["Deep Ensembles-{} calibration".format(i)] = calibration_dict
                calibration_dict = calibration_curve(fl_probs, targets, num_bins=self.num_bins)
                self.results["Flow Ensembles-{} calibration".format(i)] = calibration_dict
        
    def __get_ensembles_accuracy(self, outputs):
        targets = outputs[0][3]

        en_probs = np.mean(np.array(outputs)[:,1], axis=0)

        en_logits = np.mean(np.array(outputs)[:,0], axis=0)
        fl_probs = F.softmax(torch.from_numpy(en_logits), dim=1).numpy()

        en_preds = np.argmax(en_probs, axis=1)
        fl_pred = np.argmax(fl_probs, axis=1)

        en_acc = (en_preds == targets).mean()
        fl_acc = (fl_pred == targets).mean()

        return en_probs, en_acc, fl_probs, fl_acc, targets
    
    def write_results(filename):
        with open(filename, 'w') as outfile:
            json.dump(self.results, outfile)
            
    @staticmethod
    def load_results(filename):
        with open(filename) as json_file:
            results = json.load(json_file)
        return results
    
    @staticmethod
    def plot_results(results, ax, title, start=0):
        num_models = results["Number of models"]
        epochs_list = results["Epochs"]
        for index in range(num_models):
            accs = results["Model {}".format(index+1)]
            ax.plot(epochs_list[start:], accs[start:], lw=1, label="Model {}".format(index+1), color='C{}'.format(index)) 

        ens_probs_accs = results["Deep Ensembles-{}".format(num_models)]
        ax.plot(epochs_list[start:], ens_probs_accs[start:], "-v", lw=2, markersize=8, label="Deep Ensembles", color='C6')

        ens_probs_logits = results["Flow Ensembles-{}".format(num_models)]
        ax.plot(epochs_list[start:], ens_probs_logits[start:], "-*", lw=3, markersize=12, label="Flow Ensembles", color='C3')

        ax.set(xlabel='epoch', ylabel='accuracy', title=title)
        ax.grid()
        
    @staticmethod
    def plot_results_all(results, ax, title, start=0):
        num_models = results["Number of models"]
        epochs_list = results["Epochs"]
        xs = range(1,num_models+1,1)
        ys_deep, ys_flow = [], []
        for index in range(1, num_models+1):
            ys_deep.append(results["Deep Ensembles-{}".format(index)])
            ys_flow.append(results["Flow Ensembles-{}".format(index)])

        ax.plot(xs[start:], ys_deep[start:], lw=2, markersize=8, label="Deep Ensembles", color='C6')

        ax.plot(xs[start:], ys_flow[start:], lw=2, markersize=8, label="Flow Ensembles", color='C3')

        ax.set(xlabel='Number of models', ylabel='Accuracy', title=title)
        ax.grid()