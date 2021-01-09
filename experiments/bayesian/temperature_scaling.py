from oil.utils.utils import Eval

import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np

class ModelWithTemperature():
    def __init__(self, trainer):
        self.trainer = trainer
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
#         self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()
        
        logits, probs, preds, targets, acc = self.trainer.getTestOutputs(loader='val')
        logits = torch.from_numpy(logits)
        targets = torch.from_numpy(targets)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, targets).item()
        before_temperature_ece = ece_criterion(logits, targets).item()
        print('Validation - Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), targets)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), targets).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), targets).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('Validation - After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        logits, probs, preds, targets, acc = self.trainer.getTestOutputs(loader='test')
        logits = torch.from_numpy(logits)
        targets = torch.from_numpy(targets)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, targets).item()
        before_temperature_ece = ece_criterion(logits, targets).item()
        print('Test - Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        self.out_before = ece_criterion.out
        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), targets).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), targets).item()
        print('Test - After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        self.out_after = ece_criterion.out
        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=25):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        
        xs = []
        ys = []
        zs = []

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                xs.append(avg_confidence_in_bin)
                ys.append(accuracy_in_bin)
                zs.append(prop_in_bin)
        xs = np.array(xs)
        ys = np.array(ys)
        zs = np.array(zs)

        self.out = {
            'confidence': xs,
            'accuracy': ys,
            'p': zs,
            'ece': ece,
        }

        return ece
    
class EnsemblesWithTemperature():
    def __init__(self, trainers):
        self.trainers = trainers
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature
    def set_temperature(self):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = _ECELoss().cuda()
        
        outputs=[]
        for index, trainer in enumerate(self.trainers):
            logits, probs, preds, targets, acc = trainer.getTestOutputs(loader='val')
            outputs.append([logits, probs, preds, targets, acc])

        en_probs, en_acc, en_logits, fl_probs, fl_acc, targets = self.__get_ensembles_accuracy(outputs)
        logits = torch.from_numpy(en_logits)
        targets = torch.from_numpy(targets)
        before_temperature_nll = nll_criterion(logits, targets).item()
        before_temperature_ece = ece_criterion(logits, targets).item()
        print('Validation - Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), targets)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), targets).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), targets).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('Validation - After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        
        outputs=[]
        for index, trainer in enumerate(self.trainers):
            logits, probs, preds, targets, acc = trainer.getTestOutputs(loader='test')
            outputs.append([logits, probs, preds, targets, acc])
        en_probs, en_acc, en_logits, fl_probs, fl_acc, targets = self.__get_ensembles_accuracy(outputs)
        logits = torch.from_numpy(en_logits)
        targets = torch.from_numpy(targets)
        
        before_temperature_nll = nll_criterion(logits, targets).item()
        before_temperature_ece = ece_criterion(logits, targets).item()
        print('Test - Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))
        self.out_before = ece_criterion.out
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), targets).item()
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), targets).item()
        print('Test - After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))
        self.out_after = ece_criterion.out
        return self
            
    
    def __get_ensembles_accuracy(self, outputs):
        targets = outputs[0][3]

        en_probs = np.mean(np.array(outputs)[:,1], axis=0)

        en_logits = np.mean(np.array(outputs)[:,0], axis=0)
        fl_probs = F.softmax(torch.from_numpy(en_logits), dim=1).numpy()

        en_preds = np.argmax(en_probs, axis=1)
        fl_pred = np.argmax(fl_probs, axis=1)

        en_acc = (en_preds == targets).mean()
        fl_acc = (fl_pred == targets).mean()

        return en_probs, en_acc, en_logits, fl_probs, fl_acc, targets