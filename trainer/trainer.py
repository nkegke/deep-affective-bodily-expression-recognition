import numpy as np
import torch
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, make_barplot, features_blobs, setup_cam, returnCAM
import matplotlib as mpl
import random

mpl.use('Agg')
import matplotlib.pyplot as plt
import model.metric
import model.loss



class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, 
                 categorical=True, fusion=False,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)

        self.data_loader = data_loader
        self.categorical = categorical

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(self.data_loader)
            self.len_epoch = len_epoch

        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(self.data_loader.batch_size))
        self.criterion_categorical = criterion
        self.categorical_class_metrics = [_class + "_" + m.__name__ for _class in self.valid_data_loader.dataset.categorical_emotions for m in self.metric_ftns]
        self.train_metrics = MetricTracker('loss',
            'roc_auc_micro', 'roc_auc_macro', writer=self.writer)
        self.valid_metrics = MetricTracker('loss',
            'roc_auc_micro', 'roc_auc_macro', writer=self.writer)
            
        self.fusion = fusion
        # if self.fusion ==True:
        #     self.model_copy = copy.deepcopy(self.model)

    def _train_epoch(self, epoch, phase="train"):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        import torch.nn.functional as F
        import model.loss
        print("Finding LR")
        for param_group in self.optimizer.param_groups:
            print(param_group['lr'])

        if phase == "train":
            self.model.train()
            self.train_metrics.reset()
            torch.set_grad_enabled(True)
            metrics = self.train_metrics
        elif phase == "val" or phase == 'test':
            self.model.eval()
            self.valid_metrics.reset()
            torch.set_grad_enabled(False)
            metrics = self.valid_metrics

        outputs = []
        targets = []

        data_loader = self.data_loader if phase == "train" else self.valid_data_loader

        paths = []
        for batch_idx, (data, target, lengths, face_data) in enumerate(data_loader):
                
            data, target = data.to(self.device), target.to(self.device)
            
            if phase == "train":
                self.optimizer.zero_grad()

            out = self.model(data)

            loss = 0

            if self.categorical:
                loss_categorical = self.criterion_categorical(out['categorical'], target)
                loss += loss_categorical
                output = out['categorical'].cpu().detach().numpy()

                if self.fusion==True:
                    face_data = face_data.to(self.device)
                    face_out = self.model(face_data)
                    face_loss_categorical = self.criterion_categorical(face_out['categorical'], target)
                    loss += face_loss_categorical

                    face_output = face_out['categorical'].cpu().detach().numpy()

                    # late fusion
                    output = (output + face_output)/2.0 # average
                    # output = np.maximum(output, face_output) # maximum

                target = target.cpu().detach().numpy()
                targets.append(target)
                paths.append(lengths)
                outputs.append(output)
                    


            if phase == "train":
                loss.backward()
                self.optimizer.step()

            if batch_idx % self.log_step == 0:
                if not self.categorical:
                    loss_categorical = torch.tensor([np.nan]).float()
                
                self.logger.debug('{} Epoch: {} {} Loss: {:.6f} '.format(
                    phase,
                    epoch,
                    self._progress(batch_idx, data_loader),
                    loss.item()))

            if batch_idx == self.len_epoch:
                break



        if phase == "train":
            self.writer.set_step(epoch)
        else:
            self.writer.set_step(epoch, phase)

        metrics.update('loss', loss.item())


        if self.categorical:
            metrics.update('loss_categorical', loss_categorical.item())
            output = np.concatenate(outputs, axis=0)
            target = np.concatenate(targets, axis=0)

            ap = model.metric.average_precision(output, target)
            roc_auc = model.metric.roc_auc(output, target)

            metrics.update("roc_auc_micro", model.metric.roc_auc(output, target, average='micro'))
            metrics.update("roc_auc_macro", model.metric.roc_auc(output, target, average='macro'))

            self.writer.add_figure('%s roc auc per class' % phase, make_barplot(roc_auc, self.valid_data_loader.dataset.categorical_emotions, 'roc auc'))


        log = metrics.result()


        if phase == "train":
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if self.categorical:
                self.writer.save_results(output, "output_train")
                self.writer.save_results(target, "target_train")
                paths = np.concatenate(paths, axis=0)
                self.writer.save_results(paths, "paths_train")

            if self.do_validation:
                val_log = self._train_epoch(epoch, phase="val")
                log.update(**{'val_' + k: v for k, v in val_log.items()})

            return log

        elif phase == "val" or phase == 'test':
            if self.categorical:
                self.writer.save_results(output, "output")

            return metrics.result()


    def _progress(self, batch_idx, data_loader):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(data_loader, 'n_samples'):
            current = batch_idx * data_loader.batch_size
            total = data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
