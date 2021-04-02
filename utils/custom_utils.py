import os, math
from tsnecuda import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import copy
import random
import numpy as np

from utils.storage import save_statistics


class TSNE_logger:
    def __init__(self, num_classes=None, **kwargs):
        self.num_classes = num_classes
        self.tsne = TSNE(**kwargs)

    def run(self, features, labels=None, plot=False):
        embeddings = self.tsne.fit_transform(features)
        if plot:
            self.plot(embeddings, labels, self.num_classes)
        
    def plot(self, embeddings, labels, num_classes):
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap('jet', num_classes), marker='.')
        plt.colorbar(ticks=range(num_classes))
        #plt.clim(-0.5, 9.5)
        plt.show()

        
class PCA_logger:
    def __init__(self, num_classes=None, **kwargs):
        self.num_classes = num_classes
        self.pca = PCA(**kwargs)

    def run(self, features, labels=None, plot=False):
        embeddings = self.pca.fit_transform(features)
        if plot:
            self.plot(embeddings, labels, self.num_classes)
        
    def plot(self, embeddings, labels, num_classes):
        vis_x = embeddings[:, 0]
        vis_y = embeddings[:, 1]
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap('jet', num_classes), marker='.')
        plt.colorbar(ticks=range(num_classes))
        #plt.clim(-0.5, 9.5)
        plt.show()


class LogLandscape:
    def __init__(self, args, phase, num_total_tasks, num_inner_steps=5, log_period=1):
        self.args = args
        self.way = args.num_classes_per_set
        self.shot = args.num_samples_per_class
        self.phase = phase
        self.num_total_tasks = num_total_tasks
        self.num_inner_steps = num_inner_steps
        self.log_period = log_period

        self.min_loss_curve = [[] for _ in range(num_inner_steps)]
        self.max_loss_curve = [[] for _ in range(num_inner_steps)]
        self.min_grad_curve = [[] for _ in range(num_inner_steps)]
        self.max_grad_curve = [[] for _ in range(num_inner_steps)]
        self.beta = [[] for _ in range(num_inner_steps)]
        #self.min_loss = self.min_grad = self.min_beta = 9999.
        #self.max_loss = self.max_grad = self.max_beta = 0.
        self.losses = []
        self.grads = []
        #self.betas = []
        self.steps = []
        self.num_values = 0
        self.num_tasks = 0

        self.total_min_loss_curve = []
        self.total_max_loss_curve = []
        self.total_min_grad_curve = []
        self.total_max_grad_curve = []
        self.total_beta = []

        if not os.path.isdir('stats/landscape/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot)):
            os.makedirs('stats/landscape/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot))
        if not os.path.isdir('stats/gradients/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot)):
            os.makedirs('stats/gradients/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot))
        if not os.path.isdir('stats/weight_norm/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot)):
            os.makedirs('stats/weight_norm/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot))
        if not os.path.isdir('stats/updated_norm/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot)):
            os.makedirs('stats/updated_norm/{}_{}way_{}shot'.format(args.dataset_name, self.way, self.shot))

        try:
            if os.environ['WANDB_RUN_ID']:
                pass
        except:
            try:
                if os.environ['LANDSCAPE']:
                    save_statistics('.',
                                    ['iter', 'min_loss', 'max_loss', 'min_grad', 'max_grad', 'beta'], 
                                    create=True,
                                    filename='stats/landscape/{}_{}way_{}shot/landscape_{}_{}.csv'.format(args.dataset_name, self.way, self.shot,
                                                                                                          os.environ['LANDSCAPE'], self.phase))
                    if self.phase == 'train':
                        save_statistics('.',
                                        ['iter', 'inner-conv-0', 'inner-conv-1', 'inner-conv-2', 'inner-conv-3', 'inner-linear-weight', 'inner-linear-bias',
                                        'outer-conv-0', 'outer-conv-1', 'outer-conv-2', 'outer-conv-3', 'outer-linear-weight', 'outer-linear-bias'], 
                                        create=True,
                                        filename='stats/gradients/{}_{}way_{}shot/gradients_{}_{}.csv'.format(args.dataset_name, self.way, self.shot, 
                                                                                                              os.environ['LANDSCAPE'], self.phase))
                        save_statistics('.',
                                        ['iter', 'inner-conv-0', 'inner-conv-1', 'inner-conv-2', 'inner-conv-3', 'inner-linear-weight', 'inner-linear-bias',
                                        'outer-conv-0', 'outer-conv-1', 'outer-conv-2', 'outer-conv-3', 'outer-linear-weight', 'outer-linear-bias'], 
                                        create=True,
                                        filename='stats/weight_norm/{}_{}way_{}shot/weight_norm_{}_{}.csv'.format(args.dataset_name, self.way, self.shot, 
                                                                                                                  os.environ['LANDSCAPE'], self.phase))
                        save_statistics('.',
                                        ['iter', 'inner-conv-0', 'inner-conv-1', 'inner-conv-2', 'inner-conv-3', 'inner-linear-weight', 'inner-linear-bias',
                                        'outer-conv-0', 'outer-conv-1', 'outer-conv-2', 'outer-conv-3', 'outer-linear-weight', 'outer-linear-bias'], 
                                        create=True,
                                        filename='stats/updated_norm/{}_{}way_{}shot/updated_norm_{}_{}.csv'.format(args.dataset_name, self.way, self.shot, 
                                                                                                                    os.environ['LANDSCAPE'], self.phase))
                        save_statistics('.',
                                        ['iter', 'inner-conv-0-min','inner-conv-0-max', 'inner-conv-1-min', 'inner-conv-1-max', 
                                        'inner-conv-2-min', 'inner-conv-2-max', 'inner-conv-3-min', 'inner-conv-3-max', 
                                        'inner-linear-weight-min', 'inner-linear-weight-max', 'inner-linear-bias-min', 'inner-linear-bias-max'],
                                        create=True,
                                        filename='stats/gradients/{}_{}way_{}shot/gradients_{}_{}_min_max.csv'.format(args.dataset_name, self.way, self.shot, 
                                                                                                                      os.environ['LANDSCAPE'], self.phase))
                        save_statistics('.',
                                        ['iter', 'inner-conv-0-min','inner-conv-0-max', 'inner-conv-1-min', 'inner-conv-1-max', 
                                        'inner-conv-2-min', 'inner-conv-2-max', 'inner-conv-3-min', 'inner-conv-3-max', 
                                        'inner-linear-weight-min', 'inner-linear-weight-max', 'inner-linear-bias-min', 'inner-linear-bias-max'],
                                        create=True,
                                        filename='stats/weight_norm/{}_{}way_{}shot/weight_norm_{}_{}_min_max.csv'.format(args.dataset_name, self.way, self.shot, 
                                                                                                                          os.environ['LANDSCAPE'], self.phase))
                        save_statistics('.',
                                        ['iter', 'inner-conv-0-min','inner-conv-0-max', 'inner-conv-1-min', 'inner-conv-1-max', 
                                        'inner-conv-2-min', 'inner-conv-2-max', 'inner-conv-3-min', 'inner-conv-3-max', 
                                        'inner-linear-weight-min', 'inner-linear-weight-max', 'inner-linear-bias-min', 'inner-linear-bias-max'],
                                        create=True,
                                        filename='stats/updated_norm/{}_{}way_{}shot/updated_norm_{}_{}_min_max.csv'.format(args.dataset_name, self.way, self.shot, 
                                                                                                                            os.environ['LANDSCAPE'], self.phase))
            except:
                pass

    
    def __call__(self, loss, weight, origin_grad, lr, current_step):
        def zero_grad(params):
            for name, param in params.items():
                if param.requires_grad == True:
                    if param.grad is not None:
                        if torch.sum(param.grad) > 0:
                            print(param.grad)
                            param.grad.zero_()
                            params[name].grad = None
        
        zero_grad(weight)
        grad = torch.autograd.grad(loss, weight.values())
        grad = dict(zip(weight.keys(), grad))

        for key, g in grad.items():
            if g is None:
                print('Grads not found for inner loop parameter', key)
            grad[key] = grad[key].sum(dim=0)

        flatten_grad = torch.cat(list(map(torch.flatten, grad.values()))).detach().cpu()
        flatten_origin_grad = torch.cat(list(map(torch.flatten, origin_grad.values()))).detach().cpu()
        #diff_grad = np.linalg.norm(grad - origin_grad)
        diff_grad_norm = (flatten_grad - flatten_origin_grad).norm(2).item()
        step = (lr * flatten_origin_grad).norm(2).item()
        #beta = diff_grad_norm / step

        #loss = loss.item()
        #if loss < self.min_loss:
        #    self.min_loss = loss
        #elif loss > self.max_loss:
        #    self.max_loss = loss

        #if diff_grad_norm < self.min_grad:
        #    self.min_grad = diff_grad_norm
        #elif diff_grad_norm > self.max_grad:
        #    self.max_grad = diff_grad_norm

        #if beta < self.min_beta:
        #    self.min_beta = beta
        
        self.num_values += 1

        self.losses.append(loss.item())
        self.grads.append(diff_grad_norm)
        #self.betas.append(beta)
        self.steps.append(step)

        if self.num_values == 8:
            #self.min_loss_curve[current_step].append(self.min_loss)
            #self.max_loss_curve[current_step].append(self.max_loss)
            #self.min_grad_curve[current_step].append(self.min_grad)
            #self.max_grad_curve[current_step].append(self.max_grad)
            #self.beta[current_step].append(self.min_beta)
            self.min_loss_curve[current_step].append(np.min(self.losses))
            self.max_loss_curve[current_step].append(np.max(self.losses))
            self.min_grad_curve[current_step].append(np.min(self.grads))
            self.max_grad_curve[current_step].append(np.max(self.grads))
            self.beta[current_step].append(np.max(self.grads) / self.steps[np.argmax(self.grads)])
            #self.beta[current_step].append(np.min(self.betas))
            #self.beta[current_step].append(np.max(self.betas))

            self.losses.clear()
            self.grads.clear()
            #self.betas.clear()
            #self.min_loss = self.min_grad = self.min_beta = 9999.
            #self.max_loss = self.max_grad = self.max_beta = 0.
            self.num_values = 0

    def get_result(self, current_iter):
        if (current_iter + 1) % self.log_period != 0:
            return

        if self.num_tasks == self.num_total_tasks:
            for step in range(self.num_inner_steps):
                
                self.min_loss_curve[step] = np.mean(self.min_loss_curve[step])
                self.max_loss_curve[step] = np.mean(self.max_loss_curve[step])
                self.min_grad_curve[step] = np.mean(self.min_grad_curve[step])
                self.max_grad_curve[step] = np.mean(self.max_grad_curve[step])
                self.beta[step] = np.mean(self.beta[step])
                '''
                # min-max in a batch
                self.min_loss_curve[step] = np.min(self.min_loss_curve[step])
                self.max_loss_curve[step] = np.max(self.max_loss_curve[step])
                self.min_grad_curve[step] = np.min(self.min_grad_curve[step])
                self.max_grad_curve[step] = np.max(self.max_grad_curve[step])
                self.beta[step] = np.min(self.beta[step])
                '''
            
            min_loss_curve = np.mean(self.min_loss_curve)
            max_loss_curve = np.mean(self.max_loss_curve)
            min_grad_curve = np.mean(self.min_grad_curve)
            max_grad_curve = np.mean(self.max_grad_curve)
            beta = np.mean(self.beta)

            self.total_min_loss_curve.append(min_loss_curve)
            self.total_max_loss_curve.append(max_loss_curve)
            self.total_min_grad_curve.append(min_grad_curve)
            self.total_max_grad_curve.append(max_grad_curve)
            self.total_beta.append(beta)

            self.write_csv(current_iter, min_loss_curve, max_loss_curve, min_grad_curve, max_grad_curve, beta)
            self.reset_buffer()
        else:
            return

        #return self.total_min_loss_curve, self.total_max_loss_curve, \
        #    self.total_min_grad_curve, self.total_max_grad_curve, self.total_beta
        
    #def write_csv(self, current_iter, min_loss_curve, max_loss_curve, min_grad_curve, max_grad_curve, beta):
    def write_csv(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/landscape/{}_{}way_{}shot/landscape_{}_{}.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                      os.environ['LANDSCAPE'], self.phase))
        except:
            pass

    def write_csv_gradients_norm(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/gradients/{}_{}way_{}shot/gradients_{}_{}.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                      os.environ['LANDSCAPE'], self.phase))
        except:
            pass

    def write_csv_weight_norm(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/weight_norm/{}_{}way_{}shot/weight_norm_{}_{}.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                          os.environ['LANDSCAPE'], self.phase))
        except:
            pass

    def write_csv_updated_norm(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/updated_norm/{}_{}way_{}shot/updated_norm_{}_{}.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                            os.environ['LANDSCAPE'], self.phase))
        except:
            pass

    def write_csv_gradients_norm_min_max(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/gradients/{}_{}way_{}shot/gradients_{}_{}_min_max.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                              os.environ['LANDSCAPE'], self.phase))
        except:
            pass

    def write_csv_weight_norm_min_max(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/weight_norm/{}_{}way_{}shot/weight_norm_{}_{}_min_max.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                                  os.environ['LANDSCAPE'], self.phase))
        except:
            pass

    def write_csv_updated_norm_min_max(self, *args):
        try:
            if os.environ['LANDSCAPE']:
                save_statistics('.',
                                [*args], 
                                create=False,
                                filename='stats/updated_norm/{}_{}way_{}shot/updated_norm_{}_{}_min_max.csv'.format(self.args.dataset_name, self.way, self.shot, 
                                                                                                                    os.environ['LANDSCAPE'], self.phase))
        except:
            pass
                
    def reset_buffer(self):
        self.min_loss_curve = [[] for _ in range(self.num_inner_steps)]
        self.max_loss_curve = [[] for _ in range(self.num_inner_steps)]
        self.min_grad_curve = [[] for _ in range(self.num_inner_steps)]
        self.max_grad_curve = [[] for _ in range(self.num_inner_steps)]
        self.beta = [[] for _ in range(self.num_inner_steps)]

        self.losses = []
        self.grads = []
        #self.betas = []
        self.steps = []
        #self.min_loss = self.min_grad = self.min_beta = 9999.
        #self.max_loss = self.max_grad = self.max_beta = 0.
        self.num_values = 0
        self.num_tasks = 0
        
    def reset_all(self):
        self.__init__(self.phase, self.num_total_tasks, self.num_inner_steps, self.log_period)

        
def centralized_gradient(names_grads_copy, use_gc=True, gc_conv_only=False, standardization=False):
    if use_gc:
        updated_names_grads_copy = dict()
        for key, grad in names_grads_copy.items():
            if standardization:
                if len(list(grad.size())) > 3:
                    std = grad.view(grad.size(0), -1).std(dim=1).view(-1, 1, 1, 1).detach()
                elif len(list(grad.size())) > 1:
                    std = grad.view(grad.size(0), -1).std(dim=1).view(grad.size(0), 1).detach()
                else:
                    std = grad.std(dim=0).detach()
                    #std[torch.isnan(std)] = 1.
                std[std == 0] = std[std == 0] + 1e-5
                #print(key, grad, std)
            if gc_conv_only:
                if len(list(grad.size()))>3:
                    grad = grad.add(-grad.mean(dim=tuple(range(1,len(list(grad.size())))), keepdim=True))
            else:
                if len(list(grad.size()))>1:
                    grad = grad.add(-grad.mean(dim=tuple(range(1,len(list(grad.size())))), keepdim=True))
            #grad = grad / std.expand_as(grad)
            updated_names_grads_copy[key] = grad
        return updated_names_grads_copy
    else:
        return names_grads_copy


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    import numpy as np

    def softmax(v):
        e = np.exp(v)
        return e / e.sum()
    
    digits = load_digits()
    logger = TSNE_logger(10, perplexity=10, learning_rate=10)
    logger.run(digits.data, digits.target, True)

    logger = PCA_logger(10)
    logger.run(softmax(digits.data), digits.target, True)
