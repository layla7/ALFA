from functools import partial, reduce
import os

import numpy as np
import wandb
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from meta_neural_network_architectures import VGGReLUNormNetwork, ResNet12
from inner_loop_optimizers import (GradientDescentLearningRule,
                                   LSLRGradientDescentLearningRule,
                                   LSLRGradientDescentLearningRuleALFA)
from utils.custom_utils import (LogLandscape, 
                                TSNE_logger,
                                PCA_logger)
from utils.storage import save_statistics


def set_torch_seed(seed):
    """
    Sets the pytorch seeds for current experiment run
    :param seed: The seed (int)
    :return: A random number generator to use
    """
    rng = np.random.RandomState(seed=seed)
    torch_seed = rng.randint(0, 999999)
    torch.manual_seed(seed=torch_seed)

    return rng


def get_partial_weights(arg):
    for name, value in arg:
        if 'attention' not in name:
            yield (name, value)


class MAMLFewShotClassifier(nn.Module):
    def __init__(self, im_shape, device, args):
        """
        Initializes a MAML few shot learning system
        :param im_shape: The images input size, in batch, c, h, w shape
        :param device: The device to use to use the model on.
        :param args: A namedtuple of arguments specifying various hyperparameters.
        """
        super(MAMLFewShotClassifier, self).__init__()
        os.environ['LANDSCAPE'] = args.wandb_run_name
        self.args = args
        self.device = device
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda
        self.im_shape = im_shape
        self.current_epoch = 0
        self.gradients_buffer = []
        self.gradients_counter = 0
        self.tsne_logger = TSNE_logger(num_classes=args.num_classes_per_set, perplexity=10, learning_rate=10)
        self.pca_logger = PCA_logger(num_classes=args.num_classes_per_set)
        self.probs = []
        self.true_probs = []
        self.train_total_min_loss = []
        self.train_total_max_loss = []
        self.val_total_min_loss = []
        self.val_total_max_loss = []
        self.train_landscape_logger = LogLandscape(args, 'train', 2, args.number_of_training_steps_per_iter)
        self.val_landscape_logger = LogLandscape(args, 'val', 600, args.number_of_evaluation_steps_per_iter)

        #save_statistics('.',
        #                ['step', 'min', 'max'], create=True,
        #                filename='landscape_{}_train.csv'.format(os.environ['LANDSCAPE']))
        #save_statistics('.',
        #                ['step', 'min', 'max'], create=True,
        #                filename='landscape_{}_val.csv'.format(os.environ['LANDSCAPE']))

        self.rng = set_torch_seed(seed=args.seed)

        if self.args.backbone == 'ResNet12':
            self.classifier = ResNet12(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)
        else:
            self.classifier = VGGReLUNormNetwork(im_shape=self.im_shape, num_output_classes=self.args.
                                                 num_classes_per_set,
                                                 args=args, device=device, meta_classifier=True).to(device=self.device)

        self.task_learning_rate = args.init_inner_loop_learning_rate
        names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())

        if self.args.alfa:
            self.inner_loop_optimizer = LSLRGradientDescentLearningRuleALFA(device=device,
                                                                        init_learning_rate=self.task_learning_rate,
                                                                        init_weight_decay=args.init_inner_loop_weight_decay,
                                                                        total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                        use_learnable_weight_decay=self.args.alfa,
                                                                        use_learnable_learning_rates=self.args.alfa,
                                                                        alfa=self.args.alfa, random_init=self.args.random_init)
            self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)
        elif self.args.learnable_per_layer_per_step_inner_loop_learning_rate:
            self.inner_loop_optimizer = LSLRGradientDescentLearningRule(device=device,
                                                                        init_learning_rate=self.task_learning_rate,
                                                                        total_num_inner_loop_steps=self.args.number_of_training_steps_per_iter,
                                                                        use_learnable_learning_rates=True)
            self.inner_loop_optimizer.initialise(names_weights_dict=names_weights_copy)
        else:
            self.inner_loop_optimizer = GradientDescentLearningRule(device=device,
                                                                    learning_rate=self.task_learning_rate)



        print("Inner Loop parameters")
        for key, value in self.inner_loop_optimizer.named_parameters():
            print(key, value.shape)

        self.use_cuda = args.use_cuda
        self.device = device
        self.args = args
        self.to(device)
        self.num_conv_layers = len(names_weights_copy) - 2

        print("Outer Loop parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

        # L2F
        if self.args.attenuate:
            num_layers = len(names_weights_copy)
            self.attenuator = nn.Sequential(
                nn.Linear(num_layers, num_layers),
                nn.ReLU(inplace=True),
                nn.Linear(num_layers, num_layers),
                nn.Sigmoid()
                #nn.Softplus()
            ).to(device=self.device)

        # ALFA
        if self.args.alfa:
            num_layers = len(names_weights_copy)
            input_dim = num_layers*2
            self.regularizer = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim)
            ).to(device=self.device)

        if self.args.attenuate:
            if self.args.alfa:
                self.optimizer = optim.Adam([
                    {'params': self.classifier.parameters()},
                    {'params': self.inner_loop_optimizer.parameters()},
                    {'params': self.regularizer.parameters()},
                    {'params': self.attenuator.parameters()},
                ],lr=args.meta_learning_rate, amsgrad=False)
            else:
                self.optimizer = optim.Adam([
                    {'params': self.classifier.parameters()},
                    {'params': self.attenuator.parameters()},
                ],lr=args.meta_learning_rate, amsgrad=False)
        else:
            if self.args.alfa:
                if self.args.random_init:
                    self.optimizer = optim.Adam([
                        {'params': self.inner_loop_optimizer.parameters()},
                        {'params': self.regularizer.parameters()},
                    ], lr=args.meta_learning_rate, amsgrad=False)
                else:
                    self.optimizer = optim.Adam([
                        {'params': self.classifier.parameters()},
                        {'params': self.inner_loop_optimizer.parameters()},
                        {'params': self.regularizer.parameters()},
                    ], lr=args.meta_learning_rate, amsgrad=False)
            else:
                if self.args.learnable_per_layer_per_step_inner_loop_learning_rate:
                    self.optimizer = optim.Adam([
                        {'params': self.classifier.trainable_parameters()},
                        {'params': self.inner_loop_optimizer.parameters()},
                    ], lr=args.meta_learning_rate, amsgrad=False)
                else:
                    self.optimizer = optim.Adam([
                        {'params': self.classifier.trainable_parameters()},
                    ], lr=args.meta_learning_rate, amsgrad=False)

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.total_epochs,
                                                              eta_min=self.args.min_learning_rate)

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            print(torch.cuda.device_count())
            if torch.cuda.device_count() > 1:
                self.to(torch.cuda.current_device())
                self.classifier = nn.DataParallel(module=self.classifier)
            else:
                self.to(torch.cuda.current_device())

            self.device = torch.cuda.current_device()


    def get_task_embeddings(self, x_support_set_task, y_support_set_task, names_weights_copy):
        # Use gradients as task embeddings
        support_loss, support_preds, support_latent_features = self.net_forward(x=x_support_set_task,
                                                                                y=y_support_set_task,
                                                                                weights=names_weights_copy,
                                                                                backup_running_statistics=True,
                                                                                training=True, 
                                                                                num_step=0)

        if torch.cuda.device_count() > 1:
            self.classifier.module.zero_grad(names_weights_copy)
        else:
            self.classifier.zero_grad(names_weights_copy)
        grads = torch.autograd.grad(support_loss, names_weights_copy.values(), create_graph=True)

        layerwise_mean_grads = []

        for i in range(len(grads)):
            layerwise_mean_grads.append(grads[i].mean())

        layerwise_mean_grads = torch.stack(layerwise_mean_grads)

        return layerwise_mean_grads

    def attenuate_init(self, task_embeddings, names_weights_copy):
        # Generate attenuation parameters
        gamma = self.attenuator(task_embeddings)
        self.gamma = gamma

        # for fixed gamma
        '''
        try:
            if os.environ['FIXED_GAMMA']:
                gamma = gamma.detach()
                
                gamma[0] = 2.0
                gamma[1] = 1.5
                gamma[2] = 1.0
                gamma[3] = 0.01
                gamma[4] = 0.15
                gamma[5] = 0.15
                
                gamma[0] = 0.5
                gamma[1] = 0.5
                gamma[2] = 0.5
                gamma[3] = 0.5
                gamma[4] = 0.5
                gamma[5] = 0.5
                
                
        except:
            pass
        '''
        #gamma[1] = 0.7414
        #gamma[2] = 0.3875
        #gamma[3] = 0.01286
        #gamma[4] = 0.1459
        #gamma[5] = 0.008303

        updated_names_weights_copy = dict()
        i = 0
        for key in names_weights_copy.keys():
            updated_names_weights_copy[key] = gamma[i] * names_weights_copy[key]
            i+=1

        return updated_names_weights_copy


    def get_per_step_loss_importance_vector(self):
        """
        Generates a tensor of dimensionality (num_inner_loop_steps) indicating the importance of each step's target
        loss towards the optimization loss.
        :return: A tensor to be used to compute the weighted average of the loss, useful for
        the MSL (Multi Step Loss) mechanism.
        """
        loss_weights = np.ones(shape=(self.args.number_of_training_steps_per_iter)) * (
                1.0 / self.args.number_of_training_steps_per_iter)
        decay_rate = 1.0 / self.args.number_of_training_steps_per_iter / self.args.multi_step_loss_num_epochs
        min_value_for_non_final_losses = 0.03 / self.args.number_of_training_steps_per_iter
        for i in range(len(loss_weights) - 1):
            curr_value = np.maximum(loss_weights[i] - (self.current_epoch * decay_rate), min_value_for_non_final_losses)
            loss_weights[i] = curr_value

        curr_value = np.minimum(
            loss_weights[-1] + (self.current_epoch * (self.args.number_of_training_steps_per_iter - 1) * decay_rate),
            1.0 - ((self.args.number_of_training_steps_per_iter - 1) * min_value_for_non_final_losses))
        loss_weights[-1] = curr_value
        loss_weights = torch.Tensor(loss_weights).to(device=self.device)
        return loss_weights

    def get_inner_loop_parameter_dict(self, params):
        """
        Returns a dictionary with the parameters to use for inner loop updates.
        :param params: A dictionary of the network's parameters.
        :return: A dictionary of the parameters to use for the inner loop optimization process.
        """
        param_dict = dict()
        for name, param in params:
            if param.requires_grad:
                if self.args.enable_inner_loop_optimizable_bn_params:
                    param_dict[name] = param.to(device=self.device)
                else:
                    if "norm_layer" not in name:
                        param_dict[name] = param.to(device=self.device)

        return param_dict

    def perturb_weights(self, loss, names_weights_copy, gamma=1e-2):
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        updated_names_weights_copy = dict()
        names_grads_copy = dict()
        grads = torch.autograd.grad(loss, names_weights_copy.values())
        names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
        
        for key, grad in names_grads_copy.items():
            if grad is None:
                print('Grads not found for inner loop parameter', key)
            names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
        
        for key in names_weights_copy.keys():
            grad = names_grads_copy[key]
            weight_norm = names_weights_copy[key].norm()
            grad_norm = grad.norm()
            if grad_norm > gamma * weight_norm:
                grad = grad.div(grad_norm).mul(gamma)
            updated_names_weights_copy[key] = names_weights_copy[key].add(grad)
        return updated_names_weights_copy

    def apply_inner_loop_update(self, loss, names_weights_copy, generated_alpha_params, generated_beta_params, use_second_order, 
                                current_step_idx, lr, names_grads_copy=None, prior=None, l2_lambda=1e-2):
        """
        Applies an inner loop update given current step's loss, the weights to update, a flag indicating whether to use
        second order derivatives and the current step's index.
        :param loss: Current step's loss with respect to the support set.
        :param names_weights_copy: A dictionary with names to parameters to update.
        :param use_second_order: A boolean flag of whether to use second order derivatives.
        :param current_step_idx: Current step's index.
        :return: A dictionary with the updated weights (name, param)
        """
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.classifier.module.zero_grad(params=names_weights_copy)
        else:
            self.classifier.zero_grad(params=names_weights_copy)

        #if prior is not None and current_step_idx > 0:
        #    reg = torch.tensor(0.)
        #    for key in names_weights_copy.keys():
        #        reg = reg + (names_weights_copy[key] - prior[key]).norm()
        #    reg = 1 / reg * l2_lambda
        #    loss = loss + reg

        evaluation = False if not names_grads_copy else True

        if not names_grads_copy:
            grads = torch.autograd.grad(loss, names_weights_copy.values(),
                                        create_graph=use_second_order)
            names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

            for key, grad in names_grads_copy.items():
                if grad is None:
                    print('Grads not found for inner loop parameter', key)
                names_grads_copy[key] = names_grads_copy[key].sum(dim=0)

                if 'linear' in key and self.args.clip_linear_grad:
                    if names_grads_copy[key].norm().item() > self.args.clip_linear_grad:
                        names_grads_copy[key] = names_grads_copy[key] / names_grads_copy[key].norm() * self.args.clip_linear_grad

        names_weights_copy = {key: value[0] for key, value in names_weights_copy.items()}

        if not self.args.alfa:
            if self.args.body_only_update and not evaluation:
                names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                            names_grads_wrt_params_dict=names_grads_copy,
                                                                            num_step=current_step_idx,
                                                                            lr=[lr, lr, lr, lr, 0., 0.])
            names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                         names_grads_wrt_params_dict=names_grads_copy,
                                                                         num_step=current_step_idx,
                                                                         lr=lr)

        else:
            names_weights_copy = self.inner_loop_optimizer.update_params(names_weights_dict=names_weights_copy,
                                                                        names_grads_wrt_params_dict=names_grads_copy,
                                                                        generated_alpha_params=generated_alpha_params,
                                                                        generated_beta_params=generated_beta_params,
                                                                        num_step=current_step_idx)

        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
        names_weights_copy = {
            name.replace('module.', ''): value.unsqueeze(0).repeat(
                [num_devices] + [1 for i in range(len(value.shape))]) for
            name, value in names_weights_copy.items()}

        return names_weights_copy, names_grads_copy

    def get_across_task_loss_metrics(self, total_losses, total_accuracies):
        losses = dict()

        losses['loss'] = torch.mean(torch.stack(total_losses))
        losses['accuracy'] = np.mean(total_accuracies)

        return losses

    def forward(self, data_batch, epoch, use_second_order, use_multi_step_loss_optimization, num_steps, training_phase, current_iter):
        """
        Runs a forward outer loop pass on the batch of tasks using the MAML/++ framework.
        :param data_batch: A data batch containing the support and target sets.
        :param epoch: Current epoch's index
        :param use_second_order: A boolean saying whether to use second order derivatives.
        :param use_multi_step_loss_optimization: Whether to optimize on the outer loop using just the last step's
        target loss (True) or whether to use multi step loss which improves the stability of the system (False)
        :param num_steps: Number of inner loop steps.
        :param training_phase: Whether this is a training phase (True) or an evaluation phase (False)
        :return: A dictionary with the collected losses of the current outer forward propagation.
        """
        if training_phase:
            landscape_logger = self.train_landscape_logger
        else:
            landscape_logger = self.val_landscape_logger

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        [b, ncs, spc] = y_support_set.shape

        self.num_classes_per_set = ncs

        total_losses = []
        total_accuracies = []
        per_task_target_preds = [[] for i in range(len(x_target_set))]
        per_task_target_latent_features = [[] for i in range(len(x_target_set))]

        if torch.cuda.device_count() > 1:
            self.classifier.module.zero_grad()
        else:
            self.classifier.zero_grad()
        for task_id, (x_support_set_task, y_support_set_task, x_target_set_task, y_target_set_task) in \
                enumerate(zip(x_support_set,
                              y_support_set,
                              x_target_set,
                              y_target_set)):
            adaptation_losses = []
            grads_norm = []
            weights_norm = []
            updated_norm = []
            task_losses = []
            task_accuracies = []
            per_step_support_accuracy = []
            per_step_target_accuracy = []
            per_step_loss_importance_vectors = self.get_per_step_loss_importance_vector()
            names_weights_copy = self.get_inner_loop_parameter_dict(self.classifier.named_parameters())
            num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

            names_weights_copy = {
                name.replace('module.', ''): value.unsqueeze(0).repeat(
                    [num_devices] + [1 for i in range(len(value.shape))]) for
                name, value in names_weights_copy.items()}


            n, s, c, h, w = x_target_set_task.shape

            x_support_set_task = x_support_set_task.view(-1, c, h, w)
            y_support_set_task = y_support_set_task.view(-1)
            x_target_set_task = x_target_set_task.view(-1, c, h, w)
            y_target_set_task = y_target_set_task.view(-1)

            # Attenuate the initialization for L2F
            if self.args.attenuate:
                # Obtain gradients from support set for task embedding
                task_embeddings = self.get_task_embeddings(x_support_set_task=x_support_set_task,
                                                           y_support_set_task=y_support_set_task,
                                                           names_weights_copy=names_weights_copy)
                
                names_weights_copy = self.attenuate_init(task_embeddings=task_embeddings,
                                                         names_weights_copy=names_weights_copy)

            prior_hypothesis = {key: value for key, value in names_weights_copy.items()}
            for num_step in range(num_steps):
                losses = []
                grads = []
                generated_alpha_params = {}
                generated_beta_params = {}

                if self.args.alfa:

                    support_loss_grad = torch.autograd.grad(support_loss, names_weights_copy.values(), retain_graph=True)
                    per_step_task_embedding = []
                    for k, v in names_weights_copy.items():
                        per_step_task_embedding.append(v.mean())
                
                    for i in range(len(support_loss_grad)):
                        per_step_task_embedding.append(support_loss_grad[i].mean())

                    per_step_task_embedding = torch.stack(per_step_task_embedding)

                    generated_params = self.regularizer(per_step_task_embedding)
                    num_layers = len(names_weights_copy)

                    generated_alpha, generated_beta = torch.split(generated_params, split_size_or_sections=num_layers)
                    g = 0
                    for key in names_weights_copy.keys():
                        generated_alpha_params[key] = generated_alpha[g]
                        generated_beta_params[key] = generated_beta[g]
                        g+=1
                    
                support_loss, support_preds, support_latent_features = self.net_forward(x=x_support_set_task,
                                                                                        y=y_support_set_task,
                                                                                        weights=names_weights_copy,
                                                                                        backup_running_statistics=
                                                                                        True if (num_step == 0) else False,
                                                                                        training=True, 
                                                                                        num_step=num_step)
                    
                names_weights_copy_updated, origin_grad = self.apply_inner_loop_update(loss=support_loss,
                                                                  names_weights_copy=names_weights_copy,
                                                                  generated_beta_params=generated_beta_params,
                                                                  generated_alpha_params=generated_alpha_params,
                                                                  use_second_order=use_second_order,
                                                                  current_step_idx=num_step,
                                                                  lr=self.task_learning_rate,
                                                                  prior=prior_hypothesis)
                
                grads_norm.append([grad.norm().item() for grad in origin_grad.values()])
                weights_norm.append([weight.norm().item() for weight in names_weights_copy.values()])
                updated_norm.append(self.inner_loop_optimizer.updated_norm)

                for coeff in np.arange(0.5, 4.5, 0.5):
                    lr = coeff * self.task_learning_rate
                    names_weights_copy_, _ = self.apply_inner_loop_update(loss=support_loss,
                                                                    names_weights_copy=names_weights_copy,
                                                                    generated_beta_params=generated_beta_params,
                                                                    generated_alpha_params=generated_alpha_params,
                                                                    use_second_order=use_second_order,
                                                                    current_step_idx=num_step,
                                                                    lr=lr,
                                                                    names_grads_copy=origin_grad)

                    support_loss_, support_preds_, support_latent_features_ = self.net_forward(x=x_support_set_task,
                                                                                            y=y_support_set_task,
                                                                                            weights=names_weights_copy_,
                                                                                            backup_running_statistics=
                                                                                            True if (num_step == 0) else False,
                                                                                            training=False,
                                                                                            num_step=num_step)
                    landscape_logger(support_loss_, names_weights_copy_, origin_grad, lr, num_step)
                names_weights_copy = names_weights_copy_updated
                
                adaptation_losses.append(support_loss)

                if use_multi_step_loss_optimization and training_phase and epoch < self.args.multi_step_loss_num_epochs:
                    target_loss, target_preds, target_latent_features = self.net_forward(x=x_target_set_task,
                                                                                         y=y_target_set_task, 
                                                                                         weights=names_weights_copy,
                                                                                         backup_running_statistics=False, 
                                                                                         training=True,
                                                                                         num_step=num_step)
                    
                    task_losses.append(per_step_loss_importance_vectors[num_step] * target_loss)

                else:
                    if num_step == (self.args.number_of_training_steps_per_iter - 1):
                        target_loss, target_preds, target_latent_features = self.net_forward(x=x_target_set_task,
                                                                                             y=y_target_set_task, 
                                                                                             weights=names_weights_copy,
                                                                                             backup_running_statistics=False, 
                                                                                             training=True,
                                                                                             num_step=num_step)

                        task_losses.append(target_loss)
            landscape_logger.num_tasks += 1

            #per_task_target_preds[task_id] = target_preds.detach().cpu().numpy()
            per_task_target_preds[task_id] = torch.nn.functional.softmax(target_preds.detach().cpu(), dim=1).numpy()
            per_task_target_latent_features[task_id] = target_latent_features.detach().cpu().numpy()
            _, predicted = torch.max(target_preds.data, 1)

            accuracy = predicted.float().eq(y_target_set_task.data.float()).cpu().float()
            task_losses = torch.sum(torch.stack(task_losses))
            total_losses.append(task_losses)
            total_accuracies.extend(accuracy)

            try:
                if int(os.environ['LOG_EMBEDDING']):
                    #self.tsne_logger.run(np.array(per_task_target_latent_features[0]), y_target_set_task.detach().cpu().numpy(), True)
                    #self.tsne_logger.run(np.array(per_task_target_latent_features[0]), predicted.detach().cpu().numpy(), True)
                    self.pca_logger.run(np.array(per_task_target_preds[0]), y_target_set_task.detach().cpu().numpy(), True)
            except KeyError:
                pass

            if not training_phase:
                if torch.cuda.device_count() > 1:
                    self.classifier.module.restore_backup_stats()
                else:
                    self.classifier.restore_backup_stats()
        
        losses = self.get_across_task_loss_metrics(total_losses=total_losses,
                                                   total_accuracies=total_accuracies)

        if current_iter != 'test':
            landscape_logger.get_result(current_iter)

        for idx, item in enumerate(per_step_loss_importance_vectors):
            losses['loss_importance_vector_{}'.format(idx)] = item.detach().cpu().numpy()

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        '''
        probs = []
        true_probs = []
        for i, (logit, c) in enumerate(zip(per_task_target_preds[0], y_target_set_task.tolist())):
            #print(i, logit, c, np.argmax(logit), True if c == np.argmax(logit) else False, softmax(logit))
            if c != np.argmax(logit):
                probs.append(softmax(logit)[np.argmax(logit)])
                true_probs.append(softmax(logit)[c])
        fig, (ax1, ax2) = plt.subplots(1, 2)
        plt.plot(list(map(np.log, probs)), color='red')
        plt.plot(list(map(np.log, true_probs)), color='blue')
        plt.show()
        '''
        
        losses['adaptation_loss'] = torch.mean(torch.stack(adaptation_losses))
        losses['adaptation_std'] = torch.std(torch.stack(adaptation_losses))
        
        grads_stats = []
        weights_stats = []
        updated_stats = []
        grads_norm = np.array(grads_norm)
        weights_norm = np.array(weights_norm)
        updated_norm = np.array(updated_norm)

        for i in range(grads_norm.shape[1]):
            grads_stats.append(np.min(grads_norm[:, i]))
            grads_stats.append(np.max(grads_norm[:, i]))
            weights_stats.append(np.min(weights_norm[:, i]))
            weights_stats.append(np.max(weights_norm[:, i]))
            updated_stats.append(np.min(updated_norm[:, i]))
            updated_stats.append(np.max(updated_norm[:, i]))
        
        grads_norm = np.mean(grads_norm, axis=0)
        weights_norm = np.mean(weights_norm, axis=0)
        updated_norm = np.mean(updated_norm, axis=0)
        return losses, per_task_target_preds, total_losses, (grads_norm, weights_norm, updated_norm, grads_stats, weights_stats, updated_stats)

    def net_forward(self, x, y, weights, backup_running_statistics, training, num_step):
        """
        A base model forward pass on some data points x. Using the parameters in the weights dictionary. Also requires
        boolean flags indicating whether to reset the running statistics at the end of the run (if at evaluation phase).
        A flag indicating whether this is the training session and an int indicating the current step's number in the
        inner loop.
        :param x: A data batch of shape b, c, h, w
        :param y: A data targets batch of shape b, n_classes
        :param weights: A dictionary containing the weights to pass to the network.
        :param backup_running_statistics: A flag indicating whether to reset the batch norm running statistics to their
         previous values after the run (only for evaluation)
        :param training: A flag indicating whether the current process phase is a training or evaluation.
        :param num_step: An integer indicating the number of the step in the inner loop.
        :return: the crossentropy losses with respect to the given y, the predictions of the base model.
        """
        preds, latent_feature = self.classifier.forward(x=x, 
                                                        params=weights,
                                                        training=training,
                                                        backup_running_statistics=backup_running_statistics, 
                                                        num_step=num_step)

        loss = F.cross_entropy(input=preds, target=y)

        return loss, preds, latent_feature

    def trainable_parameters(self):
        """
        Returns an iterator over the trainable parameters of the model.
        """
        for param in self.parameters():
            if param.requires_grad:
                yield param

    def train_forward_prop(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds, total_losses, stats = self.forward(data_batch=data_batch, 
                                                                   epoch=epoch,
                                                                   use_second_order=self.args.second_order and
                                                                   epoch > self.args.first_order_to_second_order_epoch,
                                                                   use_multi_step_loss_optimization=self.args.use_multi_step_loss_optimization,
                                                                   num_steps=self.args.number_of_training_steps_per_iter,
                                                                   training_phase=True,
                                                                   current_iter=current_iter)
        return losses, per_task_target_preds, total_losses, stats

    def evaluation_forward_prop(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop evaluation forward prop using the meta-model and base-model.
        :param data_batch: A data batch containing the support set and the target set input, output pairs.
        :param epoch: The index of the currrent epoch.
        :return: A dictionary of losses for the current step.
        """
        losses, per_task_target_preds, total_losses, stats = self.forward(data_batch=data_batch, 
                                                                   epoch=epoch, 
                                                                   use_second_order=False,
                                                                   use_multi_step_loss_optimization=True,
                                                                   num_steps=self.args.number_of_evaluation_steps_per_iter,
                                                                   training_phase=False,
                                                                   current_iter=current_iter)

        return losses, per_task_target_preds

    def meta_update(self, loss, current_iter, total_losses):
        """
        Applies an outer loop update on the meta-parameters of the model.
        :param loss: The current crossentropy loss.
        """
        self.optimizer.zero_grad()
        loss.backward()
        #self.optimizer.pc_backward(total_losses)
        #if 'imagenet' in self.args.dataset_name:
        #    for name, param in self.classifier.named_parameters():
        #        if param.requires_grad:
        #            param.grad.data.clamp_(-10, 10)  # not sure if this is necessary, more experiments are needed
        #for name, param in self.classifier.named_parameters():
        #    print(param.mean())
        self.optimizer.step()

    def run_train_iter(self, data_batch, epoch, current_iter):
        """
        Runs an outer loop update step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """
        epoch = int(epoch)
        if self.current_epoch != epoch:
            self.current_epoch = epoch

        if not self.training:
            self.train()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds, total_losses, stats = self.train_forward_prop(data_batch=data_batch, epoch=epoch, current_iter=current_iter)

        self.meta_update(loss=losses['loss'], current_iter=current_iter, total_losses=total_losses)

        grads_norm, weights_norm, updated_norm, grads_stats, weights_stats, updated_stats = stats

        grads = []
        weights = []
        for name, weight in self.classifier.named_parameters():
            if 'norm_layer' not in name:
                grads.append(weight.grad.norm().item())
                weights.append(weight.norm().item())
        grads_norm = grads_norm.tolist()
        weights_norm = weights_norm.tolist()
        updated_norm = updated_norm.tolist()
        grads_norm.extend(grads)
        weights_norm.extend(weights)

        logger = self.train_landscape_logger
        logger.write_csv_gradients_norm(current_iter, *grads_norm)
        logger.write_csv_weight_norm(current_iter, *weights_norm)
        logger.write_csv_updated_norm(current_iter, *updated_norm)
        logger.write_csv_gradients_norm_min_max(current_iter, *grads_stats)
        logger.write_csv_weight_norm_min_max(current_iter, *weights_stats)
        logger.write_csv_updated_norm_min_max(current_iter, *updated_stats)
        

        losses['learning_rate'] = self.scheduler.get_lr()[0]
        self.optimizer.zero_grad()
        self.zero_grad()
        self.scheduler.step(epoch=epoch)
        
        return losses, per_task_target_preds, self.scheduler.get_last_lr()[0]

    def run_validation_iter(self, data_batch, current_iter):
        """
        Runs an outer loop evaluation step on the meta-model's parameters.
        :param data_batch: input data batch containing the support set and target set input, output pairs
        :param epoch: the index of the current epoch
        :return: The losses of the ran iteration.
        """

        if self.training:
            self.eval()

        x_support_set, x_target_set, y_support_set, y_target_set = data_batch

        x_support_set = torch.Tensor(x_support_set).float().to(device=self.device)
        x_target_set = torch.Tensor(x_target_set).float().to(device=self.device)
        y_support_set = torch.Tensor(y_support_set).long().to(device=self.device)
        y_target_set = torch.Tensor(y_target_set).long().to(device=self.device)

        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_target_preds = self.evaluation_forward_prop(data_batch=data_batch, epoch=self.current_epoch, current_iter=current_iter)

        losses['loss'].backward() # uncomment if you get the weird memory error
        self.zero_grad()
        self.optimizer.zero_grad()

        return losses, per_task_target_preds

    def save_model(self, model_save_dir, state):
        """
        Save the network parameter state and experiment state dictionary.
        :param model_save_dir: The directory to store the state at.
        :param state: The state containing the experiment state and the network. It's in the form of a dictionary
        object.
        """
        state['network'] = self.state_dict()
        torch.save(state, f=model_save_dir)

    def load_model(self, model_save_dir, model_name, model_idx):
        """
        Load checkpoint and return the state dictionary containing the network state params and experiment state.
        :param model_save_dir: The directory from which to load the files.
        :param model_name: The model_name to be loaded from the direcotry.
        :param model_idx: The index of the model (i.e. epoch number or 'latest' for the latest saved model of the current
        experiment)
        :return: A dictionary containing the experiment state and the saved model parameters.
        """
        filepath = os.path.join(model_save_dir, "{}_{}".format(model_name, model_idx))
        state = torch.load(filepath)
        state_dict_loaded = state['network']
        self.load_state_dict(state_dict=state_dict_loaded)
        return state
