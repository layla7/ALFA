import tqdm.autonotebook as tqdm
import os, glob, shutil, pickle
import numpy as np
import sys
from utils.storage import build_experiment_folder, save_statistics, save_to_json
import time
import torch
import wandb
from torch.utils.tensorboard import SummaryWriter
#from matplotlib import pyplot as plt


class ExperimentBuilder(object):
    def __init__(self, args, data, model, device):
        """
        Initializes an experiment builder using a named tuple (args), a data provider (data), a meta learning system
        (model) and a device (e.g. gpu/cpu/n)
        :param args: A namedtuple containing all experiment hyperparameters
        :param data: A data provider of instance MetaLearningSystemDataLoader
        :param model: A meta learning system instance
        :param device: Device/s to use for the experiment
        """
        self.args, self.device = args, device
        self.model = model

        project_name = args.dataset_name + '-' + args.backbone+'-{}way-{}shot'.format(args.num_classes_per_set, args.num_samples_per_class)
        if args.wandb:
            try:
                wandb_id = os.environ['WANDB_RUN_ID']
            except KeyError:
                wandb_id = wandb.util.generate_id()

            wandb.init(
                project=project_name,
                config=vars(args),
                id=wandb_id,
                resume='allow'
            )
            wandb.run.name = args.wandb_run_name
            wandb.run.save()
            wandb.watch(self.model)

            self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
                #experiment_name='experiments/'+project_name+'/'+self.args.experiment_name+'_{}'.format(wandb_id))
                experiment_name='experiments/'+project_name+'/'+self.args.wandb_run_name+'_{}'.format(wandb_id))
        else:
            try:
                wandb_id = os.environ['WANDB_RUN_ID']
                self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
                    #experiment_name='experiments/'+project_name+'/'+self.args.experiment_name+'_{}'.format(wandb_id))
                    experiment_name='experiments/'+project_name+'/'+self.args.wandb_run_name+'_{}'.format(wandb_id))
            except KeyError:
                self.saved_models_filepath, self.logs_filepath, self.samples_filepath = build_experiment_folder(
                    #experiment_name='experiments/'+project_name+'/'+self.args.experiment_name)
                    experiment_name='experiments/'+project_name+'/'+self.args.wandb_run_name)

        self.model.logs_filepath = self.logs_filepath

        copy_target = ['experiment_config', 'experiment_scripts', 'utils', 'data.py', 'experiment_builder.py',
                       'few_shot_learning_system.py', 'inner_loop_optimizers.py', 'meta_neural_network_architectures.py', 
                       'train_maml_system.py']
        backup_filepath = self.logs_filepath.replace('logs', 'backup')
        for item in copy_target:
            if os.path.isdir(item):
                target_path = backup_filepath + '/{}'.format(item)
                os.makedirs(target_path, exist_ok=True)
                shutil.copytree(item, target_path, dirs_exist_ok=True)
            else:
                shutil.copy2(item, backup_filepath + '/{}'.format(item))
            
        self.tb_writer = SummaryWriter(self.logs_filepath+'/tensorboard')

        self.total_losses = dict()
        self.state = dict()
        self.state['best_val_acc'] = 0.
        self.state['best_val_iter'] = 0
        self.state['current_iter'] = 0
        self.state['current_iter'] = 0
        self.start_epoch = 0
        self.max_models_to_save = self.args.max_models_to_save
        self.create_summary_csv = False

        experiment_path = os.path.abspath(self.args.experiment_name)
        exp_name = experiment_path.split('/')[-1]
        log_base_dir = 'logs'
        os.makedirs(log_base_dir, exist_ok=True)

        log_dir = os.path.join(log_base_dir, exp_name)
        print(log_dir)

        if self.args.continue_from_epoch == 'from_scratch':
            self.create_summary_csv = True

        elif self.args.continue_from_epoch == 'latest':
            checkpoint = os.path.join(self.saved_models_filepath, "train_model_latest")
            print("attempting to find existing checkpoint", )
            if os.path.exists(checkpoint):
                self.state = \
                    self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                          model_idx='latest')
                self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

            else:
                self.args.continue_from_epoch = 'from_scratch'
                self.create_summary_csv = True
        elif int(self.args.continue_from_epoch) >= 0:
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=self.args.continue_from_epoch)
            self.start_epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)

        self.data = data(args=args, current_iter=self.state['current_iter'])

        print("train_seed {}, val_seed: {}, at start time".format(self.data.dataset.seed["train"],
                                                                  self.data.dataset.seed["val"]))
        self.total_epochs_before_pause = self.args.total_epochs_before_pause
        self.state['best_epoch'] = int(self.state['best_val_iter'] / self.args.total_iter_per_epoch)
        self.epoch = int(self.state['current_iter'] / self.args.total_iter_per_epoch)
        self.augment_flag = True if 'omniglot' in self.args.dataset_name.lower() else False
        self.start_time = time.time()
        self.epochs_done_in_this_run = 0
        print(self.state['current_iter'], int(self.args.total_iter_per_epoch * self.args.total_epochs))

    def build_summary_dict(self, total_losses, phase, summary_losses=None):
        """
        Builds/Updates a summary dict directly from the metric dict of the current iteration.
        :param total_losses: Current dict with total losses (not aggregations) from experiment
        :param phase: Current training phase
        :param summary_losses: Current summarised (aggregated/summarised) losses stats means, stdv etc.
        :return: A new summary dict with the updated summary statistics information.
        """
        if summary_losses is None:
            summary_losses = dict()

        for key in total_losses:
            summary_losses["{}_{}_mean".format(phase, key)] = np.mean(total_losses[key])
            summary_losses["{}_{}_std".format(phase, key)] = np.std(total_losses[key])

        return summary_losses

    def build_loss_summary_string(self, summary_losses):
        """
        Builds a progress bar summary string given current summary losses dictionary
        :param summary_losses: Current summary statistics
        :return: A summary string ready to be shown to humans.
        """
        output_update = ""
        for key, value in zip(list(summary_losses.keys()), list(summary_losses.values())):
            if "loss" in key or "accuracy" in key:
                value = float(value)
                output_update += "{}: {:.4f}, ".format(key, value)

        return output_update

    def merge_two_dicts(self, first_dict, second_dict):
        """Given two dicts, merge them into a new dict as a shallow copy."""
        z = first_dict.copy()
        z.update(second_dict)
        return z

    def train_iteration(self, train_sample, sample_idx, epoch_idx, total_losses, current_iter, pbar_train):
        """
        Runs a training iteration, updates the progress bar and returns the total and current epoch train losses.
        :param train_sample: A sample from the data provider
        :param sample_idx: The index of the incoming sample, in relation to the current training run.
        :param epoch_idx: The epoch index.
        :param total_losses: The current total losses dictionary to be updated.
        :param current_iter: The current training iteration in relation to the whole experiment.
        :param pbar_train: The progress bar of the training.
        :return: Updates total_losses, train_losses, current_iter
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = train_sample
        data_batch = (x_support_set, x_target_set, y_support_set, y_target_set)

        if sample_idx == 0:
            print("shape of data", x_support_set.shape, x_target_set.shape, y_support_set.shape,
                  y_target_set.shape)

        losses, _, current_lr = self.model.run_train_iter(data_batch=data_batch, epoch=epoch_idx, current_iter=self.state['current_iter'])

        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        train_losses = self.build_summary_dict(total_losses=total_losses, phase="train")
        train_output_update = self.build_loss_summary_string(losses)

        pbar_train.update(1)
        #pbar_train.set_description("training phase {} -> {}".format(self.epoch, train_output_update))

        current_iter += 1

        return train_losses, total_losses, current_iter, current_lr

    def evaluation_iteration(self, val_sample, total_losses, pbar_val, phase, current_iter):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_val: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        '''
        def hook_fn_wandb_log(m, i, o):
            wandb.log({'attention': wandb.Image(x_target_set * o)})

        hooks = {}
        def hook(net, hook_fn):
            for name, layer in net._modules.items():
                for name, layer2 in layer.items():
                    if 'conv' in name:
                        hooks[name+'_channel'] = layer2.attention_layer.channel_attention.register_forward_hook(hook_fn)
                        hooks[name+'_spatial'] = layer2.attention_layer.spatial_attention.register_forward_hook(hook_fn)
                    else:
                        continue
        hook(self.model.classifier, hook_fn_wandb_log)
        '''

        losses, _ = self.model.run_validation_iter(data_batch=data_batch, current_iter=current_iter)
        for key, value in zip(list(losses.keys()), list(losses.values())):
            if key not in total_losses:
                total_losses[key] = [float(value)]
            else:
                total_losses[key].append(float(value))

        val_losses = self.build_summary_dict(total_losses=total_losses, phase=phase)
        val_output_update = self.build_loss_summary_string(losses)

        pbar_val.update(1)
        #pbar_val.set_description("val_phase {} -> {}".format(self.epoch, val_output_update))
        #for key, value in hook.values():
        #    print(key)
        #    value.remove()

        return val_losses, total_losses

    def test_evaluation_iteration(self, val_sample, model_idx, sample_idx, per_model_per_batch_preds, pbar_test):
        """
        Runs a validation iteration, updates the progress bar and returns the total and current epoch val losses.
        :param val_sample: A sample from the data provider
        :param total_losses: The current total losses dictionary to be updated.
        :param pbar_test: The progress bar of the val stage.
        :return: The updated val_losses, total_losses
        """
        x_support_set, x_target_set, y_support_set, y_target_set, seed = val_sample
        data_batch = (
            x_support_set, x_target_set, y_support_set, y_target_set)

        losses, per_task_preds = self.model.run_validation_iter(data_batch=data_batch, current_iter='test')

        per_model_per_batch_preds[model_idx].extend(list(per_task_preds))

        test_output_update = self.build_loss_summary_string(losses)

        pbar_test.update(1)
        #pbar_test.set_description("test_phase {} -> {}".format(self.epoch, test_output_update))

        return per_model_per_batch_preds

    def save_models(self, model, epoch, state):
        """
        Saves two separate instances of the current model. One to be kept for history and reloading later and another
        one marked as "latest" to be used by the system for the next epoch training. Useful when the training/val
        process is interrupted or stopped. Leads to fault tolerant training and validation systems that can continue
        from where they left off before.
        :param model: Current meta learning model of any instance within the few_shot_learning_system.py
        :param epoch: Current epoch
        :param state: Current model and experiment state dict.
        """
        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_{}".format(int(epoch))),
                         state=state)

        model.save_model(model_save_dir=os.path.join(self.saved_models_filepath, "train_model_latest"),
                         state=state)

        print("saved models to", self.saved_models_filepath)

    def pack_and_save_metrics(self, start_time, create_summary_csv, train_losses, val_losses, state):
        """
        Given current epochs start_time, train losses, val losses and whether to create a new stats csv file, pack stats
        and save into a statistics csv file. Return a new start time for the new epoch.
        :param start_time: The start time of the current epoch
        :param create_summary_csv: A boolean variable indicating whether to create a new statistics file or
        append results to existing one
        :param train_losses: A dictionary with the current train losses
        :param val_losses: A dictionary with the currrent val loss
        :return: The current time, to be used for the next epoch.
        """
        epoch_summary_losses = self.merge_two_dicts(first_dict=train_losses, second_dict=val_losses)

        if 'per_epoch_statistics' not in state:
            state['per_epoch_statistics'] = dict()

        for key, value in epoch_summary_losses.items():

            if key not in state['per_epoch_statistics']:
                state['per_epoch_statistics'][key] = [value]
            else:
                state['per_epoch_statistics'][key].append(value)

        epoch_summary_string = self.build_loss_summary_string(epoch_summary_losses)
        epoch_summary_losses["epoch"] = self.epoch
        epoch_summary_losses['epoch_run_time'] = time.time() - start_time

        if create_summary_csv:
            self.summary_statistics_filepath = save_statistics(self.logs_filepath, list(epoch_summary_losses.keys()),
                                                               create=True)
            self.create_summary_csv = False

        start_time = time.time()
        print("epoch {} -> {}".format(epoch_summary_losses["epoch"], epoch_summary_string))

        self.summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                           list(epoch_summary_losses.values()))
        return start_time, state

    def evaluated_test_set_using_the_best_models(self, top_n_models):
        per_epoch_statistics = self.state['per_epoch_statistics']
        val_acc = np.copy(per_epoch_statistics['val_accuracy_mean'])
        val_idx = np.array([i for i in range(len(val_acc))])
        sorted_idx = np.argsort(val_acc, axis=0).astype(dtype=np.int32)[::-1][:top_n_models]

        sorted_val_acc = val_acc[sorted_idx]
        val_idx = val_idx[sorted_idx]
        print(sorted_idx)
        print(sorted_val_acc)

        top_n_idx = val_idx[:top_n_models]
        per_model_per_batch_preds = [[] for i in range(top_n_models)]
        per_model_per_batch_targets = [[] for i in range(top_n_models)]
        per_model_per_batch_gammas = [[] for i in range(top_n_models)]
        test_losses = [dict() for i in range(top_n_models)]
        for idx, model_idx in enumerate(top_n_idx):
            self.state = \
                self.model.load_model(model_save_dir=self.saved_models_filepath, model_name="train_model",
                                      model_idx=model_idx + 1)
            with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_test:
                for sample_idx, test_sample in enumerate(
                        self.data.get_test_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                   augment_images=False)):
                    #print(test_sample[4])
                    per_model_per_batch_targets[idx].extend(np.array(test_sample[3]))
                    per_model_per_batch_preds = self.test_evaluation_iteration(val_sample=test_sample,
                                                                               sample_idx=sample_idx,
                                                                               model_idx=idx,
                                                                               per_model_per_batch_preds=per_model_per_batch_preds,
                                                                               pbar_test=pbar_test)
                    if self.args.attenuate:
                        per_model_per_batch_gammas[idx].append(self.model.gamma.tolist())

            '''
            plt.hist(list(map(np.log, self.model.probs)), bins=[0.1*i for i in range(-35, 1)], 
                     ls='dashed', edgecolor='k', lw=1, color='red', alpha=0.7, label='predicted')
            plt.hist(list(map(np.log, self.model.true_probs)), bins=[0.1*i for i in range(-35, 1)], 
                     ls='dashed', edgecolor='k', lw=1, color='blue', alpha=0.7, label='true target')
            plt.legend(loc='upper left')
            plt.show()
            '''

        # for i in range(top_n_models):
        #     print("test assertion", 0)
        #     print(per_model_per_batch_targets[0], per_model_per_batch_targets[i])
        #     assert np.equal(np.array(per_model_per_batch_targets[0]), np.array(per_model_per_batch_targets[i]))
        
        per_batch_preds = np.mean(per_model_per_batch_preds, axis=0)
        per_batch_max = np.argmax(per_batch_preds, axis=2)
        per_batch_targets = np.array(per_model_per_batch_targets[0]).reshape(per_batch_max.shape)
        accuracy = np.mean(np.equal(per_batch_targets, per_batch_max))
        accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max))

        per_batch_accuracy = np.mean(np.equal(per_batch_targets, per_batch_max), axis=1)
        per_batch_accuracy_std = np.std(np.equal(per_batch_targets, per_batch_max), axis=1)
        print(np.equal(per_batch_targets, per_batch_max).shape)

        if self.args.attenuate:
            per_batch_gammas = np.mean(per_model_per_batch_gammas, axis=0)

        test_losses = {"test_accuracy_mean": accuracy, "test_accuracy_std": accuracy_std}
        dataset_name = self.data.dataset.dataset_name

        _ = save_statistics(self.logs_filepath,
                            list(test_losses.keys()),
                            create=True, filename="test_summary_{}.csv".format(dataset_name))

        summary_statistics_filepath = save_statistics(self.logs_filepath,
                                                      list(test_losses.values()),
                                                      create=False, filename="test_summary_{}.csv".format(dataset_name))

        if self.args.attenuate:
            gamma_key = ['gamma-{}'.format(i) for i in range(len(self.model.gamma.tolist()))]
            save_statistics(self.logs_filepath, 
                            ['task number', 'accuracy', 'std of accuracy', *gamma_key], create=True, 
                            filename='test_summary_per_task_{}.csv'.format(dataset_name))

            for i, (acc, std, gammas) in enumerate(zip(per_batch_accuracy, per_batch_accuracy_std, per_batch_gammas), 1):
                save_statistics(self.logs_filepath, 
                                [i, acc, std, *gammas.tolist()], create=False, 
                                filename='test_summary_per_task_{}.csv'.format(dataset_name))

        print('\n')
        print('test accuracy mean: {:.4f} test accuracy std: {:.4f}'.format(*test_losses.values()))
        print("saved test performance at", summary_statistics_filepath)

    def run_experiment(self):
        """
        Runs a full training experiment with evaluations of the model on the val set at every epoch. Furthermore,
        will return the test set evaluation results on the best performing validation model.
        """
        num_conv_layers = self.model.num_conv_layers
        if self.args.attenuate:
            gammas = {}
            for i in range(num_conv_layers):
                gammas['conv-{}-weight'.format(i)] = []
                #gammas['conv-{}-bias'.format(i)] = []
            gammas['classifier-weight'] = []
            gammas['classifier-bias'] = []

        while (self.state['current_iter'] < (self.args.total_epochs * self.args.total_iter_per_epoch)) and (self.args.evaluate_on_test_set_only == False):
            with tqdm.tqdm(initial=self.state['current_iter'],
                        total=int(self.args.total_iter_per_epoch * self.args.total_epochs)) as pbar_train:

                for train_sample_idx, train_sample in enumerate(
                        self.data.get_train_batches(total_batches=int(self.args.total_iter_per_epoch *
                                                                      self.args.total_epochs) - self.state[
                                                                      'current_iter'],
                                                    augment_images=self.augment_flag)):
                    # print(self.state['current_iter'], (self.args.total_epochs * self.args.total_iter_per_epoch))
                    train_losses, total_losses, self.state['current_iter'], current_lr = self.train_iteration(
                        train_sample=train_sample,
                        total_losses=self.total_losses,
                        epoch_idx=(self.state['current_iter'] /
                                   self.args.total_iter_per_epoch),
                        pbar_train=pbar_train,
                        current_iter=self.state['current_iter'],
                        sample_idx=self.state['current_iter'])

                    if self.args.attenuate:
                        for i in range(num_conv_layers):
                            gammas['conv-{}-weight'.format(i)].append(self.model.gamma[i].item())
                            #gammas['conv-{}-weight'.format(i)].append(self.model.gamma[2*i].item())
                            #gammas['conv-{}-bias'.format(i)].append(self.model.gamma[2*i+1].item())
                        gammas['classifier-weight'].append(self.model.gamma[i+1].item())
                        gammas['classifier-bias'].append(self.model.gamma[i+2].item())

                    if self.state['current_iter'] % self.args.wandb_log_period == 0 and self.args.wandb:
                        wandb.log({'train_loss_mean': train_losses['train_loss_mean'],
                                   'train_loss_std': train_losses['train_loss_std'],
                                   'train_accuracy_mean': train_losses['train_accuracy_mean'],
                                   'train_accuracy_std': train_losses['train_accuracy_std'],
                                   'train_adaptation_loss_mean': train_losses['train_adaptation_loss_mean'],
                                   'train_adaptation_loss_std': train_losses['train_adaptation_loss_std']},
                                  step=self.state['current_iter'])                        

                        if self.args.attenuate:
                            mean = list(map(lambda x: np.mean(x), list(gammas.values())))
                            #weight_mean = mean[::2]
                            #bias_mean = mean[1::2]
                            std = list(map(lambda x: np.std(x), list(gammas.values())))
                            #weight_std = std[::2]
                            #bias_std = std[1::2]

                            for i in range(num_conv_layers):
                                wandb.log({'gamma conv-{}-weight mean'.format(i): mean[i],
                                           'gamma conv-{}-weight std'.format(i): std[i]},
                                          step=self.state['current_iter'])
                            wandb.log({'gamma classifier-weight mean': mean[i+1],
                                       'gamma classifier-weight std': std[i+1]},
                                      step=self.state['current_iter'])
                            wandb.log({'gamma classifier-bias mean': mean[i+2],
                                       'gamma classifier-bias std': std[i+2]},
                                      step=self.state['current_iter'])
                            # for stochastic logging for gamma
                            for value in gammas.values():
                                value.clear()
                            
                    if self.state['current_iter'] % self.args.total_iter_per_epoch == 0:

                        total_losses = dict()
                        val_losses = dict()
                        with tqdm.tqdm(total=int(self.args.num_evaluation_tasks / self.args.batch_size)) as pbar_val:
                            for _, val_sample in enumerate(
                                    self.data.get_val_batches(total_batches=int(self.args.num_evaluation_tasks / self.args.batch_size),
                                                              augment_images=False)):
                                val_losses, total_losses = self.evaluation_iteration(val_sample=val_sample,
                                                                                     total_losses=total_losses,
                                                                                     pbar_val=pbar_val, phase='val',
                                                                                     current_iter=self.state['current_iter'])

                        if val_losses["val_accuracy_mean"] > self.state['best_val_acc']:
                            print("Best validation accuracy", val_losses["val_accuracy_mean"])
                            self.state['best_val_acc'] = val_losses["val_accuracy_mean"]
                            self.state['best_val_iter'] = self.state['current_iter']
                            self.state['best_epoch'] = int(
                                self.state['best_val_iter'] / self.args.total_iter_per_epoch)

                        if self.args.wandb:
                            wandb.log({'val_accuracy_mean': val_losses['val_accuracy_mean'],
                                    'val_accuracy_std': val_losses['val_accuracy_std'],
                                    'best_val_acc': self.state['best_val_acc']},
                                    step=self.state['current_iter'])

                        self.epoch += 1
                        self.state = self.merge_two_dicts(first_dict=self.merge_two_dicts(first_dict=self.state,
                                                                                          second_dict=train_losses),
                                                          second_dict=val_losses)

                        self.save_models(model=self.model, epoch=self.epoch, state=self.state)

                        self.start_time, self.state = self.pack_and_save_metrics(start_time=self.start_time,
                                                                                 create_summary_csv=self.create_summary_csv,
                                                                                 train_losses=train_losses,
                                                                                 val_losses=val_losses,
                                                                                 state=self.state)

                        self.total_losses = dict()

                        self.epochs_done_in_this_run += 1

                        save_to_json(filename=os.path.join(self.logs_filepath, "summary_statistics.json"),
                                     dict_to_store=self.state['per_epoch_statistics'])

                        if self.epochs_done_in_this_run >= self.total_epochs_before_pause:
                            print("train_seed {}, val_seed: {}, at pause time".format(self.data.dataset.seed["train"],
                                                                                      self.data.dataset.seed["val"]))
                            #sys.exit()
        self.evaluated_test_set_using_the_best_models(top_n_models=5)
        d = {'train_total_min': self.model.train_total_min_loss,
             'train_total_max': self.model.train_total_max_loss,
             'val_total_min': self.model.val_total_min_loss,
             'val_total_max': self.model.val_total_max_loss}
        with open('landscape/landscape_{}.pkl'.format(os.environ['LANDSCAPE']), 'wb') as f:
            pickle.dump(d, f)
