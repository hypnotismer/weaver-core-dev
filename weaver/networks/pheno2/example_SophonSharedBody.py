import numpy as np
import awkward as ak
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import tqdm
import time
import os
from collections import defaultdict, Counter

import sklearn.metrics as m
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')

from utils.logger import _logger
from utils.nn.tools import (
    _concat,
    AllGather,
)
from utils.import_tools import import_module

ParticleTransformer = import_module(os.path.join(os.path.dirname(__file__), '../ParticleTransformer2024Plus.py'), 'ParT').ParticleTransformer

'''
This code is modified from example_Sophon.py
 - add freeze_mode
 - define FC layer outside of the main ParT body
 - allow custom merge_after_nth_layer
 - new eval/test utilities: custom BkgRej maker; custom label_cls_nodes and label_stored for saving outpout nodes
'''

def apply_sequential(module_list, x):
    for m in module_list:
        x = m(x)
    return x

def ffn_layers(input_dim=None, output_dim=None, fc_params=[], bias_last=True):
    layers = nn.ModuleList()
    in_dim = input_dim
    for out_dim, drop_rate in fc_params:
        layers.append(nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(drop_rate)))
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, output_dim, bias=bias_last))
    return layers


class ParticleTransformerSophonSharedBodyWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.export_embed = kwargs.pop('export_embed', False)
        self.merge_after_nth_layer = kwargs.pop('merge_after_nth_layer', -1)
        self.freeze_mode = kwargs.pop('freeze_mode', False)

        fc_params = kwargs.get('fc_params', None)
        kwargs['fc_params'] = None
        self.mod = ParticleTransformer(**kwargs)
        self.fc_layers = ffn_layers(input_dim=kwargs['embed_dims'][-1], output_dim=kwargs['num_classes'], fc_params=fc_params, bias_last=True)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, *args):
        # return self.mod(features, v=lorentz_vectors, mask=mask) # not using the default foward implementation. Should add emport_embed flag
        assert len(args) % 4 == 0, "Input should be groups of {points, features, vectors, mask}"
        n = len(args) // 4

        if self.freeze_mode:
            # force eval mode (parameters all frozen, no batchnorm running stat, no dropout)
            self.mod.eval()

        # mod should return x_cls since fc_params is set to None
        x_cls = []
        for i in range(n):
            points, features, lorentz_vectors, mask = args[i*4:(i+1)*4]
            x_cls.append(self.mod(features, v=lorentz_vectors, mask=mask))

        x = torch.stack(x_cls, dim=1) # (bsz, n, hidden_dim)

        x = apply_sequential(self.fc_layers[:(self.merge_after_nth_layer+1)], x)
        x = x.mean(dim=1) # (bsz, hidden_dim)
        output = apply_sequential(self.fc_layers[(self.merge_after_nth_layer+1):], x)

        if self.mod.for_inference:
            output = torch.softmax(output, dim=1)

        return output


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=[len(v) for k, v in data_config.input_dicts.items() if k.endswith('pf_features')][0],
        num_classes=None,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=True,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    # remove the eval/test-time related options from cfg
    eval_kw = cfg.pop('eval_kw', dict())
    cfg.pop('label_cls_nodes', None)
    cfg.pop('label_stored', None)

    model = ParticleTransformerSophonSharedBodyWrapper(**cfg)
    
    # set eval_kw for ROC curve configuration
    model.eval_kw = eval_kw

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


def get_train_fn(data_config, **kwargs):
    return train_classification_sophon


def get_evaluate_fn(data_config, **kwargs):
    return evaluate_classification_sophon


def get_save_fn(data_config, **kwargs):
    return save_classification_sophon


# Customized training and evaluation functions for Sophon
# functions are adapted from https://github.com/hqucms/weaver-core/blob/main/weaver/utils/nn/tools.py

def train_classification_sophon(
        model, loss_func, opt, scheduler, train_loader, dev, epoch, steps_per_epoch=None, grad_scaler=None,
        tb_helper=None, extra_args=None):
    model.train()

    data_config = train_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    start_time = time.time()
    with tqdm.tqdm(train_loader) as tq:
        for X, y, _ in tq:
            inputs = [X[k].to(dev) for k in data_config.input_names]
            label = y[data_config.label_names[0]].long().to(dev) # label is obtained from inputs
            entry_count += label.shape[0]
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=grad_scaler is not None):
                logits = model(*inputs)
                loss = loss_func(logits, label)
            if grad_scaler is None:
                loss.backward()
                opt.step()
            else:
                grad_scaler.scale(loss).backward()
                grad_scaler.step(opt)
                grad_scaler.update()

            if scheduler and getattr(scheduler, '_update_per_step', False):
                scheduler.step()

            _, preds = logits.max(1)
            loss = loss.item()

            num_examples = label.shape[0]
            label_counter.update(label.numpy(force=True))
            num_batches += 1
            count += num_examples
            correct = (preds == label).sum().item()
            total_loss += loss
            total_correct += correct

            tq.set_postfix({
                'lr': '%.2e' % scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'],
                'Loss': '%.5f' % loss,
                'AvgLoss': '%.5f' % (total_loss / num_batches),
                'Acc': '%.5f' % (correct / num_examples),
                'AvgAcc': '%.5f' % (total_correct / count)})

            if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Train AvgLoss: %.5f, AvgAcc: %.5f' % (total_loss / num_batches, total_correct / count))
    _logger.info('Train class distribution: \n    %s', str(sorted(label_counter.items())))
    _logger.info('Max CUDA memory: %.1f MB' % (torch.cuda.max_memory_allocated(dev) / 1024.**2,))

    if tb_helper:
        tb_helper.write_scalars([
            ("Loss/train (epoch)", total_loss / num_batches, epoch),
            ("Acc/train (epoch)", total_correct / count, epoch),
            ("lr/train (epoch)", scheduler.get_last_lr()[0] if scheduler else opt.defaults['lr'], epoch),
        ])

        # # customization: store hyperparameters
        # convert_to_str = lambda x: str(x) if not isinstance(x, (int, float, bool)) else x
        # if epoch == 0:
        #     tb_helper.writer.add_hparams({k: convert_to_str(v) for k, v in model.kwargs.items()}, {})

        # update the batch state
        tb_helper.batch_train_count += num_batches

    if scheduler and not getattr(scheduler, '_update_per_step', False):
        scheduler.step()


def evaluate_classification_sophon(model, test_loader, dev, epoch, for_training=True, loss_func=None, steps_per_epoch=None,
                            eval_metrics=['roc_auc_score', 'roc_auc_score_matrix', 'confusion_matrix'],
                            tb_helper=None, extra_args=None):
    model.eval()

    data_config = test_loader.dataset.config

    label_counter = Counter()
    total_loss = 0
    num_batches = 0
    total_correct = 0
    entry_count = 0
    count = 0
    scores = []
    labels = defaultdict(list)
    labels_counts = []
    observers = defaultdict(list)
    start_time = time.time()
    eval_kw = model.module.eval_kw \
        if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)) else model.eval_kw
    with torch.no_grad():
        with tqdm.tqdm(test_loader) as tq:
            for X, y, Z in tq:
                # X, y: torch.Tensor; Z: ak.Array
                inputs = [X[k].to(dev) for k in data_config.input_names]
                y = {k: AllGather.apply(v.to(dev)) for k, v in y.items()}
                label = y[data_config.label_names[0]].long().to(dev)
                entry_count += label.shape[0]
                logits = AllGather.apply(model(*inputs))
                scores.append(torch.softmax(logits.float(), dim=1).numpy(force=True))

                for k, v in y.items():
                    labels[k].append(v.numpy(force=True))
                if not for_training:
                    for k, v in Z.items():
                        observers[k].append(v.numpy(force=True))

                num_examples = label.shape[0]
                label_counter.update(label.numpy(force=True))

                _, preds = logits.max(1)
                loss = 0 if loss_func is None else loss_func(logits, label).item()

                num_batches += 1
                count += num_examples
                correct = (preds == label).sum().item()
                total_loss += loss * num_examples
                total_correct += correct

                tq.set_postfix({
                    'Loss': '%.5f' % loss,
                    'AvgLoss': '%.5f' % (total_loss / count),
                    'Acc': '%.5f' % (correct / num_examples),
                    'AvgAcc': '%.5f' % (total_correct / count)})

                if steps_per_epoch is not None and num_batches >= steps_per_epoch:
                    break

    time_diff = time.time() - start_time
    _logger.info('Processed %d entries in total (avg. speed %.1f entries/s)' % (entry_count, entry_count / time_diff))
    _logger.info('Evaluation class distribution: \n    %s', str(sorted(label_counter.items())))

    if tb_helper:
        tb_mode = 'eval' if for_training else 'test'
        tb_helper.write_scalars([
            ("Loss/%s (epoch)" % tb_mode, total_loss / count, epoch),
            ("Acc/%s (epoch)" % tb_mode, total_correct / count, epoch),
        ])

    scores = np.concatenate(scores)
    labels = {k: _concat(v) for k, v in labels.items()}

    # customized evaluation: making ROC curves for tensorboard monitoring
    if tb_helper and for_training:
        truth_label = labels['truth_label']
        scores_dict, flag_dict = {}, {}
        
        # Default ROC curve configuration for cls_0, cls_1, cls_2
        roc_kwargs_default = {
            'label_inds_map': {
                'cls_0': [0],
                'cls_1': [1], 
                'cls_2': [2],
            },
            'comp_list': [('cls_1', 'cls_0'), ('cls_2', 'cls_0')] # ROC curves for A vs B
        }
        
        # Use provided roc_kw or fall back to default
        roc_kwargs = eval_kw.get('roc_kw', roc_kwargs_default)
        
        for name, inds in roc_kwargs.get('label_inds_map').items():
            flag_dict[name] = np.any([truth_label == i for i in inds], axis=0)
            scores_dict[name] = np.sum(scores[:, inds], axis=1)
            print(name, flag_dict[name].shape, scores_dict[name].shape)
        comp_list = roc_kwargs.get('comp_list') # e.g. [('Xbb', 'QCD'), ('Xcc', 'QCD'), ('Xcc', 'Xbb')] # ROC curves for A vs B
        
        bkgrej = {}

        f, ax = plt.subplots(figsize=(5, 5))
        ax.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), linestyle='--', color='gray', label='Random guess')

        for name_sig, name_bkg in comp_list:
            discr = scores_dict[name_sig] / (scores_dict[name_sig] + scores_dict[name_bkg])
            discr_sig, discr_bkg = discr[flag_dict[name_sig]], discr[flag_dict[name_bkg]]
            fpr, tpr, _ = m.roc_curve(
                np.concatenate([np.ones_like(discr_sig), np.zeros_like(discr_bkg)]),
                np.concatenate([discr_sig, discr_bkg])
            )
            ax.plot(tpr, fpr, label='%s vs %s (AUC=%.4f)' % (name_sig, name_bkg, m.auc(fpr, tpr)))
            bkgrej[(name_sig, name_bkg)] = np.interp(0.3, tpr, 1. / np.maximum(fpr, 1e-10)) # bkgrej at eff_sig=30%
        ax.legend()
        ax.set_xlabel('True positive rate (signal eff.)', ha='right', x=1.0); ax.set_ylabel('False positive rate (BKG eff.)', ha='right', y=1.0)
        ax.set_xlim(0, 1); ax.set_ylim(1e-4, 1), ax.set_yscale('log')

        # write ROC curve figure
        tb_helper.writer.add_figure('ROC/%s/epoch%s' % (tb_mode, str(epoch).zfill(4)), f)

        # write bkgrej values
        for name_sig, name_bkg in comp_list:
            tb_helper.write_scalars([
                ('BkgRej_%s_vs_%s/%s (epoch)' % (name_sig, name_bkg, tb_mode), bkgrej[(name_sig, name_bkg)], epoch),
            ])


    if for_training:
        return total_correct / count
    else:
        # convert 2D labels/scores
        if len(scores) != entry_count:
            if len(labels_counts):
                labels_counts = np.concatenate(labels_counts)
                scores = ak.unflatten(scores, labels_counts)
                for k, v in labels.items():
                    labels[k] = ak.unflatten(v, labels_counts)
            else:
                assert (count % entry_count == 0)
                scores = scores.reshape((entry_count, int(count / entry_count), -1)).transpose((1, 2))
                for k, v in labels.items():
                    labels[k] = v.reshape((entry_count, -1))
        observers = {k: _concat(v) for k, v in observers.items()}
        return total_correct / count, scores, labels, observers


def save_classification_sophon(args, data_config, scores, labels, observers):
    import ast
    network_options = {k: ast.literal_eval(v) for k, v in args.network_option}

    num_classes = network_options['num_classes']

    label_default = [f'label_{i}' for i in range(num_classes)]
    label_cls_nodes = network_options.get('label_cls_nodes', label_default)
    label_stored = network_options.get('label_stored', label_cls_nodes) # by default, store all classification node scores

    output = {}
    output['cls_index'] = labels['truth_label'] # classes can be too many, only store the index
    for idx, label_name in enumerate(label_cls_nodes):
        if label_name in label_stored:
            output[label_name] = (labels['truth_label'] == idx)
            output['score_' + label_name] = scores[:, idx]

    for k, v in labels.items():
        if k == data_config.label_names[0]:
            continue
        assert v.ndim == 1
        output[k] = v
    for k, v in observers.items():
        assert v.ndim == 1
        output[k] = v
    
    for k in output.keys():
        print(k, output[k])

    return output
