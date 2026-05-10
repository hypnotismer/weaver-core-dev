import time
import glob
import copy
import gc
import numpy as np
import awkward as ak

from ..logger import _logger
from .tools import _get_variable_names, _eval_expr
from .fileio import _read_files, _read_files_concurrent


def _apply_selection(table, selection):
    if selection is None:
        return table
    selected = ak.values_astype(_eval_expr(selection, table), 'bool')
    return table[selected]


def _build_new_variables(table, funcs):
    if funcs is None:
        return table
    for k, expr in funcs.items():
        if k in table.fields:
            continue
        table[k] = _eval_expr(expr, table)
    return table


def _clean_up(table, drop_branches):
    columns = [k for k in table.fields if k not in drop_branches]
    return table[columns]


def _build_weights(table, data_config, reweight_hists=None, warn=_logger.warning):
    if data_config.weight_name is None:
        raise RuntimeError('Error when building weights: `weight_name` is None!')
    if data_config.use_precomputed_weights:
        return ak.to_numpy(table[data_config.weight_name])
    else:
        x_var, y_var = data_config.reweight_branches
        x_bins, y_bins = data_config.reweight_bins
        rwgt_sel = None
        if data_config.reweight_discard_under_overflow:
            rwgt_sel = (table[x_var] >= min(x_bins)) & (table[x_var] <= max(x_bins)) & \
                (table[y_var] >= min(y_bins)) & (table[y_var] <= max(y_bins))
        # init w/ wgt=0: events not belonging to any class in `reweight_classes` will get a weight of 0 at the end
        wgt = np.zeros(len(table), dtype='float32')
        sum_evts = 0
        if reweight_hists is None:
            reweight_hists = data_config.reweight_hists
        for label, hist in reweight_hists.items():
            pos = table[label] == 1
            if rwgt_sel is not None:
                pos = (pos & rwgt_sel)
            rwgt_x_vals = ak.to_numpy(table[x_var][pos])
            rwgt_y_vals = ak.to_numpy(table[y_var][pos])
            x_indices = np.clip(np.digitize(
                rwgt_x_vals, x_bins) - 1, a_min=0, a_max=len(x_bins) - 2)
            y_indices = np.clip(np.digitize(
                rwgt_y_vals, y_bins) - 1, a_min=0, a_max=len(y_bins) - 2)
            wgt[pos] = hist[x_indices, y_indices]
            sum_evts += np.sum(pos)
        if sum_evts != len(table):
            warn(
                'Not all selected events used in the reweighting. '
                'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
                '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
            )
        if data_config.reweight_basewgt:
            wgt *= ak.to_numpy(table[data_config.basewgt_name])
        return wgt


class AutoStandardizer(object):
    r"""AutoStandardizer.

    Class to compute the variable standardization information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(
            filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()
        self.load_range = (0, data_config.preprocess.get('data_fraction', 0.1))

    def read_file(self, filelist):
        self.keep_branches = set()
        self.load_branches = set()
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                self.keep_branches.add(k)
                if k in self._data_config.var_funcs:
                    expr = self._data_config.var_funcs[k]
                    self.load_branches.update(_get_variable_names(expr))
                else:
                    self.load_branches.add(k)
        if self._data_config.selection:
            self.load_branches.update(_get_variable_names(self._data_config.selection))
        _logger.debug('[AutoStandardizer] keep_branches:\n  %s', ','.join(self.keep_branches))
        _logger.debug('[AutoStandardizer] load_branches:\n  %s', ','.join(self.load_branches))
        table = _read_files(filelist, self.load_branches, self.load_range,
                            show_progressbar=True, treename=self._data_config.treename)
        table = _apply_selection(table, self._data_config.selection)
        table = _build_new_variables(
            table, {k: v for k, v in self._data_config.var_funcs.items() if k in self.keep_branches})
        table = _clean_up(table, self.load_branches - self.keep_branches)
        return table

    def make_preprocess_params(self, table):
        _logger.info('Using %d events to calculate standardization info', len(table))
        preprocess_params = copy.deepcopy(self._data_config.preprocess_params)
        for k, params in self._data_config.preprocess_params.items():
            if params['center'] == 'auto':
                if k.endswith('_mask'):
                    params['center'] = None
                else:
                    a = ak.to_numpy(ak.flatten(table[k], axis=None))
                    low, center, high = np.percentile(a, [16, 50, 84])
                    scale = max(high - center, center - low)
                    scale = 1 if scale == 0 else 1. / scale
                    params['center'] = float(center)
                    params['scale'] = float(scale)
                preprocess_params[k] = params
                _logger.info('[AutoStandardizer] %s low=%s, center=%s, high=%s, scale=%s', k, low, center, high, scale)
        return preprocess_params

    def produce(self, output=None):
        table = self.read_file(self._filelist)
        preprocess_params = self.make_preprocess_params(table)
        self._data_config.preprocess_params = preprocess_params
        # must also propogate the changes to `data_config.options` so it can be persisted
        self._data_config.options['preprocess']['params'] = preprocess_params
        if output:
            _logger.info(
                'Writing YAML file w/ auto-generated preprocessing info to %s' % output)
            self._data_config.dump(output)
        return self._data_config


class WeightMaker(object):
    r"""WeightMaker.

    Class to make reweighting information.

    Arguments:
        filelist (list): list of files to be loaded.
        data_config (DataConfig): object containing data format information.
    """

    def __init__(self, filelist, data_config):
        if isinstance(filelist, dict):
            filelist = sum(filelist.values(), [])
        self._filelist = filelist if isinstance(filelist, (list, tuple)) else glob.glob(filelist)
        self._data_config = data_config.copy()

    def read_file(self, filelist):
        self.keep_branches = set(self._data_config.reweight_branches + self._data_config.reweight_classes +
                                 (self._data_config.basewgt_name,))
        self.load_branches = set()
        
        # 递归收集所有依赖的新变量
        def collect_dependencies(var_name, collected_vars):
            if var_name in collected_vars:
                return
            collected_vars.add(var_name)
            if var_name in self._data_config.var_funcs:
                expr = self._data_config.var_funcs[var_name]
                dependencies = _get_variable_names(expr)
                for dep in dependencies:
                    if dep in self._data_config.var_funcs:
                        collect_dependencies(dep, collected_vars)
                    else:
                        self.load_branches.add(dep)
            else:
                self.load_branches.add(var_name)
        
        # 收集所有需要的新变量及其依赖
        all_needed_vars = set()
        for k in self.keep_branches:
            collect_dependencies(k, all_needed_vars)
        
        if self._data_config.selection:
            self.load_branches.update(_get_variable_names(self._data_config.selection))
        _logger.debug('[WeightMaker] keep_branches:\n  %s', ','.join(self.keep_branches))
        _logger.debug('[WeightMaker] all_needed_vars:\n  %s', ','.join(all_needed_vars))
        _logger.debug('[WeightMaker] load_branches:\n  %s', ','.join(self.load_branches))
        table = _read_files(filelist, self.load_branches, show_progressbar=True, treename=self._data_config.treename)
        table = _apply_selection(table, self._data_config.selection)
        table = _build_new_variables(
            table, {k: v for k, v in self._data_config.var_funcs.items() if k in all_needed_vars})
        table = _clean_up(table, self.load_branches - all_needed_vars)
        return table

    def _make_raw_hists(self, table):
        x_var, y_var = self._data_config.reweight_branches
        x_bins, y_bins = self._data_config.reweight_bins
        if not self._data_config.reweight_discard_under_overflow:
            # clip variables to be within bin ranges
            x_min, x_max = min(x_bins), max(x_bins)
            y_min, y_max = min(y_bins), max(y_bins)
            _logger.info(f'Clipping `{x_var}` to [{x_min}, {x_max}] to compute the shapes for reweighting.')
            _logger.info(f'Clipping `{y_var}` to [{y_min}, {y_max}] to compute the shapes for reweighting.')
            table[x_var] = np.clip(table[x_var], min(x_bins), max(x_bins))
            table[y_var] = np.clip(table[y_var], min(y_bins), max(y_bins))

        _logger.info('Using %d events to make weights', len(table))

        sum_evts = 0
        raw_hists = {}
        for label in self._data_config.reweight_classes:
            pos = (table[label] == 1)
            x = ak.to_numpy(table[x_var][pos])
            y = ak.to_numpy(table[y_var][pos])
            hist, _, _ = np.histogram2d(x, y, bins=self._data_config.reweight_bins)
            sum_evts += hist.sum()
            if self._data_config.reweight_basewgt:
                w = ak.to_numpy(table[self._data_config.basewgt_name][pos])
                hist, _, _ = np.histogram2d(x, y, weights=w, bins=self._data_config.reweight_bins)
            raw_hists[label] = hist.astype('float32')
        return raw_hists, sum_evts, len(table)

    def _make_weights_from_raw_hists(self, raw_hists, sum_evts, total_events, table=None):
        max_weight = 0.9
        class_events = {}
        result = {label: hist.copy() for label, hist in raw_hists.items()}

        for label in self._data_config.reweight_classes:
            _logger.info('%s (unweighted):\n %s', label, str(raw_hists[label].astype('int64')))
            if self._data_config.reweight_basewgt:
                _logger.info('%s (weighted):\n %s', label, str(raw_hists[label].astype('float32')))

        if sum_evts != total_events:
            _logger.warning(
                'Only %d (out of %d) events actually used in the reweighting. '
                'Check consistency between `selection` and `reweight_classes` definition, or with the `reweight_vars` binnings '
                '(under- and overflow bins are discarded by default, unless `reweight_discard_under_overflow` is set to `False` in the `weights` section).',
                sum_evts, total_events)
            time.sleep(10)

        if self._data_config.reweight_method == 'flat':
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                hist = result[label]
                threshold_ = np.median(hist[hist > 0]) * 0.01
                nonzero_vals = hist[hist > threshold_]
                min_val, med_val = np.min(nonzero_vals), np.median(hist)  # not really used
                ref_val = np.percentile(nonzero_vals, self._data_config.reweight_threshold)
                _logger.debug('label:%s, median=%f, min=%f, ref=%f, ref/min=%f' %
                              (label, med_val, min_val, ref_val, ref_val / min_val))
                # wgt: bins w/ 0 elements will get a weight of 0; bins w/ content<ref_val will get 1
                wgt = np.clip(np.nan_to_num(ref_val / hist, posinf=0), 0, 1)
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        elif self._data_config.reweight_method == 'ref':
            # 支持两种形式：
            # 1) 全局参考直方图（数组）：对所有类应用
            # 2) 按类的参考直方图（字典）：仅对在字典中出现的类应用；其他类权重为全1
            dict_ref = isinstance(getattr(self._data_config, 'reweight_hist_ref', None), dict)
            if self._data_config.reweight_hist_ref is not None and not dict_ref:
                hist_ref_global = np.array(self._data_config.reweight_hist_ref, dtype='float32')
                assert hist_ref_global.shape == raw_hists[self._data_config.reweight_classes[0]].shape, \
                    'Error: shape of reference histogram does not match the shape of the reweighting histograms!'
                _logger.info('Using reweight method "ref" with a global hist_ref.')
            elif self._data_config.reweight_hist_ref is None:
                # use class 0 as the global reference if nothing provided
                hist_ref_global = raw_hists[self._data_config.reweight_classes[0]]
                _logger.info('Using reweight method "ref" with class-0 histogram as global reference.')
            # 当使用字典形式时，逐类读取参考直方图
            for label, classwgt in zip(self._data_config.reweight_classes, self._data_config.class_weights):
                if dict_ref:
                    refs: dict = self._data_config.reweight_hist_ref
                    if label in refs and refs[label] is not None:
                        hist_ref = np.array(refs[label], dtype='float32')
                        assert hist_ref.shape == raw_hists[label].shape, \
                            f'Error: shape of reference histogram for class {label} does not match!'
                        ratio = np.nan_to_num(hist_ref / result[label], posinf=0)
                        
                        upper = np.percentile(ratio[ratio > 0], 95)
                        wgt = np.clip(ratio / upper, 0, 1)
                        '''
                        # 按行进行percentile裁剪，避免整行饱和
                        wgt = np.zeros_like(result[label], dtype='float32')
                        for ix in range(ratio.shape[0]):
                            row = ratio[ix, :]
                            pos = row > 0
                            if np.any(pos):
                                upper_row = np.percentile(row[pos], 100 - self._data_config.reweight_threshold)
                                wgt[ix, pos] = np.clip(row[pos] / upper_row, 0, 1)
                            else:
                                wgt[ix, :] = 0
                        '''
                        _logger.info('ref method applied to class %s.', label)
                    else:
                        # 未指定的类不做 reweight，使用单位权重
                        wgt = np.ones_like(result[label], dtype='float32')
                        _logger.info('ref method skipped for class %s (unit weights).', label)
                else:
                    # 全局参考：对所有类应用
                    ratio = np.nan_to_num(hist_ref_global / result[label], posinf=0)
                    upper = np.percentile(ratio[ratio > 0], 100 - self._data_config.reweight_threshold)
                    wgt = np.clip(ratio / upper, 0, 1)
                result[label] = wgt
                # divide by classwgt here will effective increase the weight later
                class_events[label] = np.sum(raw_hists[label] * wgt) / classwgt
        # ''equalize'' all classes
        # multiply by max_weight (<1) to add some randomness in the sampling
        min_nevt = min(class_events.values()) * max_weight
        for label in self._data_config.reweight_classes:
            class_wgt = float(min_nevt) / class_events[label]
            result[label] *= class_wgt

        if self._data_config.reweight_basewgt:
            if table is None:
                raise RuntimeError('Chunked WeightMaker does not support `reweight_basewgt` yet.')
            wgts = _build_weights(table, self._data_config, reweight_hists=result)
            wgt_ref = np.percentile(wgts, 100 - self._data_config.reweight_threshold)
            _logger.info('Set overall reweighting scale factor (%d threshold) to %s (max %s)' %
                         (100 - self._data_config.reweight_threshold, wgt_ref, np.max(wgts)))
            for label in self._data_config.reweight_classes:
                result[label] /= wgt_ref

        _logger.info('weights:')
        for label in self._data_config.reweight_classes:
            _logger.info('%s:\n %s', label, str(result[label]))

        _logger.info('Raw hist * weights:')
        for label in self._data_config.reweight_classes:
            _logger.info('%s:\n %s', label, str((raw_hists[label] * result[label]).astype('int32')))

        return result

    def make_weights(self, table):
        raw_hists, sum_evts, total_events = self._make_raw_hists(table)
        return self._make_weights_from_raw_hists(raw_hists, sum_evts, total_events, table=table)

    def produce(self, output=None):
        if self._data_config.reweight_basewgt:
            table = self.read_file(self._filelist)
            wgts = self.make_weights(table)
        else:
            chunk_size = int(self._data_config.options['weights'].get('reweight_chunk_size', 50))
            chunk_size = max(1, chunk_size)
            _logger.info('Making weights in chunks of %d files', chunk_size)
            raw_hists = None
            sum_evts = 0
            total_events = 0
            for start in range(0, len(self._filelist), chunk_size):
                chunk_files = self._filelist[start:start + chunk_size]
                _logger.info(
                    'Processing weight chunk %d-%d / %d',
                    start + 1, min(start + chunk_size, len(self._filelist)), len(self._filelist))
                try:
                    table = self.read_file(chunk_files)
                except RuntimeError as e:
                    if 'Zero entries loaded' in str(e):
                        _logger.warning('Skipping empty weight chunk %d-%d: %s',
                                        start + 1, min(start + chunk_size, len(self._filelist)), e)
                        continue
                    raise
                chunk_hists, chunk_sum_evts, chunk_total_events = self._make_raw_hists(table)
                if raw_hists is None:
                    raw_hists = {label: hist.copy() for label, hist in chunk_hists.items()}
                else:
                    for label in self._data_config.reweight_classes:
                        raw_hists[label] += chunk_hists[label]
                sum_evts += chunk_sum_evts
                total_events += chunk_total_events
                del table
                gc.collect()
            if raw_hists is None:
                raise RuntimeError('Zero entries loaded when making weights.')
            wgts = self._make_weights_from_raw_hists(raw_hists, sum_evts, total_events)
        ## for debugging ##
        # # write table:
        # import pickle
        # with open('table.pkl', 'wb') as f:
        #     pickle.dump(table, f)
        # with open('table.pkl', 'rb') as f:
        #     table = pickle.load(f)
        self._data_config.reweight_hists = wgts
        # must also propogate the changes to `data_config.options` so it can be persisted
        self._data_config.options['weights']['reweight_hists'] = {k: v.tolist() for k, v in wgts.items()}
        if output:
            _logger.info('Writing YAML file w/ reweighting info to %s' % output)
            self._data_config.dump(output)
        return self._data_config
