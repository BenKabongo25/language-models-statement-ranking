"""LMStaR training: two independent towers on any Hugging Face backbone (LoRA or full fine-tuning), exact
statement-gradient replay, world-size invariant data parallelism. Ablations (``model.towers``): one shared tower,
or a trained query tower with the frozen backbone as statement tower.

q_x = pool(f_theta(x)), d_s = pool(g_phi(s)), score <q_x, d_s>; per-positive multi-positive InfoNCE over sampled
item-level negatives (uniform or UserPop); checkpoint selection on validation nDCG@5, final test of the best one.

    python -m lmstar.train --config configs/base.yaml --config configs/models/qwen3-embedding-0.6b.yaml \\
        --config configs/datasets/beauty.yaml [--config configs/variants/...] [--seed 40] [--gpus 2] [--set K=V]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch
import yaml
from tqdm.auto import tqdm

from .config import load, public, removed_features
from .data.cache import prepare_contexts, statement_tokens
from .data.dataset import TEST_CONTEXTS, HistoryDataset
from .data.prompts import history_settings
from .distributed import all_reduce_sum, barrier, broadcast_object, broadcast_params, launch, sync_grads
from .inference import evaluate, restore_best, save_best
from .model.encoder import POOLINGS, build_encoder, check_lengths, find_tower, tower_dir, tower_mode
from .model.tokenizer import load_tokenizer
from .training.gradcache import gradcache_step
from .training.sampling import NegativeSampler, SamplingStats, batch_tensors, eligible_examples, sample_rows
from .utils import batch_order, digest, set_seed, setup_logging

LOG = logging.getLogger('lmstar')
METHOD_DIR = 'lmstar'
POSITIVE_KEYS = ('batch_size', 'gradient_accumulation_steps', 'document_microbatch_size', 'bucket_size',
                 'document_eval_batch_size', 'eval_batch_size', 'epochs', 'eval_steps', 'log_steps', 'num_negatives')


# ----------------------------------------------------------------------------------------------- config
def validate_config(cfg, world=1):
    """Check a resolved configuration; resolves ``training.init_checkpoint`` to its ``best`` folder."""
    mc, tc = cfg['model'], cfg['training']
    removed = removed_features(cfg)
    if removed:
        raise ValueError(f'features removed from LMStaR: {removed}')
    if not cfg['dataset'].get('name'):
        raise ValueError('dataset.name is required (configs/datasets/<d>.yaml)')
    use_lora = bool(mc.get('use_lora', True))
    if use_lora and float(mc['lora'].get('dropout', 0)) != 0:
        raise ValueError('Gradient replay requires LoRA dropout=0')
    if not use_lora and mc.get('dtype') in ('bfloat16', 'float16') and not mc.get('autocast'):
        LOG.warning('Full fine-tuning with %s weights: AdamW updates are rounded; prefer model.dtype=float32 + '
                    'model.autocast=bfloat16 (configs/variants/full_finetuning.yaml)', mc['dtype'])
    if mc.get('pooling', 'eos') not in POOLINGS:
        raise ValueError(f'model.pooling must be one of {POOLINGS}')
    mode = tower_mode(cfg)
    if mc.get('padding_side', 'right') not in ('right', 'left'):
        raise ValueError('model.padding_side: right | left')
    if tc.get('loss', 'infonce') != 'infonce':
        raise ValueError('Only InfoNCE supported')
    if not 1 <= int(tc['epochs']) <= 10:
        raise ValueError('epochs must be between 1 and 10')
    if int(mc['query_max_length']) < 128:
        raise ValueError('query_max_length must be >= 128')
    if int(mc['document_max_length']) < 16:
        raise ValueError('document_max_length must be >= 16')
    for key in POSITIVE_KEYS:
        if int(tc[key]) <= 0:
            raise ValueError(f'training.{key} must be > 0')
    if float(tc['temperature']) <= 0:
        raise ValueError('temperature must be > 0')
    if int(tc['batch_size']) < world:
        raise ValueError('training.batch_size (global) must be >= number of GPUs')
    history_settings(cfg)
    if cfg['evaluation'].get('test_context') not in TEST_CONTEXTS:
        raise ValueError(f'evaluation.test_context must be one of {TEST_CONTEXTS}')
    init = tc.get('init', 'scratch')
    if init not in ('scratch', 'checkpoint'):
        raise ValueError('training.init must be scratch or checkpoint')
    if init == 'checkpoint':
        tc['init_checkpoint'] = str(_init_checkpoint(tc.get('init_checkpoint'), use_lora, mode))


INIT_ROLES = {'independent': ('query', 'document'), 'shared': ('shared',), 'query_only': ('query',)}


def _init_checkpoint(path, use_lora, mode='independent'):
    if not path:
        raise ValueError('training.init=checkpoint requires training.init_checkpoint (<run>/best)')
    ck = Path(path)
    if ck.name != 'best' and (ck / 'best').exists():
        ck = ck / 'best'
    for role in INIT_ROLES[mode]:
        sub = ck / tower_dir(role, use_lora)
        if not sub.is_dir():
            other = find_tower(ck, role)
            hint = ''
            if other:
                kind = 'LoRA' if other.name.endswith('_lora') else 'full fine-tuning'
                hint = f' ({other.name} found: the checkpoint was trained with {kind}, set model.use_lora accordingly)'
            raise ValueError(f'training.init_checkpoint: {sub} not found{hint}')
    return ck


def make_run_id(cfg, world):
    """Run folder name: the settings that distinguish runs, a timestamp and a hash of the configuration."""
    mc, tc, hs = cfg['model'], cfg['training'], history_settings(cfg)
    ns = tc.get('negative_sampling') or {}
    neg = f"neg{tc['num_negatives']}"
    if ns.get('strategy') == 'uniform_userpop':
        neg += f"up{round(100 * float(ns.get('userpop_ratio', 0)))}"
    hist = f"{hs['order']}-{hs['packing']}"
    if hs['views'] > 1:
        hist += f"-v{hs['views']}"
    if hs.get('target_position') == 'first':
        hist += '-tfirst'
    pooling = '' if mc.get('pooling', 'eos') == 'eos' else f"_{mc['pooling']}"
    full = '' if mc.get('use_lora', True) else '_full'
    full += {'independent': '', 'shared': '_shared', 'query_only': '_qonly'}[tower_mode(cfg)]
    init = '_ft' if tc.get('init', 'scratch') == 'checkpoint' else ''
    return (f"LMStaR_L{mc['query_max_length']}_cap{hs['max_statements_per_interaction'] or 'all'}"
            f"_{hist}{pooling}{full}{init}_{neg}_bs{tc['batch_size']}x{tc['gradient_accumulation_steps']}"
            f"_g{world}_dmb{tc['document_microbatch_size']}_tau{float(tc['temperature']):g}_seed{int(cfg['seed'])}"
            f"_{time.strftime('%Y%m%d_%H%M%S')}_{digest(cfg)[:7]}")


def model_label(cfg):
    """Backbone folder name of the runs (last component of ``model.name``)."""
    return cfg['model']['name'].rstrip('/').split('/')[-1]


def param_groups(query_params, doc_params, tc):
    """One AdamW group per trained tower; 1-D parameters (biases, norms; full fine-tuning only) get no weight decay.
    A shared tower (or a frozen statement tower) has no statement parameters: learning_rate_query only."""
    groups = []
    for params, lr in ((query_params, tc['learning_rate_query']), (doc_params, tc['learning_rate_document'])):
        if not params:
            continue
        groups.append({'params': [p for p in params if p.ndim >= 2], 'lr': float(lr)})
        flat = [p for p in params if p.ndim < 2]
        if flat:
            groups.append({'params': flat, 'lr': float(lr), 'weight_decay': 0.0})
    return groups


# ---------------------------------------------------------------------------------------------- training
class Sums:
    """Running sums of one logging window, all-reduced over the ranks."""
    FIELDS = ('loss', 'examples', 'positives', 'negatives', 'unique_docs', 'microbatches')

    def __init__(self):
        self.values = [0.] * len(self.FIELDS)

    def add(self, loss, n, positives, negatives, unique):
        for j, v in enumerate((loss * n, n, positives, negatives, unique, 1)):
            self.values[j] += v

    def reduce(self, env):
        return dict(zip(self.FIELDS, all_reduce_sum(self.values, env)))


def run(env, cfg):
    """Train one run (``env``: rank, world, device) and return the test metrics of the best checkpoint."""
    mc, tc, seed = cfg['model'], cfg['training'], int(cfg['seed'])
    set_seed(seed)
    validate_config(cfg, env.world)
    device = env.device
    if device.type != 'cuda' and os.environ.get('LMSTAR_ALLOW_CPU') != '1':
        raise RuntimeError('CUDA GPU required')
    torch.backends.cuda.matmul.allow_tf32 = True
    cfg['runtime']['world_size'] = env.world
    dataset_root = Path(cfg['output']['root']) / cfg['dataset']['name']
    folder = dataset_root / 'runs' / model_label(cfg) / METHOD_DIR / make_run_id(cfg, env.world) if env.main else None
    folder = broadcast_object(folder, env)
    setup_logging(folder, rank=env.rank)
    gpu = torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu'
    LOG.info('LMStaR | %s | GPUs=%d x %s | torch=%s', folder.name, env.world, gpu, torch.__version__)

    # data and token caches (rank 0 builds them, the other ranks then load them)
    data = HistoryDataset(cfg)
    LOG.info('Dataset: %s', data.summary())
    tok = load_tokenizer(mc['name'], mc.get('append_eos', 'auto'), mc.get('trust_remote_code', False))
    cfg['_pad_id'] = tok.pad_token_id
    cfg['_tokenizer'] = tok
    for rank0 in (True, False):
        if env.main == rank0:
            contexts = prepare_contexts(data, tok, cfg)
            ids, doc_tokens = statement_tokens(data, tok, cfg, dataset_root / 'token_cache')
        if rank0:
            barrier(env)
    index = {sid: j for j, sid in enumerate(ids)}
    train = [e for e in eligible_examples(data, 'train', index) if e.interaction_id in contexts]
    val = [e for e in data.examples('val') if e.interaction_id in contexts]
    test = [e for e in data.examples('test') if e.interaction_id in contexts]
    LOG.info('Coverage: train=%d/%d val=%d/%d test=%d/%d | test_context=%s', len(train), len(data.ids_by_split['train']),
             len(val), len(data.ids_by_split['val']), len(test), len(data.ids_by_split['test']), data.test_context)
    if len(val) != len(data.ids_by_split['val']) or len(test) != len(data.ids_by_split['test']):
        raise RuntimeError('Validation/test coverage changed!')
    if not train or not val or not test:
        raise RuntimeError('Empty split')
    sampler = NegativeSampler(data, int(tc['num_negatives']), tc.get('negative_sampling'), seed)
    LOG.info('Negative sampler: strategy=%s ratio=%.2f pool_multiplier=%d', sampler.strategy, sampler.ratio,
             sampler.multiplier)

    # towers
    use_lora, mode = bool(mc.get('use_lora', True)), tower_mode(cfg)
    towers = {'independent': 'TWO independent', 'shared': 'ONE shared', 'query_only': 'a trained query + frozen statement'}
    LOG.info('Loading %s %s backbone(s) (%s) | pooling=%s padding=%s dtype=%s autocast=%s | document_prefix=%r',
             towers[mode], mc['name'], 'LoRA' if use_lora else 'full fine-tuning', mc.get('pooling', 'eos'),
             mc.get('padding_side', 'right'), mc.get('dtype'), mc.get('autocast'), mc.get('document_prefix') or '')
    query = build_encoder(cfg, tok, 'query').to(device)
    check_lengths(cfg, query)
    if mode == 'shared':
        document = query
    else:
        document = build_encoder(cfg, tok, 'document', trainable=mode == 'independent').to(device)
    if use_lora:
        cfg['model']['lora']['target_modules'] = list(query.lora_cfg['target_modules'])     # resolved 'auto'
    writer = None
    if env.main:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(folder / 'tensorboard'))
        (folder / 'config.yaml').write_text(yaml.safe_dump(public(cfg), sort_keys=False), encoding='utf-8')
    LOG.info('Config: %s', json.dumps(public(cfg), default=str))
    ck = Path(tc['init_checkpoint']) if tc.get('init', 'scratch') == 'checkpoint' else None
    if ck is not None:
        query.load_checkpoint(ck / tower_dir('shared' if mode == 'shared' else 'query', use_lora))
        if mode == 'independent':
            document.load_checkpoint(ck / tower_dir('document', use_lora))
        LOG.info('INIT from checkpoint %s (%s of the trained tower(s) loaded; fresh optimizer)', ck,
                 'LoRA' if use_lora else 'complete weights')
    query_params = [p for p in query.parameters() if p.requires_grad]
    doc_params = [p for p in document.parameters() if p.requires_grad] if mode == 'independent' else []
    params = query_params + doc_params
    broadcast_params(params, env)                    # identical initialisation on every rank
    LOG.info('Trainable params query=%s statement=%s', f'{sum(p.numel() for p in query_params):,}',
             f'{sum(p.numel() for p in doc_params):,}')

    # optimisation
    optimizer = torch.optim.AdamW(param_groups(query_params, doc_params, tc), weight_decay=float(tc['weight_decay']))
    from transformers import get_cosine_schedule_with_warmup
    batch_size, accum = int(tc['batch_size']), int(tc['gradient_accumulation_steps'])
    steps_per_epoch = math.ceil(math.ceil(len(train) / batch_size) / accum)
    total_steps = steps_per_epoch * int(tc['epochs'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * float(tc['warmup_ratio'])), total_steps)
    if batch_size % env.world:
        LOG.warning('Global batch %d not divisible by %d GPUs: per-GPU shards differ by one example', batch_size,
                    env.world)
    LOG.info('Train: %d examples | global physical batch=%d (%.1f/GPU on %d GPU) | accumulation=%d | '
             'effective batch=%d | steps/epoch=%d | doc microbatch=%d | optimizer steps=%d | context views=%d',
             len(train), batch_size, batch_size / env.world, env.world, accum, batch_size * accum, steps_per_epoch,
             tc['document_microbatch_size'], total_steps, contexts.views)
    state = {'best': -float('inf'), 'best_step': 0, 'best_epoch': 0, 'patience': 0, 'step': 0, 'saved': False}
    start, tau, doc_chunk = time.time(), float(tc['temperature']), int(tc['document_microbatch_size'])
    selection = cfg['evaluation']['selection_metric']
    patience = int(tc['early_stopping_patience'])

    def validate(epoch, tag=''):
        metrics = evaluate(query, document, data, val, contexts, doc_tokens, index, cfg, env, 'val')
        step = state['step']
        if writer:
            for k, v in metrics.items():
                writer.add_scalar('val/' + k, v, step)
        selected = metrics.get(selection, float('nan'))
        if math.isfinite(selected) and selected > state['best']:
            state.update(best=selected, best_step=step, best_epoch=epoch, patience=0, saved=True)
            if env.main:
                save_best(query, document, tok, folder / 'best', cfg, epoch, step, metrics)
            barrier(env)
        else:
            state['patience'] += 1
        LOG.info('VALIDATION%s epoch=%d step=%d %s=%.4f best=%.4f@%d patience=%d/%d', f' [{tag}]' if tag else '',
                 epoch, step, selection, selected, state['best'], state['best_step'], state['patience'], patience)
        if writer:
            writer.flush()

    def run_epoch(epoch):
        """One epoch of accumulation windows; returns True when early stopping triggers."""
        view = (epoch - 1) % contexts.views
        ctx = contexts.view(view)
        order = batch_order(train, ctx, seed, epoch, int(tc['bucket_size']))
        batches = [order[j:j + batch_size] for j in range(0, len(order), batch_size)]
        sampler.start_epoch(epoch)
        epoch_stats, window_stats, sums = SamplingStats(), SamplingStats(), Sums()
        last_eval = -1
        optimizer.zero_grad(set_to_none=True)
        bar = tqdm(enumerate(batches), total=len(batches), disable=not env.main,
                   desc=f'Epoch {epoch}/{tc["epochs"]} (view {view})')
        for bidx, indices in bar:
            examples = [train[j] for j in indices]
            rows, npos = sample_rows(examples, data, index, sampler, window_stats)    # global batch, every rank
            window = (bidx // accum) * accum
            count = sum(len(batches[k]) for k in range(window, min(window + accum, len(batches))))
            local = list(range(env.rank, len(examples), env.world))
            if local:
                unique, idx, pm, nm = batch_tensors([rows[j] for j in local], [npos[j] for j in local])
                loss = gradcache_step(query, document, [ctx[examples[j].interaction_id] for j in local],
                                      [doc_tokens[i] for i in unique], idx, pm, nm, tau, len(local) / count,
                                      device, doc_chunk, replay=mode != 'query_only')
                sums.add(loss, len(local), sum(npos[j] for j in local),
                         sum(len(rows[j]) - npos[j] for j in local), len(unique))
            if not ((bidx + 1) % accum == 0 or bidx == len(batches) - 1):
                continue
            sync_grads(params, env)
            norm = torch.nn.utils.clip_grad_norm_(params, float(tc['max_grad_norm']))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            state['step'] += 1
            step = state['step']
            tot = sums.reduce(env)
            mean_loss = tot['loss'] / tot['examples']
            docs_per_mb = tot['unique_docs'] / max(1, tot['microbatches'])
            lrs = [g['lr'] for g in optimizer.param_groups]
            if writer:
                writer.add_scalar('train/loss', mean_loss, step)
                for gi, lr in enumerate(lrs):
                    writer.add_scalar(f'train/lr_group{gi}', lr, step)
                writer.add_scalar('train/grad_norm', float(norm), step)
                writer.add_scalar('train/avg_positives', tot['positives'] / tot['examples'], step)
                writer.add_scalar('train/avg_negatives', tot['negatives'] / tot['examples'], step)
                writer.add_scalar('train/unique_docs_per_microbatch', docs_per_mb, step)
                for metric, value in window_stats.metrics().items():
                    writer.add_scalar('negative_sampling/' + metric, value, step)
            epoch_stats.add(window_stats)
            window_stats = SamplingStats()
            if env.main:
                bar.set_postfix(loss=f'{mean_loss:.4f}', step=step, best=f"{state['best']:.2f}")
            if step % int(tc['log_steps']) == 0:
                peak = torch.cuda.max_memory_allocated(device) / 1024 ** 3 if device.type == 'cuda' else 0.
                LOG.info('TRAIN epoch=%d step=%d loss=%.6f lr=%s grad=%.3f unique_docs_per_microbatch=%.1f '
                         'gpu_peak(rank0)=%.2fGiB elapsed=%.1fm', epoch, step, mean_loss,
                         '/'.join(f'{lr:.3g}' for lr in lrs), float(norm), docs_per_mb, peak,
                         (time.time() - start) / 60)
            sums = Sums()
            if step % int(tc['eval_steps']) == 0:
                validate(epoch)
                last_eval = step
                if state['patience'] >= patience:
                    return True
                query.train()
                document.train()
        LOG.info('NEGATIVE SAMPLING epoch=%d | %s', epoch, epoch_stats.metrics())
        if last_eval != state['step']:
            validate(epoch)
            if state['patience'] >= patience:
                return True
        return False

    if ck is not None:
        validate(0, 'init checkpoint')               # the checkpoint on this dataset, before fine-tuning
    for epoch in range(1, int(tc['epochs']) + 1):
        query.train()
        document.train()
        stop = run_epoch(epoch)
        LOG.info('EPOCH %d complete | step=%d | best=%.4f', epoch, state['step'], state['best'])
        if stop:
            break
    if not state['saved']:
        raise RuntimeError('No best checkpoint saved')
    LOG.info('TRAIN COMPLETE | %.1f min | best epoch=%d step=%d %s=%.4f', (time.time() - start) / 60,
             state['best_epoch'], state['best_step'], selection, state['best'])

    # final test of the best checkpoint
    query = document = optimizer = scheduler = None          # free the training towers before the test
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    barrier(env)
    LOG.info('Restoring BOTH best towers for final test (test_context=%s)', data.test_context)
    query, document = restore_best(folder / 'best', cfg, device)
    metrics = evaluate(query, document, data, test, contexts, doc_tokens, index, cfg, env, 'test',
                       folder / 'test_predictions.csv')
    if env.main:
        for k, v in metrics.items():
            writer.add_scalar('test_final/' + k, v, state['best_step'])
        summary = {'best_step': state['best_step'], 'best_epoch': state['best_epoch'], 'best_val': state['best'],
                   'test_context': data.test_context, 'test': metrics}
        (folder / 'test_metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        writer.flush()
        writer.close()
    LOG.info('FINAL TEST | %s | %s', metrics, folder / 'test_predictions.csv')
    barrier(env)
    return metrics


# --------------------------------------------------------------------------------------------------- CLI
def build_config(argv=None, parser=None):
    p = parser or argparse.ArgumentParser(description='LMStaR: learning to rank explanation statements with LMs')
    p.add_argument('--config', action='append', required=True, help='composable YAML, applied left to right')
    p.add_argument('--seed', type=int)
    p.add_argument('--gpus', type=int, help='number of local GPUs (data parallel); default: runtime.gpus')
    p.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE')
    args = p.parse_args(argv)
    cfg = load(args.config, args.set, p.error)
    if args.seed is not None:
        cfg['seed'] = args.seed
    if args.gpus is not None:
        cfg['runtime']['gpus'] = args.gpus
    try:
        validate_config(cfg, int(cfg['runtime']['gpus']))
    except ValueError as e:
        p.error(str(e))
    return cfg


def main(argv=None):
    cfg = build_config(argv)
    launch(run, (cfg,), gpus=int(cfg['runtime']['gpus']),
           timeout_minutes=int(cfg['runtime'].get('dist_timeout_minutes', 240)))


if __name__ == '__main__':
    main()
