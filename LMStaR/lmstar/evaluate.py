"""Evaluate trained runs (re-evaluation, cross-domain transfer) and untrained backbones (zero-shot).

A run keeps its own training-time settings (backbone, pooling, prompt, history, lengths) read from
``<run>/config.yaml``; runs of earlier versions (LMStarDual v1.2+, LMStaR 2-4) are supported.

    python -m lmstar.evaluate --runs <run_a> <run_b> --splits val test [--gpus 2]
    python -m lmstar.evaluate --dataset Beauty --backbone Qwen3-Embedding-8B --select top:1 --splits test
    python -m lmstar.evaluate --runs <beauty_run> --target-dataset Toys --target-dataset Yelp      # transfer
    python -m lmstar.evaluate --zero-shot --config configs/base.yaml --config configs/models/bge-m3.yaml \\
        --config configs/datasets/beauty.yaml --splits test

Outputs (``metrics.json``, ``test_predictions.csv``, ``eval.log``): ``<run>/reeval/<tag>/``,
``<run>/transfer/<Target>/<tag>/``, ``<output.root>/<Dataset>/zero_shot/<model>/<tag>/``, and one summary CSV per
kind, ``<output.root>/{reeval,transfer,zero_shot}_<tag>.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path

import torch
import yaml

from .config import DEFAULTS, apply_sets, legacy_defaults, load, load_yaml, removed_features, resolve
from .data.cache import prepare_contexts, statement_tokens
from .data.dataset import TEST_CONTEXTS, HistoryDataset
from .distributed import barrier, launch
from .inference import evaluate, restore_best
from .model.encoder import checkpoint_complete
from .utils import read_json, setup_logging

LOG = logging.getLogger('lmstar')
KEY_METRICS = ('ndcg@5', 'ndcg@10', 'recall@5', 'recall@10', 'hr@5', 'precision@5')
METHODS = ('lmstar', 'dual_lora_infonce')          # dual_lora_infonce: runs of LMStarDual v1.x


# ---------------------------------------------------------------------------------------------- runs
def run_config(run):
    """Resolved configuration of a run (settings it predates take the values its version used)."""
    return resolve(legacy_defaults(load_yaml(Path(run) / 'config.yaml')))


def describe(run):
    """Summary of a run folder: dataset, model, recorded validation / test metrics, completeness."""
    run = Path(run)
    meta = read_json(run / 'best' / 'metadata.json', {})
    test = read_json(run / 'test_metrics.json', {})
    cfg = load_yaml(run / 'config.yaml') if (run / 'config.yaml').exists() else {}
    val = meta.get('val') or {}
    return {'run': run, 'run_id': run.name, 'dataset': (cfg.get('dataset') or {}).get('name', '?'),
            'model': (cfg.get('model') or {}).get('name', '?'),
            'best_val': test.get('best_val', val.get('ndcg@5')), 'val': val,
            'recorded_test': test.get('test') or {}, 'recorded_test_context': test.get('test_context', 'fit_only (v1.2)'),
            'complete': (checkpoint_complete(run / 'best', (cfg.get('model') or {}).get('towers') or 'independent')
                         and (run / 'config.yaml').exists()),
            'removed_features': removed_features(cfg),
            'mtime': (run / 'config.yaml').stat().st_mtime if (run / 'config.yaml').exists() else 0}


def discover(root, datasets, backbone, methods=METHODS):
    """Runs under ``<root>/<Dataset>/runs/<backbone>/<method>/``."""
    out = []
    for ds in datasets:
        for method in ([methods] if isinstance(methods, str) else methods):
            base = Path(root) / ds / 'runs' / backbone / method
            if base.exists():
                out.extend(describe(r) for r in sorted(base.iterdir()) if r.is_dir())
    return out


def select(runs, rule, filters):
    """Complete runs whose id contains every filter; ``rule``: all | top:N (best validation) | latest:N, per
    dataset."""
    runs = [r for r in runs if r['complete'] and not r['removed_features'] and all(f in r['run_id'] for f in filters)]
    if rule in (None, 'all'):
        return runs
    kind, _, n = rule.partition(':')
    n = int(n or 1)
    out = []
    for ds in sorted({r['dataset'] for r in runs}):
        group = [r for r in runs if r['dataset'] == ds]
        if kind == 'top':
            group.sort(key=lambda r: -(r['best_val'] if r['best_val'] is not None else -1e9))
        elif kind == 'latest':
            group.sort(key=lambda r: -r['mtime'])
        else:
            raise SystemExit(f'Unknown --select {rule!r} (all | top:N | latest:N)')
        out.extend(group[:n])
    return out


def print_table(runs):
    print(f"{'dataset':8} {'best_val ndcg@5':>15} {'test ndcg@5 (recorded)':>23} {'ok':>3}  run")
    for r in runs:
        best = '-' if r['best_val'] is None else f"{r['best_val']:.3f}"
        test = r['recorded_test'].get('ndcg@5')
        test = '-' if test is None else f'{test:.3f}'
        ok = 'n' if not r['complete'] else ('old' if r['removed_features'] else 'y')
        print(f"{r['dataset']:8} {best:>15} {test:>23} {ok:>3}  {r['run']}")


# -------------------------------------------------------------------------------------------- worker
def prepare(cfg, env, splits):
    """Tokenizer, dataset, query contexts and statement tokens of an evaluation (rank 0 builds the caches)."""
    from .model import tokenizer as TK
    mc = cfg['model']
    tok = TK.load_tokenizer(mc['name'], mc.get('append_eos', 'auto'), mc.get('trust_remote_code', False))
    cfg['_pad_id'] = tok.pad_token_id
    cfg['_tokenizer'] = tok
    data = HistoryDataset(cfg)
    root = Path(cfg['output']['root']) / data.name
    for rank0 in (True, False):
        if env.main == rank0:
            contexts = prepare_contexts(data, tok, cfg, splits=splits)
            ids, doc_tokens = statement_tokens(data, tok, cfg, root / 'token_cache')
        if rank0:
            barrier(env)
    return tok, data, contexts, ids, doc_tokens


def zero_shot_towers(cfg, tok, device):
    """Backbone without LoRA, shared by both towers (same pooling, padding and prompt as training)."""
    from .model import encoder as ENC
    enc = ENC.build_encoder(cfg, tok, 'query', trainable=False).to(device).eval()
    ENC.check_lengths(cfg, enc)
    return enc, enc


def worker(env, jobs, opts):
    """Run evaluation jobs (``kind``: reeval | transfer | zero_shot) on the ranks of ``env``."""
    setup_logging(None, rank=env.rank)
    for i, job in enumerate(jobs, 1):
        cfg = resolve(job['cfg'])
        cfg['evaluation']['test_context'] = opts['test_context']
        if job.get('target'):
            cfg['dataset']['name'] = job['target']
        if job.get('target_root'):
            cfg['dataset']['root'] = job['target_root']
        apply_sets(cfg, opts['set'])
        out = Path(job['out'])
        if (out / 'metrics.json').exists() and not opts['overwrite']:
            LOG.info('[%d/%d] SKIP (exists, use --overwrite): %s', i, len(jobs), out)
            continue
        if env.main:
            out.mkdir(parents=True, exist_ok=True)
            setup_logging(out, filename='eval.log', rank=0)
            (out / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        splits = tuple(opts['splits'])
        LOG.info('[%d/%d] %s | %s on %s | splits=%s | test_context=%s | GPUs=%d', i, len(jobs), job['kind'].upper(),
                 job.get('run', cfg['model']['name']), cfg['dataset']['name'], splits, opts['test_context'], env.world)
        tok, data, contexts, ids, doc_tokens = prepare(cfg, env, splits)
        index = {sid: j for j, sid in enumerate(ids)}
        if job['kind'] == 'zero_shot':
            query, document = zero_shot_towers(cfg, tok, env.device)
        else:
            query, document = restore_best(Path(job['run']) / 'best', cfg, env.device)
        result = {'kind': job['kind'], 'run': job.get('run'), 'model': cfg['model']['name'],
                  'source_dataset': job.get('source', cfg['dataset']['name']), 'dataset': data.name,
                  'test_context': opts['test_context'], 'evaluated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'world_size': env.world, 'recorded': job.get('recorded'), 'metrics': {}}
        for split in splits:
            examples = data.examples(split)
            if len(examples) != len(data.ids_by_split[split]):
                raise RuntimeError(f'{split} coverage changed')
            pred = out / 'test_predictions.csv' if (split == 'test' and not opts['no_predictions']) else None
            result['metrics'][split] = evaluate(query, document, data, examples, contexts, doc_tokens, index, cfg,
                                                env, split, pred)
        recorded = job.get('recorded') or {}
        if job['kind'] == 'reeval' and 'val' in result['metrics'] and recorded.get('best_val') is not None:
            result['val_reproduction_delta'] = round(result['metrics']['val']['ndcg@5'] - recorded['best_val'], 6)
            LOG.info('Validation reproduction: recomputed ndcg@5=%.4f recorded=%.4f delta=%+.4f',
                     result['metrics']['val']['ndcg@5'], recorded['best_val'], result['val_reproduction_delta'])
        if env.main:
            (out / 'metrics.json').write_text(json.dumps(result, indent=2, default=str), encoding='utf-8')
            row = {'kind': job['kind'], 'model': cfg['model']['name'], 'source_dataset': result['source_dataset'],
                   'dataset': data.name, 'test_context': opts['test_context'],
                   'run_id': Path(job['run']).name if job.get('run') else '', 'output': str(out)}
            for m in KEY_METRICS:
                for split in splits:
                    row[f'{split}_{m}'] = result['metrics'][split].get(m)
            append_summary(Path(opts['summary_root']) / f"{job['kind']}_{opts['tag']}.csv", row)
            LOG.info('Saved %s', out / 'metrics.json')
        del query, document
        if env.device.type == 'cuda':
            torch.cuda.empty_cache()
        barrier(env)


def append_summary(path, row):
    """Add (or replace, same ``output``) a row of a summary CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    existing, fields = [], list(row)
    if path.exists():
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
            fields = list(dict.fromkeys(list(reader.fieldnames or []) + fields))
    existing = [r for r in existing if r.get('output') != row['output']]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in existing + [row]:
            w.writerow(r)


def run_jobs(specs, targets, tag, target_root=None):
    """Jobs of the selected runs: re-evaluation on their own dataset, transfer to every other target."""
    jobs = []
    for r in specs:
        cfg = run_config(r['run'])
        recorded = {'best_val': r['best_val'], 'test': r['recorded_test'], 'test_context': r['recorded_test_context']}
        for target in targets or [None]:
            if target is None or target == cfg['dataset']['name']:
                jobs.append({'kind': 'reeval', 'run': str(r['run']), 'cfg': cfg, 'recorded': recorded,
                             'out': str(Path(r['run']) / 'reeval' / tag)})
            else:
                jobs.append({'kind': 'transfer', 'run': str(r['run']), 'cfg': cfg, 'target': target,
                             'target_root': target_root, 'source': cfg['dataset']['name'],
                             'out': str(Path(r['run']) / 'transfer' / target / tag)})
    return jobs


# ----------------------------------------------------------------------------------------------- CLI
def main(argv=None):
    p = argparse.ArgumentParser(description='Evaluate LMStaR runs (re-evaluation, transfer) and zero-shot backbones')
    p.add_argument('--runs', nargs='*', default=[], help='run directories')
    p.add_argument('--dataset', action='append', default=[], help='discover the runs of this dataset (repeatable)')
    p.add_argument('--select', default=None, help='with --dataset: all | top:N (best validation) | latest:N')
    p.add_argument('--filter', action='append', default=[], help='keep runs whose id contains this substring')
    p.add_argument('--output-root', default=DEFAULTS['output']['root'])
    p.add_argument('--backbone', default='Qwen3-Embedding-0.6B', help='run sub-folder <Dataset>/runs/<backbone>')
    p.add_argument('--method', nargs='+', default=list(METHODS), help='run sub-folders to discover')
    p.add_argument('--target-dataset', action='append', default=[], help='transfer: evaluate on this dataset')
    p.add_argument('--target-root', default=None, help="transfer: dataset.root of the target (default: the run's)")
    p.add_argument('--zero-shot', action='store_true', help='evaluate the backbone of --config without fine-tuning')
    p.add_argument('--config', action='append', default=[], help='--zero-shot: configs (model + dataset)')
    p.add_argument('--splits', nargs='+', default=['test'], choices=['val', 'test'])
    p.add_argument('--test-context', default='fit_plus_val', choices=list(TEST_CONTEXTS))
    p.add_argument('--tag', default=None, help='output sub-folder name (default: test_<context>)')
    p.add_argument('--gpus', type=int, default=1)
    p.add_argument('--set', action='append', default=[], metavar='SECTION.KEY=VALUE',
                   help='e.g. training.eval_batch_size=32 (per GPU)')
    p.add_argument('--no-predictions', action='store_true')
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--list', action='store_true', help='only list the candidate runs')
    a = p.parse_args(argv)
    tag = a.tag or f'test_{a.test_context}'
    opts = {'splits': [s for s in ('val', 'test') if s in a.splits], 'test_context': a.test_context, 'tag': tag,
            'set': a.set, 'no_predictions': a.no_predictions, 'overwrite': a.overwrite}
    if a.zero_shot:
        if not a.config:
            p.error('--zero-shot requires --config (model + dataset)')
        cfg = load(a.config, a.set, p.error)
        if not cfg['dataset'].get('name'):
            p.error('--zero-shot: dataset.name missing (configs/datasets/<d>.yaml)')
        label = cfg['model']['name'].rstrip('/').split('/')[-1]
        if cfg['model']['pooling'] != 'eos':
            label += f"_{cfg['model']['pooling']}"
        out = Path(cfg['output']['root']) / cfg['dataset']['name'] / 'zero_shot' / label / tag
        opts['summary_root'] = cfg['output']['root']
        launch(worker, ([{'kind': 'zero_shot', 'cfg': cfg, 'out': str(out)}], opts), gpus=a.gpus)
        return
    runs = [describe(r) for r in a.runs] + discover(a.output_root, a.dataset, a.backbone, a.method)
    seen = set()
    runs = [r for r in runs if not (str(r['run']) in seen or seen.add(str(r['run'])))]
    if a.list:
        print_table([r for r in runs if all(f in r['run_id'] for f in a.filter)])
        return
    if a.dataset and a.select is None and not a.runs:
        p.error('--dataset requires --select (or use --list)')
    explicit = {str(x) for x in a.runs}
    for r in runs:
        if str(r['run']) in explicit and not r['complete']:
            print(f'WARNING: incomplete run skipped (no best/ towers or config.yaml): {r["run"]}')
        if str(r['run']) in explicit and r['removed_features']:
            print(f'WARNING: run skipped, trained with removed features {r["removed_features"]}: {r["run"]}')
    chosen = select(runs, a.select if a.dataset else 'all', a.filter)
    if not chosen:
        p.error('No complete run selected')
    print_table(chosen)
    opts['summary_root'] = a.output_root
    specs = [{k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()} for r in chosen]
    launch(worker, (run_jobs(specs, a.target_dataset, tag, a.target_root), opts), gpus=a.gpus)


if __name__ == '__main__':
    main()
