"""Regenerate descriptive results from the completed frozen longer study."""
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path('results/longer-v1')


def main():
    plan = json.loads(Path('configs/study-v1.json').read_text())
    sources = {}

    def read(path):
        raw = path.read_bytes()
        sources[str(path)] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    rows = []
    for size in plan['sizes']:
        assert read(ROOT / size / 'status.json')['state'] == 'completed', size
        assert len(list((ROOT / size).glob('tuning/*/*/validation/metrics.json'))) == 8
        for method in plan['methods']:
            for seed in plan['seeds']:
                root = ROOT / size / 'main' / f'seed-{seed}' / method
                manifest = read(root / 'artifact-manifest.json')
                assert manifest['training']['steps'] == plan['main_updates']
                for mode, filename in [('deployed', 'evaluation/metrics.json'),
                                       ('policy', 'evaluation/policy-metrics.json')]+(
                    [('temperature', 'posthoc/temperature-metrics.json'),
                     ('constant', 'posthoc/constant-metrics.json')]
                    if method in ('sft', 'continued_sft') else []):
                    m = read(root / filename)
                    assert m['n'] == 3080
                    assert m.get('subset_limit') is None
                    if mode == 'temperature':
                        assert m['calibration_n'] == 1000
                    point = m['operating_points']['coverage_0.8']
                    rows.append(dict(size=size, method=method, seed=seed, mode=mode,
                                     accuracy=m['accuracy'], brier=m['correctness_brier'],
                                     coverage80=point['coverage'], error80=point['error'],
                                     source=str(root / filename)))
    aggregates = []
    for size in plan['sizes']:
        for method in plan['methods']:
            for mode in ('deployed', 'policy', 'temperature', 'constant'):
                rs = [r for r in rows if (r['size'], r['method'], r['mode']) == (size, method, mode)]
                if not rs:
                    continue
                assert len(rs) == 3
                aggregates.append(dict(size=size, method=method, mode=mode,
                                       accuracy_mean=mean(r['accuracy'] for r in rs),
                                       accuracy_seed_sd=stdev(r['accuracy'] for r in rs),
                                       brier_mean=mean(r['brier'] for r in rs),
                                       coverage80_mean=mean(r['coverage80'] for r in rs),
                                       error80_mean=mean(r['error80'] for r in rs)))
    lines = ['# Completed longer study: descriptive results', '',
             'All 24 tuning runs and 36 main runs completed. The frozen schedule totals 168,000 optimizer updates. Each main evaluation uses the full 3,080-example official BANKING77 test set; post-hoc calibration uses 1,000 reserved examples. Three seeds: 11, 22, 33.', '',
             'Initial SFT receives 4,000 updates; continued SFT, exact RL and sampled RL each receive another 4,000 from the same seed-matched SFT artifact. Thus continuation methods share exposure; SFT alone has less exposure. 0.8B and 4B use BF16 LoRA; 9B uses NF4 QLoRA.', '',
             '| Size | Method | Accuracy mean | Seed SD | Deployed Brier | Test coverage at calibration 80% target | Accepted-case error |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in aggregates:
        if r['mode'] == 'deployed':
            lines.append(f"| {r['size']} | {r['method']} | {r['accuracy_mean']:.2%} | {r['accuracy_seed_sd']:.2%} | {r['brier_mean']:.4f} | {r['coverage80_mean']:.2%} | {r['error80_mean']:.2%} |")
    lines += ['', 'Deployed confidence uses the supervised scalar head for SFT/continued SFT and the candidate-conditioned confidence policy for RL. These differ. The JSON also retains the same policy readout across all methods. Seed SD is not a confidence interval; these are descriptive averages, not significance tests. Calibration-selected thresholds do not force exactly 80% test coverage.', '',
              '| Size | Supervised method | Temperature Brier | Test coverage at calibration 80% target | Accepted-case error |',
              '|---|---|---:|---:|---:|']
    for r in aggregates:
        if r['mode'] == 'temperature':
            lines.append(f"| {r['size']} | {r['method']} | {r['brier_mean']:.4f} | {r['coverage80_mean']:.2%} | {r['error80_mean']:.2%} |")
    lines += ['', 'Temperature confidence is the calibrated selected-option probability, not a calibrated version of the scalar correctness head. This control tests a different confidence source. Accuracy is unchanged by temperature scaling.', '',
              '## First findings and limits', '',
              '- Continued supervision has the highest mean accuracy at 0.8B. Exact RL has the highest mean accuracy at 4B and 9B; sampled RL has lower mean accuracy than exact RL at every size.',
              '- The highest observed mean accuracy is 4B exact RL. This does not establish a significant win over every alternative or an intrinsic advantage of 4B over 9B; precision differs.',
              '- Continued SFT with temperature scaling has lower mean correctness Brier than either RL method at all three sizes. Confidence-aware RL therefore has not demonstrated a general advantage over simple calibration.',
              '- Brier, discrimination and accepted-case error answer different questions. The operational rows retain achieved coverage so calibration gains are not mistaken for universal deferral gains.',
              '- Paired group contrasts are available in `paired-analysis.md` when generated with `scripts/analyze_longer_study.py`. Broader transfer/robustness, remaining ablations, replicated precision checks and isolated serving benchmarks remain pending. No production default is selected.',
              '- The short pilot used different exposure and a 256-example test subset. Do not interpret differences from it as a controlled estimate of longer-training benefit.', '',
              'Regenerate with `python3 scripts/summarize_longer_study.py`. The companion JSON records every source metric/manifest hash, per-seed values and aggregate values. Original per-run uncertainty and predictions remain in their source directories.', '']
    (ROOT / 'summary.json').write_text(json.dumps(dict(scope='descriptive completed longer study; no paired significance inference', runs=rows, aggregates=aggregates, source_sha256=sources), indent=2)+'\n')
    (ROOT / 'summary.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
