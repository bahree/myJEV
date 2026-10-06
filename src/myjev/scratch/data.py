"""Original controlled routing fixtures with explicit conditional probabilities."""
import hashlib
import json
import random
from pathlib import Path
from myjev.data import validate_isolation, write_jsonl

COLORS = ('red', 'blue', 'green', 'yellow')
INSTRUCTIONS = 'Choose the signal color. Two listed signals are equally likely. Choose other if the color has no option.'
TEMPLATES = {
    'train': 'signal: {signal}; case {nonce}',
    'validation': 'case {nonce}; signal: {signal}',
    'calibration': 'signal = {signal}; case {nonce}',
    'test': 'case {nonce}; signal = {signal}',
}


def build(directory, seed=42, counts=None, version=1):
    if version not in (1, 2):raise ValueError("unknown generator version")
    path=Path(directory)
    if path.exists() and any(path.iterdir()):
        raise ValueError('dataset directory must be empty')
    counts=counts or dict(train=2048,validation=256,calibration=256,test=512)
    parts={}
    for offset, (split,n) in enumerate(counts.items()):
        rng=random.Random(seed+offset);rows=[]
        for i in range(n):
            # Disjoint surface templates, plus held-out ambiguous combinations at test.
            pairs = [('red','blue'),('green','yellow')] if split != 'test' else [('red','green'),('blue','yellow')]
            signals=list(rng.choice(pairs)) if i%4==0 else [rng.choice(COLORS)]
            available=rng.sample(list(COLORS),3)
            candidates=[dict(id=c,description=c) for c in available]+[dict(id='other',description='other')]
            rng.shuffle(candidates)
            distribution={c['id']:0. for c in candidates}
            for color in signals:
                distribution[color if color in available else 'other'] += 1/len(signals)
            latent=rng.choice(signals)
            label=latent if latent in available else 'other'
            nonce=f'{split}-{i:05d}'
            template=TEMPLATES[split]
            if version==2 and split=='train':
                template=rng.choice(['signal: {signal}; case {nonce}',
                                     'case {nonce}; signal {signal}',
                                     'record {nonce}; signal: {signal}; end'])
                nonce='x'*rng.randint(1,20)+f'-{i:05d}'
            context=template.format(signal=' or '.join(signals),nonce=nonce)
            rows.append(dict(id=nonce,group=nonce,context=context,instructions=INSTRUCTIONS,
                             candidates=candidates,label=label,known_probabilities=distribution,
                             ambiguous=len(signals)>1,template=split))
        parts[split]=rows
    validate_isolation(parts)
    path.mkdir(parents=True,exist_ok=True)
    for split,rows in parts.items():write_jsonl(path/f'{split}.jsonl',rows)
    manifest=dict(generator=f'scratch-routing-v{version}',seed=seed,counts=counts,
                  license='CC0-1.0 for these original synthetic fixtures',
                  limitations=['Small color vocabulary; not broad language understanding.',
                               'Template and test ambiguity-combination holdouts; clean task rule is shared.',
                               'Nonce includes split name; it identifies a case but carries no label information.',
                               'One latent label per row; known probabilities are evaluator-only metadata.'],
                  files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in path.glob('*.jsonl')})
    (path/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest
