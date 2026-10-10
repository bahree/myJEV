"""Plot paired decision changes from the saved hosted/local diagnostic."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path('results/decision-comparison-v1')
    source = root / 'summary.json'
    models = json.loads(source.read_text())['models']
    names = ['myjev-4b', 'myjev-4b-rl', 'decision-1']
    views = ['reverse', 'shuffle', 'renamed-ids', 'whitespace', 'instruction-paraphrase']
    values = np.array([[models[name]['modes']['native-selection']['views'][view]['changed_count']
                        for view in views] for name in names])
    fig, ax = plt.subplots(figsize=(10.5, 4.1), layout='constrained')
    ax.imshow(values, cmap='Blues', vmin=0, vmax=max(8, int(values.max())), aspect='auto')
    ax.set_xticks(range(5), ['Reverse order', 'Shuffle order', 'Rename IDs', 'Whitespace', 'Paraphrase'])
    ax.set_yticks(range(3), ['myJEV 4B supervised', 'myJEV 4B exact RL', 'Decision-1 (hosted)'])
    for i in range(3):
        for j in range(5):
            ax.text(j, i, f'{values[i, j]} / 64', ha='center', va='center',
                    color='white' if values[i, j] > max(8, int(values.max())) * .55 else '#162d3d', fontsize=12)
    ax.tick_params(length=0, pad=12)
    ax.spines[:].set_visible(False)
    ax.set_title('How often did the same request get a different answer?', pad=20, fontsize=14)
    fig.supxlabel('Counts compare each view with the original answer; all columns share the same 64 cases.\nZero observed changes do not establish invariance. Local aliases hide external candidate IDs.', fontsize=9)
    output = root / 'answer-changes.png'
    fig.savefig(output, dpi=180)
    plt.close(fig)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (root / 'figure-sources.json').write_text(json.dumps({
        'source': str(source), 'source_sha256': sha(source),
        'generator': 'scripts/plot_decision_comparison.py', 'generator_sha256': sha(Path(__file__)),
        'output': str(output), 'output_sha256': sha(output),
        'scope': 'Observed flips on 64 paired test cases. Columns are dependent. No population-invariance or matched-hardware claim.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
