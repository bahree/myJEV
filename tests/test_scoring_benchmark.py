"""Exercise generation controls with a real tiny random causal model on CPU."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
from myjev.inference import DecisionModel
from myjev.model import DecisionNetwork


def test_reference_scoring_and_generation_paths(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('scoring_paths',Path(__file__).parents[1]/'scripts/benchmark_scoring_paths.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    vocab={'[UNK]':0,'[EOS]':1,'[PAD]':2,**{f'A{i}':i+3 for i in range(32)}}
    raw=Tokenizer(WordLevel(vocab,unk_token='[UNK]'));raw.pre_tokenizer=Whitespace()
    tokenizer=PreTrainedTokenizerFast(tokenizer_object=raw,unk_token='[UNK]',eos_token='[EOS]',pad_token='[PAD]')
    torch.manual_seed(11)
    backbone=GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab),n_embd=16,n_layer=1,n_head=2,n_positions=4096,bos_token_id=1,eos_token_id=1,pad_token_id=2))
    manifest={'aliases':[f'A{i}' for i in range(32)],'alias_ids':list(range(3,35)), 'max_candidates':32,'max_tokens':4096,
              'confidence_mode':'scalar','artifact_revision':'tiny-fixture'}
    model=DecisionModel(DecisionNetwork(backbone),tokenizer,manifest)
    monkeypatch.setattr(module.DecisionModel,'load',lambda *a,**k:model)
    monkeypatch.setattr(sys,'argv',['benchmark','--artifact','fixture','--output',str(tmp_path/'out'),'--device','cpu','--repeats','1','--warmup','0'])
    module.main()
    report=json.loads((tmp_path/'out/report.json').read_text())
    assert len(report['results'])==12
    assert all('error' not in r for r in report['results'])
    for r in report['results']:
        output=r['outputs'][0]
        if r['path']=='one_token':
            assert output['format_valid'] and output['output_tokens']==1
        if r['path']=='direct':assert output['confidence'] is not None
        else:assert output['confidence'] is None
    # Malformed generated text is recorded as invalid; it must not fabricate a label.
    for r in report['results']:
        for output in r['outputs']:
            if not output['format_valid']:assert output['selected_id'] is None
