import pytest
from myjev.judging import prompt, validate_judgment

ROW={'id':'p-format','group':'g','split':'test','rubric':'format','context':'First install the package. Then run the command.',
     'instructions':'Classify format','candidates':[{'id':'0','description':'tutorial'},{'id':'1','description':'opinion'}],
     'label':'0','audit':True,'url':'private provenance'}
GOOD={'label':'0','status':'labelled','evidence':['First install the package.'],'explanation':'Contains procedural steps.'}


def test_judge_cannot_see_labels_or_audit_membership():
    value=prompt(ROW)
    assert set(value)=={'rubric','candidates','text'}
    assert value['text']==ROW['context']
    assert prompt(ROW)==value


def test_evidence_is_validated_against_exact_visible_text():
    assert validate_judgment(ROW,GOOD)==GOOD
    with pytest.raises(ValueError,match='exact excerpt'):
        validate_judgment(ROW,{**GOOD,'evidence':['A made-up passage.']})
    with pytest.raises(ValueError,match='unknown candidate'):
        validate_judgment(ROW,{**GOOD,'label':'unsupported-id'})


def test_abstention_is_not_a_confident_label():
    assert validate_judgment(ROW,{'label':None,'status':'uncertain','evidence':[],
                                 'explanation':'No defensible decision.'})['label'] is None
    with pytest.raises(ValueError,match='candidate ID'):
        validate_judgment(ROW,{**GOOD,'label':None})
    with pytest.raises(ValueError):
        validate_judgment(ROW,{**GOOD,'reviewed':True})


def test_machine_export_preserves_provenance_and_abstentions(tmp_path):
    import hashlib
    import json
    import subprocess
    import sys
    from pathlib import Path
    rows=[{**ROW,'id':'train-format','group':'train-group','split':'train'},
          {**ROW,'id':'test-format','group':'test-group','split':'test'}]
    labels=[]
    for i,row in enumerate(rows):
        labels.append({'id':row['id'],'label_source':'llm','human_reviewed':False,'reviewed':False,
            'visible_text_sha256':hashlib.sha256(row['context'].encode()).hexdigest(),
            'valid':True,'resolved_model':None,'prompt_sha256':'fixture',
            **GOOD,'status':'labelled' if i==0 else 'uncertain'})
    requests=tmp_path/'requests.jsonl'; source=tmp_path/'labels.jsonl'
    requests.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    source.write_text(''.join(json.dumps(r)+'\n' for r in labels))
    out=tmp_path/'frozen'
    subprocess.run([sys.executable,'scripts/finalize_machine_annotations.py','--requests',str(requests),
                    '--labels',str(source),'--output',str(out)],check=True)
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['human_reviewed'] is False
    assert manifest['excluded']==1
    assert manifest['counts']['test']==0
    assert json.loads((out/'train.jsonl').read_text())['label_source']=='llm'
    labels[0]['reviewed']=True
    source.write_text(''.join(json.dumps(r)+'\n' for r in labels))
    result=subprocess.run([sys.executable,'scripts/finalize_machine_annotations.py','--requests',str(requests),
                           '--labels',str(source),'--output',str(tmp_path/'invalid')],capture_output=True)
    assert result.returncode!=0


def test_machine_reference_metrics_are_explicit():
    from myjev.metrics import summarize
    rows=[{'id':'one','group':'one','label':'a','selected_id':'a','confidence':.7,
           'selection_scores':{'a':.7,'b':.3},'label_source':'llm','human_reviewed':False}]
    result=summarize(rows)
    assert result['reference_label_source']=='llm'
    assert 'not independently established human correctness' in result['metric_semantics']


def test_span_evidence_preserves_source_and_rejects_unknown_ids():
    from myjev.judging import source_spans, validate_span_judgment
    text = "Let’s install\n\t the package. " * 40
    row = {'context': text, 'candidates': [{'id': '0'}]}
    spans = source_spans(text)
    assert ''.join(spans.values()) == text
    value = {'label': '0', 'status': 'labelled', 'evidence_ids': ['s0000'], 'explanation': 'Steps supplied.'}
    assert validate_span_judgment(row, value)['evidence'] == [text[:400]]
    for ids in (['invented'], ['s0000', 's0000'], []):
        with pytest.raises(ValueError):
            validate_span_judgment(row, {**value, 'evidence_ids': ids})
    with pytest.raises(ValueError):
        validate_span_judgment(row, {**value, 'label': 0})
