import copy
import math
import pytest
from myjev.decision_comparison import (budget_reservation, canonical_response, normalize_hosted,
                                       request_views, systemone_payload)

ROW={'id':'r1','label':'billing','group':'g1','context':'Double charge.', 'instructions':'Select the banking support intent.',
     'candidates':[{'id':'billing','description':'Billing'},{'id':'tech','description':'Technical'}]}


def response(scores=None):
    return {'answers':{'route':{'type':'choice','choice':'billing','probabilities':scores or {'billing':.8,'tech':.2},'confidence':.6}}}


def test_renaming_preserves_semantic_choices_and_does_not_mutate_source():
    original=copy.deepcopy(ROW);views=list(request_views(ROW));assert len(views)==6 and ROW==original
    assert views==list(request_views(ROW))
    name,request,mapping=views[3]
    renamed=next(k for k,v in mapping.items() if v=='billing')
    assert 'billing' not in mapping
    answer=canonical_response({'selected_id':renamed,'selection_scores':{k:.8 if k==renamed else .2 for k in mapping}},request,mapping)
    assert answer['selected_id']=='billing' and answer['selection_scores']=={'billing':.8,'tech':.2}


def test_wire_payload_excludes_labels_groups_and_row_ids():
    payload=systemone_payload(ROW,'microsoft/microsoft-decision-1')
    assert set(payload)=={'model','state','questions'}
    assert payload['questions']['route']['criteria']=={'billing':'Billing','tech':'Technical'}
    assert 'label' not in payload and 'group' not in payload


def test_provider_confidence_is_separate_from_selected_probability():
    out=normalize_hosted(response(),ROW)
    assert out['confidence']==.8 and out['provider_confidence']==.6


@pytest.mark.parametrize('scores',[{'billing':float('nan'),'tech':.2},{'billing':float('inf'),'tech':.2},
                                  {'billing':True,'tech':0.},{'billing':.7,'tech':.2},
                                  {'billing':.8,'unexpected':.2},{'billing':.2,'tech':.8}])
def test_malformed_probabilities_fail_without_silent_repair(scores):
    with pytest.raises(ValueError):normalize_hosted(response(scores),ROW)


def test_missing_confidence_is_allowed_but_missing_distribution_is_not():
    r=response();del r['answers']['route']['confidence'];assert normalize_hosted(r,ROW)['provider_confidence'] is None
    del r['answers']['route']['probabilities']
    with pytest.raises(ValueError):normalize_hosted(r,ROW)


def test_budget_reserves_all_calls_at_full_context():
    assert budget_reservation(1396)==pytest.approx(1.921253376)
    with pytest.raises(ValueError):budget_reservation(0)
