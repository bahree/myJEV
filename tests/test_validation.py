import hashlib
import json
import pytest
from myjev.validation import checked_validation, select_trial


def test_validation_rejects_test_calibration_and_tampering(tmp_path):
    for name in ('test.jsonl', 'calibration.jsonl'):
        with pytest.raises(ValueError, match='validation.jsonl'):
            checked_validation(tmp_path/name)
    path=tmp_path/'validation.jsonl'
    path.write_text(json.dumps({'group':'validation'})+'\n')
    (tmp_path/'train.jsonl').write_text(json.dumps({'group':'train'})+'\n')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps({'sha256':{'validation':digest}}))
    assert checked_validation(path)[1] == digest
    (tmp_path/'train.jsonl').write_text(json.dumps({'group':'validation'})+'\n')
    with pytest.raises(ValueError, match='leakage'):
        checked_validation(path)
    path.write_text('{}\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        checked_validation(path)


def test_selection_rule_and_same_partition():
    trial={'validation_sha256':'fixed','accuracy':.7,'selection_nll':1.,'learning_rate':1e-4}
    better={**trial,'accuracy':.8,'selection_nll':2.}
    assert select_trial([trial,better]) is better
    tie={**trial,'learning_rate':3e-5}
    assert select_trial([trial,tie]) is tie
    with pytest.raises(ValueError, match='same nonempty'):
        select_trial([trial,{**trial,'validation_sha256':'other'}])
    with pytest.raises(ValueError, match='nonfinite'):
        select_trial([{**trial,'selection_nll':float('nan')}])


def test_continuation_data_rng_matches_uninterrupted_exposure():
    import random
    from myjev.data import training_order, request_from_row
    rows=[{'context':str(i),'instructions':'Select','candidates':[{'id':'a','description':'A'},
           {'id':'b','description':'B'}],'label':'a'} for i in range(7)]
    for offset in (0, 4, 7, 11):
        rng=random.Random(11)
        order, cursor=training_order(rows,rng)
        for _ in range(offset):
            if cursor == len(order):
                rng.shuffle(order); cursor=0
            request_from_row(rows[order[cursor]],rng); cursor+=1
        resumed_rng=random.Random(11)
        resumed_order,resumed_cursor=training_order(rows,resumed_rng,offset)
        assert (resumed_order,resumed_cursor)==(order,cursor)
        assert resumed_rng.getstate()==rng.getstate()
