import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('review_order',Path(__file__).parents[1]/'scripts/review_candidate_order.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_permutation_preserves_candidates_and_is_shared_across_models():
    candidates=[{'id':str(i),'description':f'Option {i}'} for i in range(77)]
    a=m.permuted_candidates(candidates,'test-1',101)
    assert a==m.permuted_candidates(candidates,'test-1',101)
    assert a!=m.permuted_candidates(candidates,'test-1',202)
    assert a!=m.permuted_candidates(candidates,'test-2',101)
    assert sorted(a,key=lambda x:int(x['id']))==candidates
    assert candidates[0]['id']=='0'
