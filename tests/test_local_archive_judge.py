import importlib.util
import json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('local_judge',Path(__file__).parents[1]/'scripts/local_archive_judge.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_local_judge_rejects_invented_evidence_and_truncation():
    row={'context':'Install the package. Then run the command.','candidates':[{'id':'tutorial','description':'Instructions'}]}
    value={'label':'tutorial','status':'labelled','evidence':['Install the package.'],'explanation':'Contains explicit steps.'}
    assert module.decode_judgment(row,json.dumps(value))['label']=='tutorial'
    with pytest.raises(ValueError):module.decode_judgment(row,json.dumps(value),True)
    value['evidence']=['This text was never supplied.']
    with pytest.raises(ValueError):module.decode_judgment(row,json.dumps(value))
    with pytest.raises(ValueError):module.decode_judgment(row,'I think this is a tutorial')
