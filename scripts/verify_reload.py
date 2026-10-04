import argparse
import json
from pathlib import Path
import torch
from myjev.inference import DecisionModel
p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
p.add_argument('--device',default='cuda:0')
p.add_argument('--output',required=True)
a=p.parse_args()
root=Path(a.run)
model=DecisionModel.load(root/'artifact',device=a.device)
state=torch.load(root/'resume.pt',map_location='cpu',weights_only=False)
params=dict(model.network.named_parameters())
for name,value in state['parameters'].items():
    torch.testing.assert_close(params[name].detach().cpu(),value,atol=0,rtol=0)
result={'parameter_reload_exact':True,'parameters_checked':len(state['parameters']),
        'artifact_revision':model.manifest['artifact_revision']}
fixture=root/'reload-fixture.json'
if fixture.exists():
    recorded=json.loads(fixture.read_text())
    actual=model.score(recorded['request'])
    assert actual==recorded['expected'],(actual,recorded['expected'])
    result['in_memory_vs_reload_response_exact']=True
else:
    result['in_memory_vs_reload_response_exact']=None
    result['note']='earlier checkpoint predates in-memory response recording; parameter reload verified'
if model.manifest['precision']=='nf4':
    base=model.network.backbone.get_base_model()
    assert base.get_input_embeddings().weight.dtype==torch.float32
    assert base.get_output_embeddings().weight.dtype==torch.float32
    result['nonquantized_embeddings_fp32']=True
Path(a.output).write_text(json.dumps(result,indent=2))
