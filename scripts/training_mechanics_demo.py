"""CPU-only arithmetic lab: LoRA, one supervised update, and confidence reward."""
import json
import torch
from myjev.objectives import rewards


def run():
    frozen_weight=torch.eye(4)
    a=torch.tensor([[1.,0.,-1.,0.]])
    b=torch.tensor([[.1],[0.],[-.1],[.2]])
    x=torch.tensor([1.,2.,0.,-1.])
    updated=frozen_weight@x+b@(a@x)
    logits=torch.nn.Parameter(torch.tensor([0.,0.,0.]))
    optimizer=torch.optim.SGD([logits],lr=.3)
    before=logits.detach().softmax(-1).tolist()
    optimizer.zero_grad()
    loss=torch.nn.functional.cross_entropy(logits[None,:],torch.tensor([1]))
    loss.backward();gradient=logits.grad.tolist();optimizer.step()
    after=logits.detach().softmax(-1).tolist()
    grid_rewards=rewards(torch.tensor([[1.,0.]]))[0]
    return {'scope':'Constructed arithmetic examples, no backbone or dataset; not model performance.',
            'lora':{'base_parameters':16,'rank1_parameters':8,'input':x.tolist(),'output':updated.tolist()},
            'one_sft_update':{'target_index':1,'before_probabilities':before,'cross_entropy':float(loss.detach()),'gradient':gradient,'after_probabilities':after},
            'reward_examples':[{'correct':c,'confidence':q,'reward':float(grid_rewards[0 if c else 1,round(q*20)])} for c,q in [(1,.9),(1,.2),(0,.9),(0,.2)]]}

if __name__=='__main__':print(json.dumps(run(),indent=2))
