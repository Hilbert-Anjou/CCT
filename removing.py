import torch
from torch import nn
from model_SICH_pretrain_CCT import PreTranNet
from Compact_Transformers.src import cct_14_7x2_224
import torch.onnx
from Compact_Transformers.src.text import text_cct_2
from tensorboardX import SummaryWriter
"""
model=torch.load('model/CCT_pretrain_0.80auc.pt')
new_model = torch.nn.Sequential( *( list(model.children())[1:2] ) )
print(new_model)
torch.save(new_model,'model/CCT_only_0.80auc.pt')
"""
#model = cct_14_7x2_224(positional_embedding='sine')
model=text_cct_2(kernel_size=1)
#model=torch.load('model/model_nonseq.pt')
#input=torch.randn(1,24, 825).to('cuda:0')
#input=torch.randn(1, 3, 224, 224)
#input2=model.tokenizer(input)
#torch.onnx.export(model.tokenizer, input, 'model/CCTtoken224.onnx')
#torch.onnx.export(model.classifier, input2, 'model/CCTclass224.onnx')
#torch.onnx.export(model.classifier.blocks[1].self_attn, input2, 'model/CCTattention224.onnx')

with SummaryWriter(comment='Net') as w:
    w.add_graph(model,(input,))
