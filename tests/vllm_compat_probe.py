"""Smoke-test installed vLLM or a source tag selected through PYTHONPATH.

Source-mode runs reuse external dependencies; they do not validate a full wheel
installation or GPU serving. Run each plugin in a separate process.
"""
import argparse
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys

args=argparse.ArgumentParser()
args.add_argument('version')
args.add_argument('kind',choices=['tool','reasoning','runtime'])
args.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
a=args.parse_args()
root=a.root
v=tuple(map(int,a.version.split('.')))
if a.kind=='runtime':
 import torch
 print(json.dumps({'python':sys.version.split()[0], 'vllm':md.version('vllm'),'torch':torch.__version__,'cuda_build':torch.version.cuda,'cuda_available':torch.cuda.is_available(),'cuda_packages':{d.metadata['Name']:d.version for d in md.distributions() if d.metadata['Name'].startswith(('nvidia-','cuda-'))}}))
 sys.exit()
from tokenizers import AddedToken, Tokenizer, models
from transformers import PreTrainedTokenizerFast
if v < (0,15,0):
 from vllm.entrypoints.openai.protocol import ChatCompletionRequest
else:
 from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
backend=Tokenizer(models.WordLevel({'[UNK]':0},unk_token='[UNK]'))
backend.add_special_tokens([AddedToken(t,special=True) for t in ['<tool_call>','</tool_call>','<think>','</think>']])
tok=PreTrainedTokenizerFast(tokenizer_object=backend)
if a.kind=='tool':
 suffix='_v0.12.0' if v <= (0,12,0) else '_v0.13.0' if v < (0,15,0) else '_v0.15.0' if v < (0,19,0) else ''
else:
 if v <= (0,12,0):
  print(json.dumps({'skipped':'README does not claim reasoning support on <=0.12'}));sys.exit()
 suffix='_v0.13.0' if v < (0,15,0) else ''
path=root/'tools'/f'bielik_vllm_{a.kind}_parser{suffix}.py'
spec=importlib.util.spec_from_file_location('matrix_plugin',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
request=ChatCompletionRequest(model='test',messages=[{'role':'user','content':'test'}])
if a.kind=='tool':
 parser=module.BielikToolParser(tok,[]) if v >= (0,19,0) else module.BielikToolParser(tok)
 result=parser.extract_tool_calls('<tool_call>{"name":"calc","arguments":{"x":2}}</tool_call>',request)
 assert result.tools_called and result.tool_calls[0].function.name=='calc'
 assert json.loads(result.tool_calls[0].function.arguments)=={'x':2}
else:
 for thinking in [False,True]:
  parser=module.BielikReasoningParser(tok,chat_template_kwargs={'enable_thinking':thinking})
  text='<think>analysis</think>answer'
  result=parser.extract_reasoning(text,request)
  assert result == (('analysis','answer') if thinking else (None,text)),result
print(json.dumps({'passed':a.kind,'file':path.name}))
