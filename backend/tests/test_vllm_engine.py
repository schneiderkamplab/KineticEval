import sys
from types import SimpleNamespace
import pytest
from kineticeval.vllm_engine import VLLMModel


def test_vllm_preserves_prompt_and_execution_provenance(monkeypatch):
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(SamplingParams=lambda **kwargs:kwargs))
    seen=[]
    class Engine:
        def generate(self,prompts,params,use_tqdm):
            seen.append((prompts,params))
            return [SimpleNamespace(prompt_token_ids=[1,2],outputs=[SimpleNamespace(text='ANSWER: A',token_ids=[3,4],finish_reason='stop')])]
    model=VLLMModel.__new__(VLLMModel)
    model.spec={'name':'test','parameters':{'max_tokens':8,'temperature':0},'context_limit':64,'revision':'pinned','dtype':'bfloat16'}
    model.tokenizer=SimpleNamespace(chat_template=None,encode=lambda prompt,add_special_tokens:[1,2])
    model.engine=Engine()
    model.version='test'
    model.execution_mode='compiled'
    model.tp=2
    model.memory_fraction=.4
    result=model.generate_batch([{'model':model.spec,'messages':[{'role':'user','content':'Exact prompt'}]}])[0]
    assert seen==[([{'prompt_token_ids':[1,2]}],{'temperature':0,'top_p':1,'max_tokens':8,'seed':None})]
    assert result['response']=='ANSWER: A'
    assert result['metadata']['tensor_parallel_size']==2
    assert result['metadata']['engine']=='vllm'
    assert result['metadata']['rendered_chat_prompt']=='Exact prompt'
    model.spec['context_limit']=3
    with pytest.raises(ValueError,match='no truncation'):
        model.generate_batch([{'model':model.spec,'messages':[{'role':'user','content':'Exact prompt'}]}])


def test_vllm_constructor_uses_pinned_source_and_total_memory_fraction(monkeypatch):
    import kineticeval.vllm_engine as module
    options=[]
    monkeypatch.setattr(module,'resolve_source',lambda spec:'/cache/pinned')
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='test',LLM=lambda **kwargs:options.append(kwargs)))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    monkeypatch.delenv('VLLM_WORKER_MULTIPROC_METHOD',raising=False)
    module.VLLMModel({'name':'model','revision':'sha','context_limit':4096},.4,2,64)
    assert options[0]['model']=='/cache/pinned'
    assert options[0]['language_model_only'] is True
    assert options[0]['tensor_parallel_size']==2
    assert options[0]['gpu_memory_utilization']==.4
    assert options[0]['compilation_config']['cudagraph_capture_sizes']==[1,2,4,8,16,32,64]


def test_hrm_uses_eager_vllm_runtime(tmp_path, monkeypatch):
    import kineticeval.vllm_engine as module
    (tmp_path/'config.json').write_text('{"model_type":"hrm_text"}')
    options=[]
    monkeypatch.setattr(module,'resolve_source',lambda spec:str(tmp_path))
    monkeypatch.setitem(sys.modules,'vllm',SimpleNamespace(__version__='test',LLM=lambda **kwargs:options.append(kwargs)))
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a,**kw:object())))
    model=module.VLLMModel({'name':'hrm','revision':'sha'},.4)
    assert options[0]['enforce_eager'] is True
    assert model.execution_mode=='eager'
