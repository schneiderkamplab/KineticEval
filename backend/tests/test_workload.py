import asyncio
from sqlalchemy import select,func
from kineticeval.db import Database,Run,Job
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition
from kineticeval.worker import Worker
from kineticeval.workload import preview


def test_workload_preview_is_read_only_and_reuses_cached_work(tmp_path):
    service=Service(Database(tmp_path/'preview.db'))
    definition=demo_definition();definition['inspector']={'enabled':True,'provider':'fake','audit_fraction':1,'audit_min':0,'audit_max':100}
    revision=service.create_evaluation(definition)['revision_id']
    request={'revision_id':revision,'models':[{'name':'fake','backend':'fake'}]}
    empty=preview(service,{'runs':[request]})
    assert empty['generation_jobs']=={'new':3}
    assert empty['groups'][0]['inspection']['exact'] is False
    with service.db.session() as s:
        assert s.scalar(select(func.count()).select_from(Job))==0
        assert s.scalar(select(func.count()).select_from(Run))==0
    service.create_run(request);asyncio.run(Worker(service.db).drain())
    cached=preview(service,{'runs':[request]})
    assert cached['generation_jobs']=={'reusable':3}
    assert cached['groups'][0]['scoring']=={'reusable':3}
    assert cached['groups'][0]['inspection']['exact']
    assert cached['known_inspection_jobs']=={'reusable':3}
    fresh=preview(service,{'runs':[{**request,'fresh':True}]})
    assert fresh['generation_jobs']=={'new':3}


def test_preview_bounds_include_caps_and_deduplicate_generations(tmp_path):
    service=Service(Database(tmp_path/'bounds.db'))
    definition=demo_definition();definition['inspector']={'enabled':True,'provider':'fake','failure_inspector_limit':1,'audit_fraction':0,'audit_min':0,'audit_max':0}
    revision=service.create_evaluation(definition)['revision_id']
    result=preview(service,{'runs':[{'revision_id':revision,'models':[{'name':label,'model_identifier':'physical','backend':'fake'} for label in ['one','two']]}]})
    assert result['response_slots']==6
    assert result['generation_jobs']=={'new':3}
    assert all(g['inspection']['maximum']==1 for g in result['groups'])
    assert result['reports_maximum']==1
