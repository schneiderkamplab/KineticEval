from kineticeval.db import Database
from kineticeval.service import Service
from kineticeval.fixtures import demo_definition


def test_defaults_reuse_revision_but_inputs_and_references_version(tmp_path):
    service=Service(Database(tmp_path/'identity.db'))
    original=service.create_evaluation(demo_definition())['revision_id']
    run=service.create_run({'revision_id':original,'models':[{'name':'fake','backend':'fake'}]})['id']
    changed=service.revise(original,{'scorer':{'name':'normalized'}})
    assert changed['revision_id']==original
    assert changed['created'] is False
    assert service.results(run)['definition']['scorer']['name']=='exact'
    modified=service.revise(original,{'prompt_template':'Question: {input}'})
    assert modified['created']
    reverted=service.revise(modified['revision_id'],{'prompt_template':demo_definition()['prompt_template']})
    assert reverted['revision_id']==original
    example=demo_definition()['examples'][0]
    corrected=service.revise(original,{'overlays':{example['id']:{'target':'changed'}}})
    assert corrected['created']


def test_noop_overlay_does_not_create_a_revision(tmp_path):
    service=Service(Database(tmp_path/'noop.db'))
    definition=demo_definition()
    revision=service.create_evaluation(definition)['revision_id']
    example=definition['examples'][0]
    saved=service.revise(revision,{'overlays':{example['id']:{'target':example['target']}}})
    assert saved['revision_id']==revision
    assert saved['created'] is False
