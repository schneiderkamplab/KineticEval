import pytest
from kineticeval.core import Store
from kineticeval.diagnostics import diagnose


def test_reuse_and_retry(tmp_path):
    s = Store(tmp_path / 'test.db')
    key = s.request('fake', ['question'])
    assert key == s.request('fake', ['question'])
    assert key != s.request('fake', ['changed'])
    job = s.claim('fake')
    s.finish(key, job['token'], error='network')
    assert s.retry(key)
    job = s.claim('fake')
    s.finish(job['id'], job['token'], output='A')
    assert not s.retry(job['id'])
    with pytest.raises(ValueError):
        s.finish(job['id'], job['token'], output='B')


def test_recovery(tmp_path):
    s = Store(tmp_path / 'test.db')
    key = s.request('fake', ['question'])
    old = s.claim('fake', -1)
    new = s.claim('fake')
    assert new['id'] == key
    with pytest.raises(ValueError):
        s.finish(key, old['token'], output='A')
    s.finish(key, new['token'], output='A')


@pytest.mark.parametrize('text,target,correct', [('yes.', 'yes', True), ('The answer is A.', 'A', True), ('Det rigtige svar er A.', 'A', True), ('D', 'A', False), ('Not A, B', 'A', None)])
def test_diagnostics(text,target,correct):
    assert diagnose(text,target)['semantic_correct'] is correct
