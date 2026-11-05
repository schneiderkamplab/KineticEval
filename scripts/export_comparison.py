"""Export compact comparisons without conflating uninspected failures with wrong answers."""
import csv
import json
import math
import sqlite3
from pathlib import Path

root=Path('artifacts')
campaign=json.loads((root/'campaign.json').read_text())
connection=sqlite3.connect(campaign['db'])
connection.row_factory=sqlite3.Row
rows=[]
for scope in ['pilot','full']:
    for benchmark,run_id in campaign[scope].items():
        query='''SELECT json_extract(i.model,'$.name') model, count(*) total,
          sum(g.status='succeeded') generated,
          sum(sc.status='succeeded') scored,
          sum(CASE WHEN sc.status='succeeded' THEN json_extract(sc.result,'$.value') ELSE 0 END) correct,
          sum(CASE WHEN d.status='succeeded' AND json_extract(d.result,'$.finding.semantic_correct') IS NOT NULL THEN 1 ELSE 0 END) semantic_known,
          sum(CASE WHEN d.status='succeeded' AND json_extract(d.result,'$.finding.semantic_correct')=1 THEN 1 ELSE 0 END) semantic_correct,
          sum(CASE WHEN json_extract(g.result,'$.metadata.finish_reason')='length' THEN 1 ELSE 0 END) token_limited,
          sum(CASE WHEN j.status='succeeded' THEN 1 ELSE 0 END) inspected
          FROM items i JOIN jobs g ON g.id=i.generation_id
          LEFT JOIN jobs sc ON sc.id=i.score_id LEFT JOIN jobs d ON d.id=i.diagnostic_id
          LEFT JOIN jobs j ON j.id=i.inspector_id WHERE i.run_id=? GROUP BY 1'''
        for result in connection.execute(query,(run_id,)):
            row={**dict(result),'benchmark':benchmark,'scope':scope,'run_id':run_id}
            for key in ['total','generated','scored','correct','semantic_known','semantic_correct','token_limited','inspected']:
                row[key]=row[key] or 0
            n=row['scored'] or 0
            if n:
                p=row['correct']/n; z=1.959963984540054; denom=1+z*z/n
                center=(p+z*z/(2*n))/denom
                margin=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
                row.update(accuracy=p,ci95_low=center-margin,ci95_high=center+margin)
            else:row.update(accuracy=None,ci95_low=None,ci95_high=None)
            row['complete']=n==row['total']
            row['inspector_status']='not run' if not row['inspected'] else 'partially assessed'
            rows.append(row)
(root/'comparison.json').write_text(json.dumps({'protocol':'zero-shot generated-answer QA, greedy BF16, thinking disabled, 64 new tokens',
    'mmlu':'Inspect Evals deduplicated English test split (13,937 examples, 57 subjects)',
    'intervals':'Wilson 95% intervals for observed strict scores; no adjustment for prompt or token-limit artifacts',
    'rows':rows},indent=2))
with (root/'comparison.csv').open('w') as handle:
    writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
for r in rows:
    if r['scope']=='full':
        print(r['benchmark'],r['model'],f"{r['generated']}/{r['total']} generated",f"{r['scored']}/{r['total']} scored",r['accuracy'],f"token-limited={r['token_limited']}")
