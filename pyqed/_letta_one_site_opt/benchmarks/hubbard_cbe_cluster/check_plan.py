"""Check the frozen plan before any submission; report exact task mapping."""
import json,sys,os
from pathlib import Path
base=Path(sys.argv[1]);plan=json.loads((base/'plan.json').read_text())
revision=plan['config']['source_revision']
if revision.startswith('PENDING'):raise RuntimeError('Algorithm validation and source freezing are not finished')
selected=set(range(len(plan['tasks'])))
if os.environ.get('INDICES'):
    selected=set()
    for piece in os.environ['INDICES'].split(','):
        parts=piece.split('-')
        if len(parts)==1:selected.add(int(parts[0]))
        elif len(parts)==2:
            lo,hi=map(int,parts)
            if lo>hi:raise ValueError('Reversed index range')
            selected.update(range(lo,hi+1))
        else:raise ValueError('Invalid index range')
    if not selected or not selected<=set(range(len(plan['tasks']))):raise ValueError('Invalid task selection')
groups={}
for index,t in enumerate(plan['tasks']):
 assert t['method']=='cbe' and t['width'] in (2,4)
 assert 0<t['holes']<t['width']*t['length'] and t['cap']>0
 if index in selected:groups.setdefault(t['memory'],[]).append(index)
 print(f"{index}: {t['width']}x{t['length']}, D={t['cap']} multiplets, holes={t['holes']}, {t['memory']}")
if len(sys.argv)>2:(Path(sys.argv[2])/'submission_groups.tsv').write_text(''.join(f"{memory}\t{','.join(map(str,indices))}\n" for memory,indices in groups.items()))
print('Selected:',','.join(map(str,sorted(selected))))
print('Revision:',revision)
print('No requested time limit or array concurrency cap.')
