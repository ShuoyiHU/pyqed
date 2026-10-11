"""Plot recorded CBE energies against sweeps and measured elapsed time."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_task(folder,record):
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    ax.plot(range(1,len(record['history'])+1),record['history'],'o-')
    ax.set(xlabel='Directional sweep',ylabel='Total energy / t',
           title=f"{record['width']}×{record['length']}, D={record['cap']} CBE")
    ax.grid(alpha=.2)
    fig.savefig(folder/'energy_vs_sweep.png',dpi=180)
    plt.close(fig)

def main(run):
    records=[]
    for folder in sorted(run.glob('task_*')):
        metadata=folder/'metadata.json'
        if not metadata.exists():continue
        m=json.loads(metadata.read_text());updates=folder/'updates.jsonl';rows=[]
        if updates.exists():
            for line in updates.read_text().splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:continue  # last write may be incomplete
        records.append((folder,m,rows))
    if not records:raise ValueError('No task outputs found')
    fig,axes=plt.subplots(len(records),2,figsize=(10,3*len(records)),squeeze=False,layout='constrained')
    for (folder,m,rows),ax in zip(records,axes):
        sites=m['width']*m['length'];completed=[r for i,r in enumerate(rows,1) if i%sites==0]
        result=folder/'result.json'
        energies=json.loads(result.read_text())['history'] if result.exists() else [r['energy'] for r in completed]
        ax[0].plot(range(1,len(energies)+1),energies,'o-')
        ax[1].plot([r['elapsed_seconds'] for r in rows],[r['energy'] for r in rows],'-')
        title=f"{m['width']}×{m['length']}, D={m['cap']} multiplets"
        for a in ax:a.set_title(title);a.set_ylabel('Total energy / t');a.grid(alpha=.2)
        ax[0].set_xlabel('Directional sweep');ax[1].set_xlabel('Solve elapsed time (s)')
    fig.savefig(run/'energy_vs_sweep_and_time.png',dpi=180)
    fig.savefig(run/'energy_vs_sweep_and_time.pdf')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);main(p.parse_args().run)
