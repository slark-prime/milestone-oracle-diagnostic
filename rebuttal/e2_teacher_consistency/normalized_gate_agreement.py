import json, sys
from collections import defaultdict
from pathlib import Path
ROOT=Path('/Users/wangzhuohan/Desktop/Projects/milestone_decompose_rl')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts/experiments'))
from taxonomy_agreement import alt_families, EVAL

OUT=ROOT/'data/logs/rl/teacher_consistency'
s0=defaultdict(int)
for l in open(OUT/'alt_stage0.jsonl'):
    d=json.loads(l)
    if d.get('accept'): s0[(d['key'][0],d['key'][1])]+=1
pr=defaultdict(int)
for l in open(OUT/'alt_probes.jsonl'):
    d=json.loads(l)
    if d.get('accept'): pr[(d['key'][0],d['key'][1])]+=1
panel={}
for l in open(ROOT/'data/logs/rl/oracle_panel_16k/gpt_oss_20b.jsonl'):
    d=json.loads(l); panel[(d['pid'],d['condition'])]=d['n_correct']
base_s0=defaultdict(dict)
for l in open(ROOT/'data/logs/rl/stage0_panel_16k/gpt_oss_20b.jsonl'):
    d=json.loads(l); base_s0[d['pid']][d['ms_idx']]=d['n_correct']
base_fams={json.loads(l)['pid']:json.loads(l) for l in open(EVAL)}

def tax(c1,c2,c3,gate):
    if c1>=1: return 'DIRECT'
    if c2>=1: return 'ROADMAP_GAP'
    if c3>=1: return 'MILESTONE_EXECUTION_GAP'
    return 'COMPOSITION_GAP' if gate else 'MISSING'

fams=alt_families()
results={}
for th_label, th in (('all',1.01),('frac>=2/3',2/3),('frac>=1/2',0.5)):
    agree=0; conf=defaultdict(int); rows=[]
    for f in fams:
        pid=f['pid']; n=len(f['milestones'])
        fr_alt=sum(1 for i in range(n) if s0[(pid,i)]>=1)/n
        nb=len(base_fams[pid]['milestones'])
        fr_base=sum(1 for i in range(nb) if base_s0.get(pid,{}).get(i,0)>=1)/nb
        if th>1:  # all-milestones 原版
            g_alt = fr_alt>=1.0; g_base = fr_base>=1.0
        else:
            g_alt = fr_alt>=th; g_base = fr_base>=th
        a=tax(pr[(pid,'C1_direct')],pr[(pid,'C2_descriptions')],pr[(pid,'C3_gold_answers')],g_alt)
        b=tax(panel.get((pid,'C1_direct'),0),panel.get((pid,'C2_descriptions'),0),panel.get((pid,'C3_gold_answers'),0),g_base)
        agree+=(a==b); conf[(b,a)]+=1
    n=len(fams)
    results[th_label]={'agree':agree,'n':n,'confusion':{f"{x} -> {y}":c for (x,y),c in sorted(conf.items(),key=lambda z:-z[1])}}
    print(f"{th_label:12s}: {agree}/{n} = {agree/n:.0%}")
    for (x,y),c in sorted(conf.items(),key=lambda z:-z[1]):
        if x!=y: print(f"    {x} -> {y}: {c}")

json.dump(results, open(OUT/'normalized_gate_agreement.json','w'), indent=2)
print('written:', OUT/'normalized_gate_agreement.json')
