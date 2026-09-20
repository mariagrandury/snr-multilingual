"""Read-only analysis of repo inputs; write paper-local audit tables only.
Uses the report snapshot underlying the requested figures. No repo pipelines run.
"""
from pathlib import Path
import csv, json, math, statistics, collections, itertools, hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
A=ROOT/'src/signal-and-noise/analysis'
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def write(name,rows):
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def num(v):
    try: return float(v)
    except (ValueError,TypeError): return float('nan')
def corr(xs,ys):
    if len(xs)<5: return float('nan')
    mx,my=statistics.mean(xs),statistics.mean(ys)
    a=sum((x-mx)**2 for x in xs);b=sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x,y in zip(xs,ys))/math.sqrt(a*b) if a*b>0 else float('nan')
SIZES=['175M','350M','600M','1B','1.7B']
mask={r['task']:r for r in read(A/'rq00_gate_and_curves/pretraining/predictivity/above_random_mask.csv')}
tasks={r['task'] for r in read(A/'rq02_decision_accuracy/pretraining/predictivity/da_early_small_per_task.csv')}
tasks.discard('bpb_macro');tasks.discard('train_loss')
report=ROOT/'src/signal-and-noise/data/ladder-report/ladder_report.csv'
raw=read(report)
cells={r['cell']:r for r in raw if r['cell']}
valid={c for c,r in cells.items() if r['size'] in SIZES and num(r['run__complete'])==1 and num(r['run__diverged'])==0}
headline={c for c in valid if cells[c]['seed']=='1904' and cells[c]['scheme'] in ('A','B')}
print('snapshot cells:',len(cells),'healthy analysis runs:',len(valid),'headline runs:',len(headline))
print('analysis sizes:',dict(collections.Counter(cells[c]['size'] for c in valid)))
bycell=collections.defaultdict(list)
for r in raw:
    if r['cell'] in headline: bycell[r['cell']].append(r)
# task -> size -> variant -> {checkpoint percentage: score}; exact shared tenths.
data=collections.defaultdict(lambda:collections.defaultdict(dict))
for c,rs in bycell.items():
    rs.sort(key=lambda r:int(r['iter']))
    meta=rs[-1];target=int(float(meta['run__target_iters']))
    final=next((r for r in rs if int(r['iter'])==target),None)
    if final is None: continue
    variant=(meta['L'],meta['arch'],meta['scheme'],meta['seed'])
    for t in tasks:
        col='bpb__'+t[4:] if t.startswith('bpb_') else 'bench__'+t
        if not math.isfinite(num(final.get(col))): continue
        curve={}
        for k in range(1,11):
            row=min(rs,key=lambda r:abs(int(r['iter'])/target-k/10))
            if abs(int(row['iter'])/target-k/10)>.001: continue
            value=num(row.get(col))
            if math.isfinite(value):curve[k*10]=value
        data[t][meta['size']][variant]=curve

def da(left,right):
    common=sorted(left.keys()&right.keys());n=len(common)*(len(common)-1)//2
    if n<3:return None
    agree=sum(((left[a]>left[b])-(left[a]<left[b]))==((right[a]>right[b])-(right[a]<right[b])) for a,b in itertools.combinations(common,2))
    return agree/n,n
rows=[];bytask={}
for t,sizes in sorted(data.items()):
    ref={v:c[100] for v,c in sizes.get('1.7B',{}).items() if 100 in c}
    for size,variants in sizes.items():
        final={v:c[100] for v,c in variants.items() if 100 in c}
        for pct in range(10,101,10):
            early={v:c[pct] for v,c in variants.items() if pct in c}
            for kind,target in [('reference',ref),('checkpoint',final)]:
                ans=da(early,target)
                if ans is None:continue
                value,n=ans
                bytask[(t,size,pct,kind)]=value
                m=mask.get(t,{})
                # A missing chance baseline (BPB, LAMBADA) is not a failed gate.
                gate=math.isfinite(num(m.get('random_baseline')))
                passes=not gate or (m.get(size)=='1' and (kind=='checkpoint' or m.get('1.7B')=='1'))
                rows.append(dict(task=t,size=size,percent=pct,kind=kind,da=value,n_pairs=n,passes_gate=passes))
write('verified_da_ten_checkpoints.csv',rows)
summary=[]
for kind in ['reference','checkpoint']:
 for size in SIZES:
  for pct in range(10,101,10):
   for group in ['bpb','benchmarks']:
    sub=[r for r in rows if r['kind']==kind and r['size']==size and r['percent']==pct and r['passes_gate'] and (r['task'].startswith('bpb_'))==(group=='bpb')]
    if sub:summary.append(dict(kind=kind,size=size,percent=pct,group=group,mean_da=statistics.mean(r['da'] for r in sub),tasks=len(sub)))
write('verified_da_summary.csv',summary)
# Repository RQ4 SNR measurements; restrict size DA to the canonical 1.7B target
# and replace sparse checkpoint DA with all nine pre-final tenths.
snr=read(A/'rq03_noise_and_snr/pretraining/predictivity/snr_variants_per_task.csv')
variants=sorted({c[len('snr_'):-len('_175M')] for c in snr[0] if c.startswith('snr_') and c.endswith('_175M')})
language_tasks=collections.defaultdict(set)
for row in snr:
    task=row['task']; language=mask.get(task,{}).get('language')
    if language and task in tasks: language_tasks[language].add(task)
eligible={(r['task'],r['size'],r['percent'],r['kind']) for r in rows if r['passes_gate']}
points=collections.defaultdict(lambda:([],[]))
for r in snr:
 t=r['task'];lang=mask.get(t,{}).get('language')
 if not lang or lang in ('multi','??') or t not in tasks or len(language_tasks[lang])<2:continue
 for v in variants:
  for size in SIZES:
   x=num(r.get('snr_'+v+'_'+size))
   if not math.isfinite(x) or x<=0:continue
   for kind,pcts in [('size',[100]),('checkpoint',range(10,100,10))]:
    if kind=='size' and size=='1.7B':continue
    for pct in pcts:
     key=(t,size,pct,'reference' if kind=='size' else 'checkpoint')
     y=bytask.get(key) if key in eligible else None
     if y is not None:
      xs,ys=points[(lang,kind,v)];xs.append(math.log10(x));ys.append(y)
correlations=[]
for (lang,kind,v),(xs,ys) in sorted(points.items()):
 value=corr(xs,ys)
 if math.isfinite(value):correlations.append(dict(language=lang,kind=kind,variant=v,pearson_r=value,n=len(xs)))
write('verified_surrogates_by_language.csv',correlations)
print('DA full-grid summary:')
for size in SIZES:
 for group in ['bpb','benchmarks']:
  sub=[r for r in summary if r['kind']=='reference' and r['size']==size and r['group']==group]
  print(size,group,[(r['percent'],round(r['mean_da'],3),r['tasks']) for r in sub])
print('Surrogate leaders:')
for lang in ['en','de','es','fr','ru','zh','ar','th','tr','vi','eu']:
 print(lang,[(kind,[(r['variant'],round(r['pearson_r'],3),r['n']) for r in sorted([r for r in correlations if r['language']==lang and r['kind']==kind],key=lambda r:r['pearson_r'],reverse=True)[:1]]) for kind in ['size','checkpoint']])
audit={'source':str(report.relative_to(ROOT)),'sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'all_runs':len(cells),'healthy_175M_to_1_7B':len(valid),'headline_seed1904_A_B_runs':len(headline),'sizes':dict(collections.Counter(cells[c]['size'] for c in valid)),'fractions':[i/10 for i in range(1,11)],'min_pairs':3,'checkpoint_matching_tolerance':0.001,'surrogate_size_reference':'1.7B','surrogate_checkpoint_fractions':[i/10 for i in range(1,10)],'snr_source':'rq03_noise_and_snr/pretraining/predictivity/snr_variants_per_task.csv'}
(HERE/'verified_results_provenance.json').write_text(json.dumps(audit,indent=2)+'\n')
# Cross-task comparisons between accuracy families (no mixed loss/accuracy signs).
def scores(task,size,pct):
 return {v:c[pct] for v,c in data[task].get(size,{}).items() if pct in c}
def gated(task,size):
 m=mask.get(task,{});return math.isfinite(num(m.get('random_baseline'))) and m.get(size)!='1'
def sustained(values):
 good=[(k,v) for k,v in values if v is not None]
 return next((k for i,(k,v) in enumerate(good) if all(w>=.75 for _,w in good[i:])),None)
cross=[]
for pf,tf in [('hellaswag','multiblimp'),('multiblimp','hellaswag')]:
 pts=[t for t in tasks if mask.get(t,{}).get('family')==pf]
 tts=[t for t in tasks if mask.get(t,{}).get('family')==tf]
 for kind in ['size','checkpoint']:
  counters=collections.Counter()
  for pt in pts:
   for tt in tts:
    vals=[];informed=False
    levels=SIZES[:-1] if kind=='size' else list(range(10,101,10))
    for level in levels:
     if kind=='size':
      ans=da(scores(pt,level,100),scores(tt,'1.7B',100))
      informed |= ans is not None
      value=ans[0] if ans and not gated(pt,level) and not gated(tt,'1.7B') else None
     else:
      hits=0.;n=0
      for size in SIZES:
       ans=da(scores(pt,size,level),scores(tt,size,100))
       informed |= ans is not None
       if ans and not gated(pt,size) and not gated(tt,size):hits+=ans[0]*ans[1];n+=ans[1]
      value=hits/n if n>=3 else None
     vals.append((level,value))
    if not informed: counters['missing']+=1
    elif not any(v is not None for _,v in vals):counters['gated']+=1
    elif sustained(vals) is None:counters['never']+=1
    else:counters['reached']+=1
  out=dict(proxy_family=pf,target_family=tf,kind=kind,total=len(pts)*len(tts),**{k:counters[k] for k in ['reached','never','gated','missing']})
  cross.append(out);print('cross',out)
write('verified_cross_task_summary.csv',cross)
