import pandas as pd, glob, os, re, subprocess, sys, numpy as np, warnings
warnings.filterwarnings('ignore')
A='src/signal-and-noise/analysis'; R='/Users/mariagrandury/Projects/epfl/snr-multilingual'
mask=pd.read_csv(f'{A}/rq00_gate_and_curves/pretraining/predictivity/above_random_mask.csv').set_index('task')
SZ=['90M','175M','350M','600M','1B','1.7B']
rows=[]
for f in sorted(glob.glob(f'{A}/rq*/pretraining/predictivity*/*.csv')):
    rq=f.split('/')[3][:4]
    if rq in('rq00','rq01'): continue
    try: h=pd.read_csv(f,nrows=0).columns
    except Exception as e: continue
    if 'task' not in h: continue
    df=pd.read_csv(f,low_memory=False)
    if df.empty: rows.append((f,'EMPTY',0,0,0,0,'')); continue
    nrf=df.task.astype(str).str.startswith('rfgm_belebele').sum()
    num=[c for c in df.columns if df[c].dtype.kind=='f']
    viol=0; cells=0; kind=''
    if 'size' in df.columns:
        kind='long'
        df['size']=df['size'].astype(str)
        d=df[df['size'].isin(SZ)&df.task.isin(mask.index)]
        mv=np.array([mask.at[t,s] for t,s in zip(d.task,d['size'])],dtype=float)
        valcols=[c for c in num if c not in('frac','n_pairs','n_matching','L','seed','n_items','n_options','random_baseline','chance','n','n_runs','n_seeds','compute','tokens','step')]
        has=d[valcols].notna().any(axis=1).to_numpy() if valcols else np.zeros(len(d),bool)
        cells=int((mv==0).sum()); viol=int(((mv==0)&has).sum())
        vc=','.join(valcols[:4])
    else:
        kind='wide'
        d=df[df.task.isin(mask.index)]
        vc=''
        for s in SZ:
            cs=[c for c in num if re.search(r'(^|_)'+re.escape(s)+r'($|_)',c)]
            if not cs: continue
            mv=mask[s].reindex(d.task).to_numpy(dtype=float)
            has=d[cs].notna().any(axis=1).to_numpy()
            cells+=int((mv==0).sum()); viol+=int(((mv==0)&has).sum())
            if viol and not vc: 
                bad=[c for c in cs if d.loc[(mv==0),c].notna().any()]; vc=','.join(bad[:4])
    date=subprocess.run(['git','-C',R,'log','-1','--format=%h %ad','--date=short','--',f],capture_output=True,text=True).stdout.strip()
    rows.append((f.replace(A+'/',''),kind,len(df),nrf,cells,viol,date+' '+vc))
for r in rows: print('\t'.join(map(str,r)))
