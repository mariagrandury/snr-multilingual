import json,sys,re,collections
SC=sys.argv[1]
m=json.load(open(SC+"/models_main.json")); h=json.load(open(SC+"/models_head.json"))
for k in sorted(set(m['snr'])|set(h['snr'])):
    if m['snr'].get(k)!=h['snr'].get(k): print('snr.',k,':',json.dumps(m['snr'].get(k))[:300],'->',json.dumps(h['snr'].get(k))[:300])
for sec in ('pools','sources'):
    for k in sorted(set(m[sec])|set(h[sec])):
        if m[sec].get(k)!=h[sec].get(k): print(sec,k,json.dumps(m[sec].get(k))[:300],'->',json.dumps(h[sec].get(k))[:300])
rep=set(l.strip() for l in open(SC+'/cells_ladder_report.csv.txt'))
rem=sorted(set(m['models'])-set(h['models']))
print('report cells',len(rep))
print('removed-from-models.json cells present in published report:',[n for n in rem if n in rep])
print('report cells not in HEAD models.json:',sorted(rep-set(h['models'])))
size=lambda n: re.match(r'lm-([^-]+)',n).group(1)
print('report sizes',collections.Counter(size(n) for n in rep if n.startswith('lm-')))
n='lm-1B-L50-deep-seed28'; print(json.dumps(m['models'][n])[:700])
n='lm-90M-L8-deep-seed1904'; print(json.dumps(m['models'][n])[:500])
n='lm-90M-L8-b84-deep-seed1904'; print(json.dumps(h['models'][n])[:500])
