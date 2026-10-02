import json,sys
SC=sys.argv[1]
def flat(d,p=''):
    o={}
    if isinstance(d,dict):
        for k,v in d.items(): o.update(flat(v,p+'/'+str(k)))
    elif isinstance(d,list):
        for i,v in enumerate(d): o.update(flat(v,p+'/'+str(i)))
    else: o[p]=d
    return o
for a in ('deep','shallow'):
    m=flat(json.load(open(f'{SC}/hp_{a}_main.json'))); h=flat(json.load(open(f'{SC}/hp_{a}_head.json')))
    print(a, 'keys main',len(m),'head',len(h))
    for k in sorted(set(m)|set(h)):
        if m.get(k,'<absent>')!=h.get(k,'<absent>'): print('  ',k,m.get(k,'<absent>'),'->',h.get(k,'<absent>'))
