export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
git log -1 --format=%B 49e24dad | head -60
prev=origin/main
for c in $(git log --reverse --format=%h origin/main..HEAD -- configs/models.json); do
  git show ${prev}:configs/models.json > $1/x.json; git show ${c}:configs/models.json > $1/y.json
  venv/bin/python - $1 "$c" <<'PY'
import json,sys,re,collections
SC,c=sys.argv[1],sys.argv[2]
x=json.load(open(SC+"/x.json")); y=json.load(open(SC+"/y.json"))
xm,ym=x["models"],y["models"]
rem=sorted(set(xm)-set(ym)); add=sorted(set(ym)-set(xm)); chg=[n for n in set(xm)&set(ym) if xm[n]!=ym[n]]
print("==",c,"removed",len(rem),"added",len(add),"changed",len(chg), "snr changed:", {k:(x["snr"].get(k),y["snr"].get(k)) for k in set(x["snr"])|set(y["snr"]) if x["snr"].get(k)!=y["snr"].get(k)}, "pools/sources changed:", x["pools"]!=y["pools"], x["sources"]!=y["sources"])
print("   removed:", collections.Counter(re.sub(r"-seed\d+$","",re.sub(r"-L\d+","-L*",n)) for n in rem))
print("   added:", collections.Counter(re.sub(r"-seed\d+$","",re.sub(r"-L\d+","-L*",n)) for n in add))
for n in chg[:3]: print("   changed", n, [(k) for k in ym[n] if xm[n].get(k)!=ym[n].get(k)])
PY
  prev=$c
done
