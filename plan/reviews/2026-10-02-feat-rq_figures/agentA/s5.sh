export PATH=/opt/homebrew/bin:$PATH; cd /Users/mariagrandury/Projects/epfl/snr-multilingual
for a in deep shallow; do
  git show origin/main:src/pretrain/hyperparams/hyperparams_$a.json > $1/hm.json; git show HEAD:src/pretrain/hyperparams/hyperparams_$a.json > $1/hh.json
  venv/bin/python - $1 $a <<'PY'
import json,sys
m=json.load(open(sys.argv[1]+"/hm.json")); h=json.load(open(sys.argv[1]+"/hh.json"))
print(sys.argv[2],"configs identical:", m["configs"]==h["configs"], "global diff keys:", [k for k in h["global"] if m["global"].get(k)!=h["global"][k]] if "global" in h else [k for k in h if k!="configs" and m.get(k)!=h.get(k)])
print("  widths:", {s:(c.get("hidden_size") or c.get("d_model")) for s,c in h["configs"].items()})
PY
done
git diff origin/main HEAD -- src/pretrain/data/language_sets_scheme*.json | head -5; echo "lang sets diff lines: $(git diff origin/main HEAD -- src/pretrain/data/ configs/languages.json | grep -c '^[+-]')"; git diff --stat origin/main HEAD -- src/pretrain/data configs/languages.json configs/hf_wandb.json | cat
