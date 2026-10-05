import sys, json
sys.path.insert(0, sys.argv[1]); out=sys.argv[2]
import launch_trainings as lt
res={}
for arch in ("deep","shallow"):
    data=json.loads(lt.HYPERPARAMS[arch].read_text()) if hasattr(lt.HYPERPARAMS[arch],'read_text') else json.load(open(lt.HYPERPARAMS[arch]))
    import inspect
    for c in (lt.predictivity_cells(arch=arch) if "arch" in inspect.signature(lt.predictivity_cells).parameters else lt.predictivity_cells()):
        if arch not in lt.arches_for(c["scheme"], c["size"], c["L"]): continue
        if c["size"] not in data["configs"]: continue
        cfg=data["configs"][c["size"]]
        gbs=lt.cell_gbs(c["size"]) if hasattr(lt,"cell_gbs") else lt.GBS
        if hasattr(lt,"scale_for_gbs"): cfg=lt.scale_for_gbs(cfg,gbs)
        exp=lt.exp_name(c["size"],c["L"],arch,c["seed"],c["scheme"])
        sub=lt.DATA_SCHEMES[c["scheme"]]["subdir"]
        blend=lt.data_blend("E/english_dclm", f"F/{sub}/fineweb_L{c['L']}", c["L"])
        env=lt.cell_env(cfg,c["size"],c["seed"],exp,blend,gbs=None if gbs==lt.GBS else gbs)
        env["GBS_eff"]=gbs
        env["langs"]=sorted(lt.cell_languages(c["L"],c["scheme"]))
        env["subsets"]=lt.cell_fineweb_subsets(c["L"],c["scheme"])
        env["tokens_B"]=round(env["TRAINING_STEPS"]*gbs*lt.SEQ_LEN/1e9,3)
        res[exp]=env
json.dump({"cells":res,"schemes":{k:{kk:(sorted(vv) if isinstance(vv,(set,frozenset)) else vv) for kk,vv in v.items()} for k,v in lt.DATA_SCHEMES.items()},"seeds":lt.SEED_TRIPLES,"ladder":lt.LADDER,"tok":lt.TOKENIZER_MODEL}, open(out,"w"), default=str, indent=0)
print(len(res))
