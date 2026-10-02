import sys, json, importlib.util
def load(p,name):
    s=importlib.util.spec_from_file_location(name,p+"/launch_trainings.py"); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
SC,WT=sys.argv[1],sys.argv[2]
M=load(SC+"/mainrepo/src/pretrain","ltm"); H=load(WT+"/src/pretrain","lth")
data=json.loads(H.HYPERPARAMS["deep"].read_text())["configs"]
for size in H.LADDER:
    t=M.schedule_for(data[size])[0]
    n=M.n_checkpoints(t); step=t//n; saved=[step*i for i in range(1,n+1)]
    old=M.due_iters(saved,t); new=H.due_iters(saved,t)
    fr=lambda L:[round(100*i/t,1) for i in L]
    print(size,'target',t,'saves',n,'old due',len(old),'new due',len(new),'new⊆old',set(new)<=set(old),'old-only %',fr(sorted(set(old)-set(new)))[:25],'new-only %',fr(sorted(set(new)-set(old))))
# aromanou 20-save 1B ending 45740
t=M.schedule_for(data["1B"])[0]; saved=[2287*i for i in range(1,21)]
old=M.due_iters(saved,t); new=H.due_iters(saved,t); print('1B aromanou 20-save: old',len(old),'new',len(new),'equal',old==new, [round(100*i/t,1) for i in new])
