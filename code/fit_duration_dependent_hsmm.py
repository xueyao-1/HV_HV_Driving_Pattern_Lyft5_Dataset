"""Fit an observed-state duration-dependent semi-Markov (HSMM-style) model.

The Action labels are observed, not latent. Each completed spell has an
explicit duration, and the destination distribution is conditioned on the
current Action and its completed duration.

Example
-------
python code/fit_duration_dependent_hsmm.py --segments segments_all_vehicles.csv --output results/hsmm
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

PATTERNS=["Follow_behind","Slow_down","Catch_up","Speed_up","Fall_behind","Hold_speed"]
IDX={p:i for i,p in enumerate(PATTERNS)}; K=6; DT=.1; SEED=20260926

def episodes(path):
    raw=pd.read_csv(path).sort_values(["VehicleID","segment_index"]); rows=[]
    for driver,g in raw.groupby("VehicleID",sort=False):
        states=[]; durations=[]
        for r in g.itertuples():
            if r.pattern_label not in IDX:continue
            d=float(r.duration)*DT
            if states and IDX[r.pattern_label]==states[-1]:durations[-1]+=d
            else:states.append(IDX[r.pattern_label]);durations.append(d)
        for n in range(len(states)-1):rows.append((str(driver),states[n],states[n+1],durations[n]))
    return pd.DataFrame(rows,columns=["driver","current","next","duration"])

def fit(train,C):
    models={}; payload={"patterns":PATTERNS,"frame_dt":DT,"origins":{}}
    for j,name in enumerate(PATTERNS):
        d=train[train.current==j]; z=np.log1p(d.duration.to_numpy()); mu=float(z.mean()); sd=max(float(z.std()),1e-9)
        model=LogisticRegression(C=C,solver="lbfgs",max_iter=3000).fit(((z-mu)/sd).reshape(-1,1),d.next)
        models[j]=(model,mu,sd)
        payload["origins"][name]={"duration_log_mean":mu,"duration_log_sd":sd,
          "destination_indices":model.classes_.astype(int).tolist(),"intercept":model.intercept_.astype(float).tolist(),
          "duration_coefficient":model.coef_[:,0].astype(float).tolist(),"empirical_durations_s":d.duration.astype(float).tolist()}
    return models,payload

def predict(data,models):
    out=np.zeros((len(data),K))
    for q,r in enumerate(data.itertuples()):
        model,mu,sd=models[r.current]; out[q,model.classes_.astype(int)]=model.predict_proba([[(np.log1p(r.duration)-mu)/sd]])[0]
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--segments",type=Path,required=True);ap.add_argument("--output",type=Path,default=Path("results/hsmm"));ap.add_argument("--C",type=float,default=1.)
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True);data=episodes(a.segments);ids=data.driver.unique();rng=np.random.default_rng(SEED);rng.shuffle(ids);cut=int(.8*len(ids));train=data[data.driver.isin(ids[:cut])];test=data[data.driver.isin(ids[cut:])]
    models,payload=fit(train,a.C);prob=predict(test,models);y=test.next.to_numpy();metrics={"n":len(y),"accuracy":float(np.mean(prob.argmax(1)==y)),"nll":float(log_loss(y,prob,labels=np.arange(K))),"brier":float(np.mean(np.sum((prob-np.eye(K)[y])**2,axis=1)))}
    payload["fit"]={"C":a.C,"train_drivers":int(train.driver.nunique()),"test_drivers":int(test.driver.nunique()),"test_metrics":metrics}
    (a.output/"duration_dependent_hsmm.json").write_text(json.dumps(payload,indent=2));print(json.dumps(payload["fit"],indent=2))
if __name__=="__main__":main()
