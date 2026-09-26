"""Fit and evaluate first- and second-order Markov chains for Action patterns.

Example
-------
python code/fit_markov_chain.py --segments segments_all_vehicles.csv --output results/markov
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import GroupShuffleSplit

PATTERNS=["Follow_behind","Slow_down","Catch_up","Speed_up","Fall_behind","Hold_speed"]
IDX={p:i for i,p in enumerate(PATTERNS)}; K=len(PATTERNS); SEED=20260926

def sequences(path):
    d=pd.read_csv(path).sort_values(["VehicleID","segment_index"]); out=[]
    for driver,g in d.groupby("VehicleID",sort=False):
        states=[]
        for label in g.pattern_label:
            if label in IDX and (not states or IDX[label]!=states[-1]): states.append(IDX[label])
        if len(states)>1: out.append((str(driver),states))
    return out

def rows(seq,second=False):
    out=[]
    for driver,s in seq:
        for n in range(1 if second else 0,len(s)-1):
            out.append((driver,s[n-1] if n else -1,s[n],s[n+1]))
    return pd.DataFrame(out,columns=["driver","previous","current","next"])

def fit(train,alpha):
    c1=np.zeros((K,K)); c2=np.zeros((K,K,K))
    for r in train.itertuples():
        c1[r.current,r.next]+=1
        if r.previous>=0:c2[r.previous,r.current,r.next]+=1
    p1=np.zeros_like(c1); p2=np.zeros_like(c2)
    for j in range(K):
        allowed=np.arange(K)!=j; p1[j,allowed]=(c1[j,allowed]+alpha)/(c1[j,allowed].sum()+alpha*allowed.sum())
        for i in range(K):p2[i,j,allowed]=(c2[i,j,allowed]+alpha)/(c2[i,j,allowed].sum()+alpha*allowed.sum())
    return p1,p2

def metrics(y,p):
    return {"n":len(y),"accuracy":float(np.mean(p.argmax(1)==y)),
            "nll":float(log_loss(y,p,labels=np.arange(K))),
            "brier":float(np.mean(np.sum((p-np.eye(K)[y])**2,axis=1)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--segments",type=Path,required=True)
    ap.add_argument("--output",type=Path,default=Path("results/markov")); ap.add_argument("--alpha",type=float,default=1.)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True); seq=sequences(a.segments)
    ids=np.array([d for d,_ in seq]); splitter=GroupShuffleSplit(1,test_size=.2,random_state=SEED)
    tr,te=next(splitter.split(ids,groups=ids)); train_ids=set(ids[tr]); test_ids=set(ids[te])
    train=rows([x for x in seq if x[0] in train_ids],second=True); test=rows([x for x in seq if x[0] in test_ids],second=True)
    p1,p2=fit(train,a.alpha); y=test.next.to_numpy(); q1=np.vstack([p1[r.current] for r in test.itertuples()]); q2=np.vstack([p2[r.previous,r.current] for r in test.itertuples()])
    result={"patterns":PATTERNS,"alpha":a.alpha,"train_drivers":len(train_ids),"test_drivers":len(test_ids),
            "first_order":metrics(y,q1),"second_order":metrics(y,q2)}
    (a.output/"metrics.json").write_text(json.dumps(result,indent=2)); np.savez(a.output/"transition_matrices.npz",first_order=p1,second_order=p2)
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
