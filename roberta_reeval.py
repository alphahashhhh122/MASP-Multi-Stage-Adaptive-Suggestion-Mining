#!/usr/bin/env python3
import pandas as pd, numpy as np, json, os
print("="*70); print("ROBERTA RE-EVALUATION"); print("="*70)
for d in ["data","../data"]:
    if os.path.exists(f"{d}/train.csv") and os.path.exists(f"{d}/test.csv"):
        data_dir=d; break
else: print("ERROR: no data"); exit(1)
train=pd.read_csv(f"{data_dir}/train.csv"); test=pd.read_csv(f"{data_dir}/test.csv")
print(f"Train: {len(train)}, Test: {len(test)}")
print(f"Train cols: {list(train.columns)}")
text_col=None
for c in ["raw_text","text","review_text"]:
    if c in train.columns: text_col=c; break
if not text_col: print("ERROR: no text col"); exit(1)
print(f"Text col: {text_col}")
def to_bin(v):
    s=str(v).strip().lower(); return 1 if s in ("true","1","1.0","yes") else 0
train["label"]=train["is_suggestion"].apply(to_bin)
test["label"]=test["is_suggestion"].apply(to_bin)
print(f"Train labels: {dict(train['label'].value_counts())}")
print(f"Test labels: {dict(test['label'].value_counts())}")
if "extraction_path" in test.columns:
    print("\nTest per-path:")
    for p in sorted(test["extraction_path"].unique()):
        s=test[test["extraction_path"]==p]; print(f"  {p:5s}: n={len(s):3d} pos={(s['label']==1).sum()} neg={(s['label']==0).sum()}")
signal=["should","could","would","recommend","suggest","improve","better","fix","broken","damaged","terrible","awful","horrible","worst","poor","disappointing","unacceptable","need","must"]
print("\nPolite masking text check:")
for path in ["P5","P7","P8"]:
    if "extraction_path" not in test.columns: break
    sub=test[test["extraction_path"]==path]; v=0
    for _,row in sub.iterrows():
        words=str(row[text_col]).lower().split(); found=[w for w in signal if w in words]
        if found: v+=1; print(f"  VIOLATION {path}: {found} in '{str(row[text_col])[:60]}'")
    print(f"  {path}: {len(sub)} samples, {v} violations")
if "entry_id" in train.columns and "entry_id" in test.columns:
    ov=set(train["entry_id"])&set(test["entry_id"]); print(f"\nTrain-test overlap: {len(ov)}")
print("\nText length by path:")
if "extraction_path" in test.columns:
    for p in sorted(test["extraction_path"].unique()):
        s=test[test["extraction_path"]==p]; l=s[text_col].str.len()
        print(f"  {p:5s}: mean={l.mean():.1f} std={l.std():.1f}")
print(f"\n{'='*70}\nTRAINING (3 seeds)\n{'='*70}")
from transformers import AutoTokenizer,AutoModelForSequenceClassification,TrainingArguments,Trainer
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support,matthews_corrcoef
tokenizer=AutoTokenizer.from_pretrained("roberta-base")
def tok(ex): return tokenizer(ex["text"],truncation=True,padding="max_length",max_length=128)
train_ds=Dataset.from_dict({"text":train[text_col].fillna("").astype(str).tolist(),"label":train["label"].tolist()}).map(tok,batched=True)
test_ds=Dataset.from_dict({"text":test[text_col].fillna("").astype(str).tolist(),"label":test["label"].tolist()}).map(tok,batched=True)
all_preds=[]
for seed in [42,123,456]:
    print(f"\n--- Seed {seed} ---")
    model=AutoModelForSequenceClassification.from_pretrained("roberta-base",num_labels=2)
    args=TrainingArguments(output_dir=f"/tmp/rb_s{seed}",num_train_epochs=5,per_device_train_batch_size=16,per_device_eval_batch_size=32,learning_rate=2e-5,weight_decay=0.01,seed=seed,logging_steps=100,save_strategy="no",report_to="none")
    Trainer(model=model,args=args,train_dataset=train_ds).train()
    preds=Trainer(model=model,args=args).predict(test_ds)
    y_pred=np.argmax(preds.predictions,axis=1); y_true=np.array(test["label"].tolist())
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average="binary")
    mcc=matthews_corrcoef(y_true,y_pred)
    tp=int(((y_pred==1)&(y_true==1)).sum()); fp=int(((y_pred==1)&(y_true==0)).sum())
    fn=int(((y_pred==0)&(y_true==1)).sum()); tn=int(((y_pred==0)&(y_true==0)).sum())
    print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f} MCC={mcc:.3f} TP={tp} FP={fp} FN={fn} TN={tn}")
    all_preds.append(y_pred)
    if "extraction_path" in test.columns:
        tc=test.copy(); tc["pred"]=y_pred
        for path in sorted(tc["extraction_path"].unique()):
            s=tc[tc["extraction_path"]==path]
            t2=int(((s["pred"]==1)&(s["label"]==1)).sum()); f2=int(((s["pred"]==0)&(s["label"]==1)).sum()); fp2=int(((s["pred"]==1)&(s["label"]==0)).sum())
            rc=t2/(t2+f2) if t2+f2 else 0
            print(f"    {path:5s}: TP={t2:3d} FN={f2:3d} FP={fp2:3d} R={rc:.3f} (n={len(s)})")
print(f"\n{'='*70}\nMAJORITY VOTE\n{'='*70}")
maj=(np.array(all_preds).sum(axis=0)>=2).astype(int); y_true=np.array(test["label"].tolist())
p,r,f1,_=precision_recall_fscore_support(y_true,maj,average="binary")
print(f"  P={p:.3f} R={r:.3f} F1={f1:.3f}")
if "extraction_path" in test.columns:
    tc=test.copy(); tc["pred"]=maj
    for path in sorted(tc["extraction_path"].unique()):
        s=tc[tc["extraction_path"]==path]
        t2=int(((s["pred"]==1)&(s["label"]==1)).sum()); f2=int(((s["pred"]==0)&(s["label"]==1)).sum()); fp2=int(((s["pred"]==1)&(s["label"]==0)).sum())
        rc=t2/(t2+f2) if t2+f2 else 0; tag="ARTIFACT" if rc>0.8 and path in ["P5","P7","P8"] else ""
        print(f"    {path:5s}: TP={t2:3d} FN={f2:3d} FP={fp2:3d} R={rc:.3f} {tag}")
print(f"\nMASP F1=0.946 | RoBERTa F1={f1:.3f}")
print(f"MASP MF1=0.751 | RoBERTa MF1=--- (no extraction)")
print(f"\n{'='*70}\nDONE - save the entire output for the submission record\n{'='*70}")
