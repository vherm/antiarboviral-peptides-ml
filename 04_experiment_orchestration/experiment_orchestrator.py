#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse, json, logging, math, sys, time, traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from openpyxl import load_workbook
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel, SelectKBest, VarianceThreshold, chi2, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, matthews_corrcoef, precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

MATRIX_SHEET = 'MEM_Master'

@dataclass
class RunConfig:
    matrix_path: str
    manifest_path: str
    output_dir: str
    n_splits: int = 5
    n_repeats: int = 5
    inner_splits: int = 3
    seed: int = 42
    search_mode: str = 'none'
    n_iter: int = 20
    n_jobs: int = -1
    save_models: bool = False
    resume: bool = False
    update_matrix_copy: bool = False


def norm(v: Any) -> str:
    return str(v).strip().casefold()


def parse_args():
    p = argparse.ArgumentParser(description='Orquestador de la Matriz Experimental Maestra')
    p.add_argument('--matrix', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--experiment-ids')
    p.add_argument('--phase')
    p.add_argument('--priority')
    p.add_argument('--status', default='Pendiente')
    p.add_argument('--negative-strategy')
    p.add_argument('--representation')
    p.add_argument('--model')
    p.add_argument('--feature-selection')
    p.add_argument('--max-experiments', type=int)
    p.add_argument('--n-splits', type=int, default=5)
    p.add_argument('--n-repeats', type=int, default=5)
    p.add_argument('--inner-splits', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--search-mode', choices=['none','grid','random'], default='none')
    p.add_argument('--n-iter', type=int, default=20)
    p.add_argument('--n-jobs', type=int, default=-1)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-models', action='store_true')
    p.add_argument('--update-matrix-copy', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    return p.parse_args()


def setup_logger(path: Path):
    logger = logging.getLogger(str(path)); logger.setLevel(logging.INFO); logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh = logging.FileHandler(path, encoding='utf-8'); fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)
    return logger


def load_matrix(path):
    df = pd.read_excel(path, sheet_name=MATRIX_SHEET)
    required = {'ID Experimento','Fase','Prioridad','Estrategia Negativos','Representación','Selección Variables','Modelo','Estado'}
    missing = required.difference(df.columns)
    if missing: raise ValueError(f'Faltan columnas en la matriz: {sorted(missing)}')
    return df


def load_manifest(path):
    df = pd.read_csv(path)
    required = {'negative_strategy','representation','input_file','label_column'}
    missing = required.difference(df.columns)
    if missing: raise ValueError(f'Faltan columnas en el manifiesto: {sorted(missing)}')
    return df


def filter_experiments(df, args):
    out = df.copy()
    if args.experiment_ids:
        ids = {x.strip() for x in args.experiment_ids.split(',') if x.strip()}
        out = out[out['ID Experimento'].astype(str).isin(ids)]
    for col, value in [('Fase',args.phase),('Prioridad',args.priority),('Estado',args.status),('Estrategia Negativos',args.negative_strategy),('Representación',args.representation),('Modelo',args.model),('Selección Variables',args.feature_selection)]:
        if value: out = out[out[col].astype(str).map(norm) == norm(value)]
    out = out.sort_values(['Prioridad','ID Experimento'])
    return out.head(args.max_experiments) if args.max_experiments else out


def resolve_dataset(exp, manifest):
    m = manifest[(manifest['negative_strategy'].map(norm)==norm(exp['Estrategia Negativos'])) & (manifest['representation'].map(norm)==norm(exp['Representación']))]
    if m.empty: raise FileNotFoundError(f"No existe manifiesto para {exp['Estrategia Negativos']} + {exp['Representación']}")
    if len(m)>1: raise ValueError('El manifiesto contiene entradas duplicadas para la combinación solicitada')
    return m.iloc[0]


def load_dataset(row):
    path = Path(str(row['input_file']))
    if not path.exists(): raise FileNotFoundError(path)
    df = pd.read_csv(path) if path.suffix.lower()=='.csv' else pd.read_excel(path, sheet_name=row.get('sheet_name',0) if pd.notna(row.get('sheet_name',0)) else 0)
    label = str(row['label_column'])
    ids = [x.strip() for x in str(row.get('id_columns','#|peptide_id|sequence|id')).split('|') if x.strip()]
    y = pd.to_numeric(df[label], errors='raise').astype(int)
    X = df.drop(columns=[label]+[c for c in ids if c in df.columns], errors='ignore')
    X = X.drop(columns=X.select_dtypes(include=['object','string']).columns, errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce')
    if X.shape[1]==0: raise ValueError('No quedaron características numéricas')
    if y.nunique()!=2: raise ValueError('Se esperaban exactamente dos clases')
    return X, y, X.columns.astype(str).tolist()


def create_selector(name, n_features, seed):
    k = max(1, min(500, n_features)); key = norm(name)
    if key in {'ninguna','none','c0'}: return 'passthrough'
    if 'variancethreshold' in key: return VarianceThreshold(0.0)
    if 'f_classif' in key or 'anova' in key: return SelectKBest(f_classif, k=k)
    if 'chi2' in key or 'chi²' in key: return SelectKBest(chi2, k=k)
    base = LogisticRegression(solver='liblinear', max_iter=2000, random_state=seed)
    if key=='rfe': return RFE(base, n_features_to_select=max(1,min(200,n_features//2)), step=0.1)
    if 'rfecv' in key: return RFECV(base, step=0.1, cv=StratifiedKFold(3,shuffle=True,random_state=seed), scoring='f1', n_jobs=1, min_features_to_select=max(1,min(20,n_features)))
    if 'sfm-l1' in key:
        return SelectFromModel(LogisticRegression(penalty='l1',solver='liblinear',C=0.1,max_iter=2000,random_state=seed), threshold='median')
    if 'sfm-rf' in key:
        return SelectFromModel(RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=seed,n_jobs=1), threshold='median')
    if 'sfm-xgb' in key:
        if XGBClassifier is None: raise ImportError('xgboost no está instalado')
        return SelectFromModel(XGBClassifier(n_estimators=200,max_depth=4,learning_rate=0.05,random_state=seed,n_jobs=1,eval_metric='logloss'), threshold='median')
    raise ValueError(f'Método FS no reconocido: {name}')


def create_model(name, seed):
    key = norm(name)
    if key=='svm': return SVC(C=1,kernel='rbf',gamma='scale',probability=True,class_weight='balanced',random_state=seed)
    if key in {'random forest','rf'}: return RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=seed,n_jobs=1)
    if key=='xgboost':
        if XGBClassifier is None: raise ImportError('xgboost no está instalado')
        return XGBClassifier(n_estimators=200,learning_rate=0.05,max_depth=4,subsample=0.9,colsample_bytree=0.9,random_state=seed,n_jobs=1,eval_metric='logloss')
    if key=='mlp': return MLPClassifier(hidden_layer_sizes=(100,50),alpha=1e-4,max_iter=800,early_stopping=True,random_state=seed)
    raise ValueError(f'Modelo no reconocido: {name}')


def param_space(name):
    key=norm(name)
    if key=='svm': return {'classifier__C':[0.1,1,10],'classifier__kernel':['linear','rbf'],'classifier__gamma':['scale','auto']}
    if key in {'random forest','rf'}: return {'classifier__n_estimators':[200,400],'classifier__max_depth':[None,10,20],'classifier__max_features':['sqrt','log2']}
    if key=='xgboost': return {'classifier__n_estimators':[100,200,400],'classifier__learning_rate':[0.01,0.05,0.1],'classifier__max_depth':[3,4,6]}
    if key=='mlp': return {'classifier__hidden_layer_sizes':[(100,),(100,50)],'classifier__alpha':[1e-4,1e-3],'classifier__solver':['adam','lbfgs']}
    return {}


def create_pipeline(exp, n_features, seed):
    fs = str(exp['Selección Variables']); model_name = str(exp['Modelo'])
    steps=[('imputer',SimpleImputer(strategy='median'))]
    if 'chi2' in norm(fs) or 'chi²' in norm(fs): steps.append(('scaler',MinMaxScaler()))
    elif norm(model_name) in {'svm','mlp'} or 'sfm-l1' in norm(fs): steps.append(('scaler',StandardScaler()))
    else: steps.append(('scaler','passthrough'))
    steps += [('feature_selection',create_selector(fs,n_features,seed)),('classifier',create_model(model_name,seed))]
    return Pipeline(steps)


def mcc_scorer(estimator, X, y): return matthews_corrcoef(y, estimator.predict(X))


def build_estimator(pipe, model_name, cfg, seed):
    if cfg.search_mode=='none': return pipe
    params=param_space(model_name)
    inner=StratifiedKFold(cfg.inner_splits,shuffle=True,random_state=seed)
    if cfg.search_mode=='grid': return GridSearchCV(pipe,params,scoring=mcc_scorer,cv=inner,n_jobs=cfg.n_jobs,refit=True)
    n=min(cfg.n_iter,int(np.prod([len(v) for v in params.values()])))
    return RandomizedSearchCV(pipe,params,n_iter=n,scoring=mcc_scorer,cv=inner,n_jobs=cfg.n_jobs,random_state=seed,refit=True)


def score_values(estimator,X):
    if hasattr(estimator,'predict_proba'): return estimator.predict_proba(X)[:,1]
    if hasattr(estimator,'decision_function'): return estimator.decision_function(X)
    return estimator.predict(X).astype(float)


def metrics(y_true,y_pred,y_score):
    tn,fp,fn,tp=confusion_matrix(y_true,y_pred,labels=[0,1]).ravel()
    out={'accuracy':accuracy_score(y_true,y_pred),'precision':precision_score(y_true,y_pred,zero_division=0),'sensitivity':tp/(tp+fn) if tp+fn else 0,'specificity':tn/(tn+fp) if tn+fp else 0,'f1':f1_score(y_true,y_pred,zero_division=0),'mcc':matthews_corrcoef(y_true,y_pred),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    try: out['roc_auc']=roc_auc_score(y_true,y_score)
    except ValueError: out['roc_auc']=float('nan')
    try: out['pr_auc']=average_precision_score(y_true,y_score)
    except ValueError: out['pr_auc']=float('nan')
    return out


def selected_features(estimator,names):
    pipe=estimator.best_estimator_ if hasattr(estimator,'best_estimator_') else estimator
    sel=pipe.named_steps.get('feature_selection')
    if sel=='passthrough' or sel is None: return names
    if hasattr(sel,'get_support'):
        mask=sel.get_support()
        if len(mask)==len(names): return [n for n,k in zip(names,mask) if k]
    return []


def summarize(df):
    out={'n_folds':int(len(df))}
    for m in ['accuracy','precision','sensitivity','specificity','f1','mcc','roc_auc','pr_auc','training_time_sec','inference_time_sec']:
        v=pd.to_numeric(df[m],errors='coerce').dropna()
        if len(v):
            mean=float(v.mean()); std=float(v.std(ddof=1)) if len(v)>1 else 0.0
            out[m]={'mean':mean,'std':std,'median':float(v.median()),'min':float(v.min()),'max':float(v.max()),'ci95_low':mean-1.96*std/math.sqrt(len(v)) if len(v)>1 else mean,'ci95_high':mean+1.96*std/math.sqrt(len(v)) if len(v)>1 else mean}
    return out


def run_experiment(exp, manifest_row, cfg):
    exp_id=str(exp['ID Experimento']); outdir=Path(cfg.output_dir)/exp_id; outdir.mkdir(parents=True,exist_ok=True)
    if cfg.resume and (outdir/'summary.json').exists(): return {'experiment_id':exp_id,'status':'skipped'}
    log=setup_logger(outdir/'run.log')
    (outdir/'config.json').write_text(json.dumps({'experiment':exp.to_dict(),'manifest':manifest_row.to_dict(),'run_config':asdict(cfg)},ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    X,y,names=load_dataset(manifest_row); log.info('X=%s y=%s',X.shape,y.shape)
    cv=RepeatedStratifiedKFold(cfg.n_splits,cfg.n_repeats,random_state=cfg.seed)
    folds=[]; preds=[]; params=[]; features=[]
    for idx,(tr,te) in enumerate(cv.split(X,y),1):
        rep=(idx-1)//cfg.n_splits+1; fold=(idx-1)%cfg.n_splits+1; seed=cfg.seed+idx
        est=build_estimator(create_pipeline(exp,X.shape[1],seed),str(exp['Modelo']),cfg,seed)
        t=time.perf_counter(); est.fit(X.iloc[tr],y.iloc[tr]); train_time=time.perf_counter()-t
        t=time.perf_counter(); yp=est.predict(X.iloc[te]); ys=score_values(est,X.iloc[te]); inf=time.perf_counter()-t
        rec=metrics(y.iloc[te].to_numpy(),yp,ys); rec.update({'experiment_id':exp_id,'repeat':rep,'fold':fold,'fold_index':idx,'seed':seed,'train_size':len(tr),'test_size':len(te),'training_time_sec':train_time,'inference_time_sec':inf}); folds.append(rec)
        preds.append(pd.DataFrame({'experiment_id':exp_id,'repeat':rep,'fold':fold,'row_index':te,'y_true':y.iloc[te].to_numpy(),'y_pred':yp,'y_score':ys}))
        params.append({'experiment_id':exp_id,'repeat':rep,'fold':fold,'best_params':json.dumps(getattr(est,'best_params_',{}),ensure_ascii=False)})
        for f in selected_features(est,names): features.append({'experiment_id':exp_id,'repeat':rep,'fold':fold,'feature':f})
        if cfg.save_models: joblib.dump(est,outdir/f'model_repeat{rep}_fold{fold}.joblib')
        log.info('Fold %s MCC=%.4f F1=%.4f',idx,rec['mcc'],rec['f1'])
    fdf=pd.DataFrame(folds); fdf.to_csv(outdir/'fold_metrics.csv',index=False)
    pd.concat(preds,ignore_index=True).to_csv(outdir/'predictions.csv',index=False)
    pd.DataFrame(params).to_csv(outdir/'best_params_by_fold.csv',index=False)
    if features:
        sdf=pd.DataFrame(features); sdf.to_csv(outdir/'selected_features_by_fold.csv',index=False)
        sdf.groupby('feature').size().sort_values(ascending=False).rename('fold_frequency').reset_index().to_csv(outdir/'selected_feature_frequency.csv',index=False)
    summary=summarize(fdf); summary.update({'experiment_id':exp_id,'status':'completed','negative_strategy':str(exp['Estrategia Negativos']),'representation':str(exp['Representación']),'feature_selection':str(exp['Selección Variables']),'model':str(exp['Modelo']),'dataset':str(manifest_row['input_file']),'n_samples':len(X),'n_features':X.shape[1],'finished_at':datetime.now().isoformat()})
    (outdir/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    return summary


def update_matrix(matrix_path, output_dir, exp_id, status, progress):
    src=Path(matrix_path); target=Path(output_dir)/(src.stem+'_actualizada.xlsx')
    if not target.exists(): target.write_bytes(src.read_bytes())
    wb=load_workbook(target); ws=wb[MATRIX_SHEET]; headers={c.value:i for i,c in enumerate(ws[1],1)}
    for r in range(2,ws.max_row+1):
        if str(ws.cell(r,headers['ID Experimento']).value)==exp_id:
            ws.cell(r,headers['Estado']).value=status; ws.cell(r,headers['% Avance']).value=progress; break
    wb.save(target)


def global_summary(output):
    rows=[]
    for p in Path(output).glob('EXP-*/summary.json'):
        try: s=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        row={k:s.get(k) for k in ['experiment_id','status','negative_strategy','representation','feature_selection','model','dataset','n_samples','n_features']}
        for m in ['mcc','f1','accuracy','sensitivity','specificity','roc_auc','pr_auc']:
            row[m+'_mean']=s.get(m,{}).get('mean'); row[m+'_std']=s.get(m,{}).get('std')
        rows.append(row)
    if rows: pd.DataFrame(rows).sort_values(['mcc_mean','pr_auc_mean'],ascending=False).to_csv(Path(output)/'global_summary.csv',index=False)


def main():
    a=parse_args(); cfg=RunConfig(a.matrix,a.manifest,a.output,a.n_splits,a.n_repeats,a.inner_splits,a.seed,a.search_mode,a.n_iter,a.n_jobs,a.save_models,a.resume,a.update_matrix_copy)
    Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
    matrix=load_matrix(cfg.matrix_path); manifest=load_manifest(cfg.manifest_path); selected=filter_experiments(matrix,a)
    if selected.empty: print('No se encontraron experimentos'); return 1
    print(selected[['ID Experimento','Estrategia Negativos','Representación','Selección Variables','Modelo']].to_string(index=False))
    if a.dry_run: return 0
    failures=[]
    for _,exp in selected.iterrows():
        exp_id=str(exp['ID Experimento'])
        try:
            if cfg.update_matrix_copy: update_matrix(cfg.matrix_path,cfg.output_dir,exp_id,'En ejecución',0.1)
            result=run_experiment(exp,resolve_dataset(exp,manifest),cfg)
            if cfg.update_matrix_copy and result.get('status')=='completed': update_matrix(cfg.matrix_path,cfg.output_dir,exp_id,'Completado',1.0)
        except Exception as e:
            tb=traceback.format_exc(); failures.append({'experiment_id':exp_id,'error':str(e),'traceback':tb})
            d=Path(cfg.output_dir)/exp_id; d.mkdir(parents=True,exist_ok=True); (d/'error.txt').write_text(tb,encoding='utf-8')
            if cfg.update_matrix_copy: update_matrix(cfg.matrix_path,cfg.output_dir,exp_id,'Bloqueado',0.0)
            print(f'[ERROR] {exp_id}: {e}',file=sys.stderr)
    if failures: pd.DataFrame(failures).to_csv(Path(cfg.output_dir)/'failed_experiments.csv',index=False)
    global_summary(cfg.output_dir); print(f'Ejecución finalizada. Fallos: {len(failures)}'); return 0 if not failures else 2

if __name__=='__main__': raise SystemExit(main())
