from pathlib import Path
import sys, json, hashlib, re
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'deps'))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

OUT=ROOT/'analysis'; OUT.mkdir(exist_ok=True)
FIG=ROOT/'latex'/'figures'; FIG.mkdir(parents=True,exist_ok=True)
s=yaml.safe_load((ROOT/'data/settings.yml').read_text(encoding='utf-8'))['input']
file=ROOT/'data/Tesla.xlsx'
tr=pd.read_excel(file,sheet_name=s['train-sheet']).dropna(subset=[s['target']]).reset_index(drop=True)
te=pd.read_excel(file,sheet_name=s['test-sheet']).dropna(subset=[s['target']]).reset_index(drop=True)
NUM=s['features']['numerical']; CAT=s['features']['categorical']; Y=s['target']
YEAR,MILE,POWER,EQUIP=NUM; MODEL,COUNTRY,AP=CAT; TITLE='款式／刊登標題'
PAT={'Performance':r'performance|plaid|ludicrous|p100d|p90d|p85d','Long Range':r'long range|maximale reichweite|grande autonomie|gran autonom|100d|lr ',
 'AWD':r'awd|dual motor|dual-motor|allrad|4wd|4x4|intégrale|integrale','Entry trim':r'rwd|standard range|sr\+|sr\b|propulsion|hinterrad',
 'New generation':r'highland|juniper|plaid|facelift|refresh','HW4':r'hw4|hw 4|hardware 4|ai4'}
FLAGS=list(PAT); INTER=['Model S x year','Model X x year','Model Y x year']
EN=dict(zip(NUM,['Year','Mileage','Power','Equipment']))|dict(zip(CAT,['Model','Country','Autopilot']))|{'荷蘭':'NL','德國':'DE','義大利':'IT','西班牙':'ES','比利時':'BE','奧地利時':'AT','奧地利':'AT','法國':'FR','盧森堡':'LU'}
assert len(tr)==3002 and len(te)==752
assert not tr[NUM+CAT+[Y]].isna().any().any()
y=tr[Y].to_numpy(float); yt=te[Y].to_numpy(float)
def groups(df):
 return df['車商'].fillna(pd.Series(['_row'+str(i) for i in range(len(df))],index=df.index)).astype(str).to_numpy()
g=groups(tr); gt=groups(te)
folds=list(KFold(5,shuffle=True,random_state=42).split(tr)); gf=list(GroupKFold(5).split(tr,y,g))
def features(df,mean):
 x=df.copy()
 for k,p in PAT.items(): x[k]=x[TITLE].fillna('').str.lower().str.contains(p).astype(float)
 for k in INTER: x[k]=(x[MODEL]==k.split(' x ')[0]).astype(float)*(x[YEAR]-mean)
 return x
def rank(x,t):
 scores={c:np.corrcoef(x[c],t)[0,1]**2 for c in NUM}
 ts=pd.Series(t,index=x.index); total=((ts-ts.mean())**2).sum()
 for c in CAT:
  means=ts.groupby(x[c]).transform('mean'); scores[c]=float(((means-ts.mean())**2).sum()/total)
 return sorted(scores,key=scores.get,reverse=True),scores
def fitpredict(train,target,test,kind='C',k=5,alpha=0):
 mean=train[YEAR].mean(); x=features(train,mean); v=features(test,mean)
 nums=NUM.copy(); cats=CAT.copy()
 if kind in ['C','no_flags','no_inter','ridge']:
  nums+= ([] if kind=='no_flags' else FLAGS)+([] if kind=='no_inter' else INTER)
 if kind=='B':
  chosen,_=rank(train,target); chosen=chosen[:k]; nums=[c for c in NUM if c in chosen]; cats=[c for c in CAT if c in chosen]
 if kind=='simple': nums=[MILE]; cats=[]
 if kind=='mean':
  m=DummyRegressor().fit(x[NUM],target); return m.predict(v[NUM]),m
 pre=ColumnTransformer([('n','passthrough' if kind=='RF' else StandardScaler(),nums),('c',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cats)])
 model=RandomForestRegressor(n_estimators=200,random_state=42,n_jobs=4) if kind=='RF' else (Ridge(alpha=alpha) if kind=='ridge' else LinearRegression())
 m=Pipeline([('pre',pre),('model',model)]).fit(x,target)
 return m.predict(v),m
def cv(kind,fs,k=5,alpha=0):
 res=[]
 for a,b in fs:
  p,_=fitpredict(tr.iloc[a],y[a],tr.iloc[b],kind,k,alpha)
  res.append([mean_absolute_error(y[b],p),np.sqrt(mean_squared_error(y[b],p)),r2_score(y[b],p)])
 return np.array(res)
ks=[]
for k in range(1,8):
 a=cv('B',folds,k); ks.append([k,a[:,0].mean(),a[:,0].std(ddof=1)])
kd=pd.DataFrame(ks,columns=['k','MAE','SD']); best=kd.loc[kd.MAE.idxmin()]; kval=int(kd[kd.MAE<=best.MAE+best.SD/np.sqrt(5)].k.min()); kd.to_csv(OUT/'k_search.csv',index=False)
print('Selected k',kval,flush=True)
res=[]; preds={}; cvs={}; gcvs={}
for kind in ['mean','simple','RF','A','B','C','no_flags','no_inter']:
 a=cv(kind,folds,kval); b=cv(kind,gf,kval); p,m=fitpredict(tr,y,te,kind,kval)
 preds[kind]=p; cvs[kind]=a; gcvs[kind]=b
 row=dict(model=kind,cv_mae=a[:,0].mean(),cv_sd=a[:,0].std(ddof=1),cv_rmse=a[:,1].mean(),cv_rmse_sd=a[:,1].std(ddof=1),group_mae=b[:,0].mean(),group_sd=b[:,0].std(ddof=1),test_mae=mean_absolute_error(yt,p),test_rmse=np.sqrt(mean_squared_error(yt,p)),test_r2=r2_score(yt,p),within3000=float(np.mean(abs(yt-p)<=3000)))
 res.append(row); print(row,flush=True)
pd.DataFrame(res).to_csv(OUT/'results.csv',index=False)
pd.DataFrame({'actual':yt,**preds}).to_csv(OUT/'predictions.csv',index=False)
# A training-only regularization sensitivity; fixed C remains the final descriptive model.
tuning=[]
for alpha in [.01,.1,1,10,100,1000]:
 a=cv('ridge',gf,alpha=alpha); tuning.append([alpha,a[:,0].mean(),a[:,0].std(ddof=1)])
pd.DataFrame(tuning,columns=['alpha','group_mae','group_sd']).to_csv(OUT/'ridge_sensitivity.csv',index=False)
# Cluster bootstrap: paired loss gap, preserving dealer groups in held-out data.
rng=np.random.default_rng(42); _,idx=np.unique(gt,return_inverse=True); counts=np.bincount(idx); ngrp=len(counts)
tests=[]
for baseline in ['A','B','RF','no_flags','no_inter']:
 gap=abs(yt-preds[baseline])-abs(yt-preds['C']); sums=np.bincount(idx,weights=gap); boot=[]
 for _ in range(10000):
  draw=rng.integers(0,ngrp,ngrp); boot.append(sums[draw].sum()/counts[draw].sum())
 tests.append(dict(comparison=baseline+' minus C',gap=gap.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975),paired_group_p=stats.ttest_rel(gcvs[baseline][:,0],gcvs['C'][:,0]).pvalue))
pd.DataFrame(tests).to_csv(OUT/'comparisons.csv',index=False)
# Full-rank final-model coefficient parametrization; use actual training mean, common references.
x=features(tr,tr[YEAR].mean()); nums=NUM+FLAGS+INTER
scale=x[nums].std(ddof=0); z=(x[nums]-x[nums].mean())/scale
z.columns=[EN.get(c,c) for c in nums]
refs={}; cat_counts={}
for col in CAT:
 vals=tr[col].astype(str); ref=vals.mode()[0]; refs[EN[col]]=ref
 for val in sorted(vals.unique()):
  if val==ref: continue
  name=EN[col]+': '+EN.get(val,val); z[name]=(vals==val).astype(float); cat_counts[name]=int((vals==val).sum())
D=sm.add_constant(z).astype(float); ols=sm.OLS(y,D).fit()
# A singleton category has unit leverage: HC3 inference is not identifiable for its coefficient.
single=[n for n,c in cat_counts.items() if c==1]; eligible=~(D[single].sum(axis=1)>0) if single else np.ones(len(D),bool)
Dr=D.loc[eligible].drop(columns=single); robust=sm.OLS(y[eligible],Dr).fit(cov_type='HC3',use_t=True)
ci=robust.conf_int(); coef=[]
for i,name in enumerate(D.columns):
 if name=='const': continue
 identifiable=name not in single
 coef.append(dict(feature=name,beta=ols.params[name],low=ci.loc[name,0] if identifiable else np.nan,high=ci.loc[name,1] if identifiable else np.nan,p=robust.pvalues[name] if identifiable else np.nan,vif=variance_inflation_factor(D.to_numpy(),i),n=cat_counts.get(name,len(tr))))
pd.DataFrame(coef).to_csv(OUT/'coefficients.csv',index=False)
p,_=fitpredict(tr,y,te,'C'); pt,_=fitpredict(tr,y,tr,'C'); assert np.max(abs(ols.fittedvalues-pt))<1e-4
# Test cases always refer to C.
err=yt-p; worst=np.argsort(abs(err))[::-1][:5]
cases=te.loc[worst,[MODEL,TITLE,YEAR,MILE,POWER,COUNTRY,Y]].copy(); cases['pred']=p[worst];cases['error']=err[worst];cases['test_row']=worst+1;cases.to_csv(OUT/'largest_errors.csv',index=False)
slice_rows=[]
for val in te[MODEL].unique():
 mask=(te[MODEL]==val).to_numpy(); slice_rows.append(dict(model=val,n=int(mask.sum()),**{k:float(abs(yt[mask]-v[mask]).mean()) for k,v in preds.items()}))
pd.DataFrame(slice_rows).to_csv(OUT/'slices.csv',index=False)
_,ranking=rank(tr,y)
summary={'train':len(tr),'test':len(te),'hash':hashlib.sha256(file.read_bytes()).hexdigest(),'columns':list(tr.columns),'references':refs,'ranking':{EN.get(k,k):v for k,v in ranking.items()},'correlations':{EN[c]:{'pearson':float(stats.pearsonr(tr[c],y).statistic),'spearman':float(stats.spearmanr(tr[c],y).statistic)} for c in NUM},'numeric_summary':tr[NUM+[Y]].describe().to_dict(),'year_mileage_r':float(tr[[YEAR,MILE]].corr().iloc[0,1]),'dealer_groups_train':len(np.unique(g)),'dealer_groups_test':ngrp,'negative_C':int((p<0).sum()),'singleton':single,'rank':int(np.linalg.matrix_rank(D)),'p':D.shape[1],'selected_k':kval,'ridge':tuning,'coefficient_scales':{EN.get(k,k):float(v) for k,v in scale.items()}}
# Preserve identities; assess a proxy overlap without changing the supplied split.
month='首次登記年月' if '首次登記年月' in tr else None
keys=[MODEL,YEAR,MILE]+([month] if month else [])
tk=set(map(tuple,tr[keys].fillna('NA').to_numpy())); overlap=np.array([tuple(row) in tk for row in te[keys].fillna('NA').to_numpy()]); summary['proxy_keys']=keys;summary['overlap_test']=int(overlap.sum());summary['nonoverlap_n']=int((~overlap).sum());summary['nonoverlap_mae']={k:float(abs(yt[~overlap]-preds[k][~overlap]).mean()) for k in ['A','C','RF']}
suspect=(tr[YEAR]<2008)&(tr[MODEL]=='Model Y');summary['suspect_training_rows']=list(np.where(suspect)[0]+2)
if suspect.any():
 pp,_=fitpredict(tr.loc[~suspect],y[~suspect],te,'C');summary['exclude_suspect_test_mae']=float(abs(yt-pp).mean())
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=float),encoding='utf-8')
print('SUMMARY',json.dumps(summary,ensure_ascii=False,default=float),flush=True)
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,ax=plt.subplots(1,3,figsize=(7,2.05),layout='constrained')
for a,c,label in zip(ax[:2],[MILE,YEAR],['Mileage (1,000 km)','Registration year']):
 a.scatter(tr[c]/(1000 if c==MILE else 1),y/1000,s=3,alpha=.28,color='#176b87');a.set_xlabel(label);a.set_ylabel('Asking price (1,000 EUR)')
mods=['Model 3','Model Y','Model S','Model X'];ax[2].boxplot([y[(tr[MODEL]==v).to_numpy()]/1000 for v in mods],tick_labels=['3','Y','S','X'],flierprops={'markersize':2});ax[2].set_xlabel('Tesla model');ax[2].set_ylabel('Asking price (1,000 EUR)')
fig.savefig(FIG/'eda.pdf');plt.close(fig)
fig,ax=plt.subplots(figsize=(3.3,1.75),layout='constrained');ax.scatter(p/1000,err/1000,s=7,alpha=.45,color='#176b87');ax.axhline(0,c='black',lw=.7);ax.set_xlabel('Predicted price (1,000 EUR)');ax.set_ylabel('Actual - predicted\n(1,000 EUR)');
for i,j in enumerate(worst[:3]):ax.annotate(str(i+1),(p[j]/1000,err[j]/1000),fontsize=9,color='#b02d28')
fig.savefig(FIG/'residuals.pdf');plt.close(fig)
print('DONE',flush=True)
