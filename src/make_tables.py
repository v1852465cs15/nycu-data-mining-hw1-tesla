from pathlib import Path
import csv,json,math
ROOT=Path(__file__).resolve().parents[1]; A=ROOT/'analysis'; L=ROOT/'latex'
def read(name):
 rows=list(csv.DictReader((A/name).open(encoding='utf-8-sig')))
 for r in rows:
  for k,v in r.items():
   try:r[k]=float(v) if v else float('nan')
   except ValueError:pass
 return rows
def write(n,t):(L/(n+'.tex')).write_text(t,encoding='utf-8')
R={r['model']:r for r in read('results.csv')}; S=json.loads((A/'summary.json').read_text(encoding='utf-8')); C={r['feature']:r for r in read('coefficients.csv')}
def f(v):return f'{v:,.0f}'
mac={
'GroupA':f(R['A']['group_mae']),'GroupC':f(R['C']['group_mae']),'TestA':f(R['A']['test_mae']),'TestC':f(R['C']['test_mae']),'TestRF':f(R['RF']['test_mae']),
'PriceMin':f(S['numeric_summary']['售價']['min']),'PriceMax':f(S['numeric_summary']['售價']['max']),
'MileageR':f"${S['correlations']['Mileage']['pearson']:.3f}$",'YearR':f"${S['correlations']['Year']['pearson']:.3f}$",'PowerR':f"${S['correlations']['Power']['pearson']:.3f}$",'YearMileageR':f"${S['year_mileage_r']:.3f}$",'SelectedK':str(S['selected_k']),'NumParameters':str(S['p']),
'RidgeBest':f(min(S['ridge'],key=lambda r:r[1])[1]),'RidgeAlpha':str(min(S['ridge'],key=lambda r:r[1])[0]),'TrainGroups':f(S['dealer_groups_train']),
'GainAC':f(R['A']['test_mae']-R['C']['test_mae']),'GainPct':f"{100*(1-R['C']['test_mae']/R['A']['test_mae']):.1f}",'GapRF':f(R['C']['test_mae']-R['RF']['test_mae']),
'ExcessC':f(R['C']['test_mae']-3000),'ExcessRF':f(R['RF']['test_mae']-3000),'WithinC':f"{100*R['C']['within3000']:.1f}",'WithinRF':f"{100*R['RF']['within3000']:.1f}",
'OverlapN':str(S['overlap_test']),'NonoverlapN':str(S['nonoverlap_n']),'NonoverlapA':f(S['nonoverlap_mae']['A']),'NonoverlapC':f(S['nonoverlap_mae']['C']),'NonoverlapRF':f(S['nonoverlap_mae']['RF']),'SuspectMAE':f(S['exclude_suspect_test_mae'])}
write('numbers','\n'.join('\\newcommand{\\'+k+'}{'+v+'}' for k,v in mac.items()))
def table(rows,head,cols,caption,label):
 return '\\begin{table}[h!]\n\\centering\\small\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{'+cols+'}\n\\toprule\n'+head+'\\\\\n\\midrule\n'+'\\\\\n'.join(rows)+'\\\\\n\\bottomrule\n\\end{tabular}\n\\caption{'+caption+'}\n\\label{'+label+'}\n\\end{table}\n'
rows=[]
for key,name in [('售價','Price (EUR)'),('年份（首次登記）','Year'),('里程（km）','Mileage (km)'),('功率（kW）','Power (kW)'),('配備數量','Equipment')]:
 d=S['numeric_summary'][key]
 if name=='Year': rows.append(f"Year & {d['mean']:.2f} & {d['std']:.2f} & {d['min']:.0f}--{d['max']:.0f}")
 else: rows.append(name+' & '+f(d['mean'])+' & '+f(d['std'])+' & '+f(d['min'])+'--'+f(d['max']))
write('summary_table',table(rows,'Variable & Mean & SD & Range','lrrr','Training-only summary ($n=3,002$). Inherited fields retain suspect values; range is observed, not a validity bound.','tab:data'))
rows=[]
for kind,name in [('mean','Mean'),('simple','Mileage'),('RF','RF (raw)'),('A','OLS A'),('C','OLS C (ours)')]:
 d=R[kind]; vals=[f(d['group_mae'])+' $\\pm$ '+f(d['group_sd']),f(d['test_mae']),f(d['test_rmse']),f"{d['test_r2']:.3f}"]
 if kind=='RF':vals=['\\textbf{'+v.replace(' $\\pm$ ','} $\\pm$ \\textbf{')+'}' for v in vals]
 rows.append(name+' & '+' & '.join(vals))
write('results_table',table(rows,'Method & Group MAE & Test MAE & RMSE & $R^2$','lrrrr','Main results: errors in EUR; Group MAE is five-fold mean $\\pm$ sample SD. RMSE and $R^2$ use all 752 test rows. Bold marks the best values.','tab:results'))
rows=[]
for k,name in [('A','A: raw'),('B','B: top-5'),('C','C: full'),('no_flags','C $-$ flags'),('no_inter','C $-$ interactions')]:
 d=R[k];rows.append(name+' & '+f(d['cv_mae'])+' $\\pm$ '+f(d['cv_sd'])+' & '+f(d['group_mae'])+' $\\pm$ '+f(d['group_sd'])+' & '+f(d['test_mae']))
write('ablation_table',table(rows,'Set & KFold MAE & Group MAE & Test MAE','lrrr','Matched OLS ablations, EUR. All folds and preprocessing rules are shared. Lower is better; C is best in every column.','tab:abl').replace('C: full &','\\textbf{C: full} &'))
rows=[]
for d in read('comparisons.csv'):
 if d['comparison'].startswith(('no_flags','no_inter')):continue
 p=d['paired_group_p'];ps='$<.001$' if p<.001 else f'{p:.3f}'
 rows.append(d['comparison'].replace('minus C','$-$ C')+' & '+f(d['gap'])+' & ['+f(d['ci_low'])+', '+f(d['ci_high'])+'] & '+ps)
write('comparisons_table',table(rows,'Comparison & Gap & 95\\% cluster CI & $p$','lrrr','Test-MAE gaps (EUR): positive favors C. CI: paired dealer-cluster bootstrap, 10,000 draws; $p$: exploratory paired grouped-fold $t$-test.','tab:gaps'))
rows=[]
for d in read('slices.csv'):
 rows.append(str(d['model']).replace('Model ','')+' & '+f(d['n'])+' & '+f(d['A'])+' & '+f(d['B'])+' & '+f(d['C']))
write('slices_table',table(rows,'Model & $n$ & A & B & C','lrrrr','Test MAE by model (EUR). All observed test slices are shown; small $n$ limits rare-model interpretation.','tab:slices'))
rows=[]
for name,d in C.items():
 nm=name.replace('Model: Model ','Model ').replace('Model: ','').replace('Country: ','').replace('Autopilot: ','').replace(' x year',r'$\times$year').replace('New generation','New gen.').replace('Long Range','Long range')
 interval='---' if math.isnan(d['low']) else f"[{d['low']/1000:.2f}, {d['high']/1000:.2f}]"
 p='---' if math.isnan(d['p']) else ('$<.001$' if d['p']<.001 else f"{d['p']:.3f}")
 rows.append(nm+f" & {d['beta']/1000:.2f} & {interval} & {p} & {d['vif']:.1f}")
write('coefficients_table',table(rows,r'Term & $\beta$ & 95\% CI & $p$ & VIF','lrrrr',r'Final Set C coefficients and HC3 intervals in 1,000 EUR. Standardized numeric terms, reference-coded category contrasts; all 28 non-intercept terms shown. Cybertruck ($n=1$): uncertainty unavailable. Model contrasts apply at the training mean year.','tab:coef'))
sc=S['coefficient_scales'];eff=lambda k:C[k]['beta']/sc[k]
write('interpretation',r'At the training mean registration year, Model X and S contrasts are EUR '+f(C['Model: Model X']['beta'])+' and '+f(C['Model: Model S']['beta'])+r' relative to Model 3 (Table~\ref{tab:coef}). The common year slope is EUR '+f(eff('Year'))+r' per year newer; S and X add '+f(eff('Model S x year'))+' and '+f(eff('Model X x year'))+r' EUR/year. This supports heterogeneous slopes, not a causal effect of aging. The Model Y interaction interval crosses zero, so the group gain does not establish a distinct Y slope.'+'\n\n'+r'Mileage contributes about EUR '+f(eff('Mileage')*10000)+r' per additional 10,000 km. A Performance mention corresponds to about EUR '+f(eff('Performance'))+r' holding the other terms fixed. AWD has a negative conditional coefficient despite its intuitive premium: it overlaps power, model and trim flags, and is a seller-text indicator. VIF reaches '+f"{max(d['vif'] for d in C.values()):.1f}"+r', so correlated design and omitted trim details are plausible explanations. Equipment and EAP intervals cross zero; retaining them preserves the matched raw-feature baseline, not a claim that they are individually essential.')
rows=[]
cases=read('largest_errors.csv')[:3]
for i,d in enumerate(cases):rows.append(str(i+1)+' & '+f(d['售價'])+' & '+f(d['pred'])+' & '+f(d['error']))
write('cases_table',table(rows,'Case & Actual & Predicted & Error','lrrr','Three largest absolute errors of final Set C (EUR). Error = actual $-$ predicted; context and explanations below.','tab:cases'))
write('case_discussion',r'\textbf{Case 1:} Model X 75D Dual Motor, France, 2018, 49,931 km, 193 kW. The EUR 75,900 advertisement is far above its estimate. The supplied C analysis found 27 comparable 2017--2019 Model X ``75'' training listings with median EUR 27,900 and maximum EUR 39,999. Unobserved condition or unusual pricing is plausible; an error cannot prove mispricing.'+'\n\n'+r'\textbf{Case 2:} Model X ``75 kWh Dual Motor'', Belgium, 2023, 32,000 km, 244 kW. The high asking price and registration year conflict with the older-looking title. Incorrect metadata or an unobserved configuration could explain the miss; neither is independently verified.'+'\n\n'+r'\textbf{Case 3:} Model S 85 Base, Netherlands, 2014, 670,000 km, 300 kW. Mileage exceeds the training maximum of 635,000 km. The linear extrapolation produces EUR $-15,506$, despite an actual EUR 12,000. This is a model failure, not just noisy labels. C produces one negative test prediction. The shared underprediction pattern in Figure~\ref{fig:resid} motivates input-consistency and coverage flags first; validated battery condition and a new held-out evaluation should precede any automatic correction.')
print('Tables generated')
