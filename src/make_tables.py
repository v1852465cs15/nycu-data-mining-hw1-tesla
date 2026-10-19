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
for d in read('car_comparisons.csv'):
 if d['comparison'].startswith(('no_flags','no_inter')):continue
 p=d['paired_group_p'];ps='$<.001$' if p<.001 else f'{p:.3f}'
 rows.append(d['comparison'].replace('minus C','$-$ C')+' & '+f(d['gap'])+' & ['+f(d['ci_low'])+', '+f(d['ci_high'])+'] & '+ps)
write('comparisons_table',table(rows,'Comparison & Gap & 95\\% CI & $p$','lrrr','Test-MAE gaps (EUR): positive favors C. CI: paired bootstrap over 752 test cars, 10,000 draws; $p$: exploratory paired grouped-fold $t$-test.','tab:gaps'))
rows=[]
for d in read('slices.csv'):
 rows.append(str(d['model']).replace('Model ','')+' & '+f(d['n'])+' & '+f(d['A'])+' & '+f(d['B'])+' & '+f(d['C']))
write('slices_table',table(rows,'Model & $n$ & A & B & C','lrrrr','Test MAE by model (EUR). All observed test slices are shown; small $n$ limits rare-model interpretation.','tab:slices'))
# Final revision: regenerate all final tables from saved results.
# E1--E4 are transcriptions, not newly calculated experiments.
rows=[]
natural={}
order=['Year','Mileage','Power','Equipment','Performance','Long Range','AWD','Entry trim','New generation','HW4',
       'Model S x year','Model X x year','Model Y x year','Model: Model S','Model: Model X','Model: Model Y','Model: Roadster','Model: Cybertruck','Model: Unknown',
       'Country: IT','Country: ES','Country: BE','Country: DE','Country: FR','Country: AT','Country: LU','Autopilot: FSD','Autopilot: EAP']
flags=['Performance','Long Range','AWD','Entry trim','New generation','HW4']
for name in order:
 d=C[name]; factor=S['coefficient_scales'][name] if name in flags else 1
 nm=name.replace('Model: Model ','Model ').replace('Model: ','').replace('Country: ','').replace('Autopilot: ','').replace(' x year',r'$\times$year').replace('New generation','New gen.').replace('Long Range','Long range')
 beta=d['beta']/factor/1000
 interval='---' if math.isnan(d['low']) else f"[{d['low']/factor/1000:.2f}, {d['high']/factor/1000:.2f}]"
 pv='---' if math.isnan(d['p']) else ('$<.001$' if d['p']<.001 else f"{d['p']:.3f}")
 rows.append(nm+f" & {beta:.2f} & {interval} & {pv} & {d['vif']:.1f}")
write('coefficients_table',table(rows,r'Term & $\beta$ & 95\% CI & $p$ & VIF','lrrrr',r'Final Set C: HC1 intervals, $\beta$/CI in 1,000 EUR. Continuous/interaction terms: per SD; flags: 0-to-1; categories: reference contrasts. All 28 terms shown. Rare classes (Roadster, Cybertruck, Unknown; $n<20$) are not interpreted; singleton uncertainty is suppressed.','tab:coef'))
write('interpretation',r"""\paragraph{Interpretation.} Each newer year adds EUR 1,882 for Model 3, about EUR 4,178 for S and EUR 4,994 for X; Y does not differ from Model 3. This supports heterogeneous slopes, not a causal effect of aging. Trim matters beyond power and model: HW4, new generation, Performance and Long Range carry premiums and the entry trim a discount (Table~\ref{tab:coef}). Italy (+EUR 2,775), Spain ($-$EUR 1,430) and Belgium ($-$EUR 842) differ from the Netherlands; other countries do not. FSD adds EUR 866; EAP is not significant.

AWD ($-$EUR 1,461) corrects for overlap: 88\% of AWD-flagged listings also carry Long Range or Performance and report more power (318 vs 282 kW), so those terms take most of the premium; a Performance AWD car still nets +EUR 1,480. Raw AWD listings are also cheaper (EUR 34,596 vs EUR 36,016), so the sign is not a reversal.""")
tests={(d['model'],d['protocol']):d for d in read('hypothesis_tests.csv')}
def pv(p):
 return f'{p:.4f}'.lstrip('0') if p<.001 else f'{p:.3f}'.lstrip('0')
rows=[]
for ident,comparison,kind in [('H1','Remove flags','no_flags'),('H2','Remove interactions','no_inter'),('H3','B minus C','B')]:
 cells=[]
 for protocol in ['KFold','GroupKFold']:
  d=tests[kind,protocol];cells.append(f(d['delta'])+' ('+pv(d['p'])+')')
 rows.append(ident+' & '+comparison+' & '+' & '.join(cells))
write('hypotheses_table',table(rows,r'ID & Test & KFold $\Delta$ ($p$) & Group $\Delta$ ($p$)','llrr',r'H1--H3 are supported in this benchmark. $\Delta$: MAE minus C (EUR), positive favors C. Two-sided paired $t$-tests on five matched fold MAEs; overlapping training fits make $p$ exploratory. Fold losses are archived.','tab:hyp').replace('\\small','\\footnotesize').replace('{3pt}','{2pt}'))
rows=[]
for d in read('team_extra_checks.csv'):
 pv='$<.001$' if d['p']=='<.001' else f"{d['p']:.3f}"
 rows.append(d['id']+' & '+d['check']+' & '+f(d['delta'])+' & '+pv)
write('extra_checks_table',table(rows,r'ID & Additional check & $\Delta$ MAE & $p$','llrr',r'Additional exploratory checks E1--E4: KFold MAE change from C (EUR 3,244), positive = worse; paired fold $t$-tests. E3 removes model and its year terms. These do not redefine H1--H3.','tab:extra'))
rows=[]
for d in read('slice_metrics.csv'):
 rows.append(d['model'].replace('Model ','')+' & '+f(d['n'])+' & '+f(d['A'])+' & '+f(d['B'])+' & '+f(d['C'])+' & '+f(d['gain'])+r'\% & '+f"{d['median_pct']:.1f}"+r'\% & '+f(d['within_pct'])+r'\%')
write('slices_table',table(rows,r'Model & $n$ & A & B & C & Gain & Med. & $\leq$3k','lrrrrrrr',r'Test MAE (EUR). Gain $=100(C/A-1)\%$; negative means lower error. Med.: C median absolute percentage error; $\leq$3k: share within EUR 3,000. All aggregates all 752 cars; rare classes are not interpreted.','tab:slices').replace('\\small',r'\fontsize{8}{9.5}\selectfont').replace('{3pt}','{1.6pt}'))
rows=[]
for i,d in enumerate(read('largest_errors.csv')[:5]):rows.append(str(i+1)+' & '+f(d['售價'])+' & '+f(d['pred'])+' & '+f(d['error']))
write('cases_table',table(rows,'Case & Actual & Predicted & Error','lrrr','Five largest absolute errors of Set C (EUR). Error = actual $-$ predicted. Listing context below.','tab:cases'))
write('case_discussion',r"""\textbf{Cases 1 and 4:} Model X 75D, France, 2018, and Model S 70, Spain, 2016. We find comparable training maxima EUR 39,999 (27 X ``75'' cars) and EUR 34,000 (nine S 70 cars), below their asking prices. Unusual pricing is plausible, not verified mispricing. \textbf{Case 2:} Model X 75 Dual Motor, Belgium, 2023: the title and 244 kW suggest an older 75D, conflicting with registration. \textbf{Case 3:} Model S 85, Netherlands, 2014: 670,000 km exceeds the training maximum 635,000; linear extrapolation produces the only negative prediction. \textbf{Case 5:} Model S P85 RWD, Spain, 2013: price is near the comparable median EUR 26,170, but the Performance pattern misses ``P85'', ``RWD'' activates entry trim, and the S slope lowers the prediction.

\paragraph{Shared pattern.} Old S cars are weakest (Cases 3--5; 16.7\% median percentage error). The constant euro/year slope may be too steep at old ages. We find 70\% of the top 20 errors are underpredictions (residual skew 1.14). Proposed checks include flattening age effects, a minimum-price clip, P85/P90 parsing and title/power/year consistency. These test-inspired proposals are not applied, to avoid further test tuning.""")
print('Final tables generated')
