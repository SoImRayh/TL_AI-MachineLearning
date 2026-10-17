import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
HERE=Path(__file__).resolve().parent
R=str(Path(__file__).resolve().parents[1]/'resultados')+'/'; RAW=str(Path(__file__).resolve().parents[3]/'datasets/gta6_fan_expectations_2026.csv')
BG='#212121'; FG='#FFFFFF'; MUT='#9E9E9E'; GRID='#424242'; ACC='#64B5F6'; ACC2='#FFB74D'
plt.rcParams.update({'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,'text.color':FG,
 'axes.labelcolor':FG,'xtick.color':FG,'ytick.color':FG,'axes.edgecolor':MUT,'font.family':'serif',
 'font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':13,'axes.grid':True,'grid.color':GRID,
 'grid.linewidth':0.6,'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
out=lambda n: str(HERE/n)
raw=pd.read_csv(RAW); res=pd.read_csv(R+'treino_validacao_todos_casos.csv')
CL=['Baixo','Alto','Muito alto','Altíssimo','Máximo']
# 1 classes original vs grouped
fig,axs=plt.subplots(1,2,figsize=(9,3.4),gridspec_kw={'width_ratios':[1.6,1]})
o=raw.hype_score_1to10.value_counts().sort_index()
axs[0].bar(o.index.astype(str),o.values,color=[MUT]*6+[ACC]*4); axs[0].set_title('Notas originais (1 a 10)',fontsize=13); axs[0].grid(axis='x',visible=False)
for x,v in zip(range(10),o.values): axs[0].text(x,v+20,str(v),ha='center',fontsize=9)
d=pd.read_csv(R+'distribuicao_classes.csv',index_col=0).iloc[:,0].reindex(CL)
axs[1].bar(['1–6','7','8','9','10'],d.values,color=[MUT]+[ACC]*4); axs[1].set_title('Alvo em 5 classes',fontsize=13); axs[1].grid(axis='x',visible=False)
for x,v in zip(range(5),d.values): axs[1].text(x,v+20,str(v),ha='center',fontsize=10)
for a in axs: a.set_ylim(0,1600)
axs[0].set_ylabel('respondentes'); fig.tight_layout(); fig.savefig(out('classes.png'),dpi=200)
# 2 sinais: pagar e trailer por nota
g=raw.groupby('hype_score_1to10')[['willingness_to_pay_usd','trailer1_times_watched']].mean()
fig,axs=plt.subplots(1,2,figsize=(9,3.3))
axs[0].plot(g.index,g.willingness_to_pay_usd,'-o',color=ACC); axs[0].set(title='Quanto pagaria (US$, média)',xlabel='nota de hype',xticks=range(1,11))
axs[1].plot(g.index,g.trailer1_times_watched,'-o',color=ACC2); axs[1].set(title='Vezes que viu o trailer (média)',xlabel='nota de hype',xticks=range(1,11))
fig.tight_layout(); fig.savefig(out('sinais.png'),dpi=200)
# 3 alpha
a=pd.read_csv(R+'busca_alpha.csv'); fig,ax=plt.subplots(figsize=(6.2,3.8))
for (f,gg),c,mk in zip(a.groupby('familia',sort=False),[ACC,ACC2,FG],['o','s','^']):
    ax.plot(gg.alpha,gg.validacao_f1_macro,marker=mk,color=c,label=f.replace('LR ',''))
ax.axhline(0.0776,color=MUT,ls=':',lw=1); ax.text(1.2e-5,0.085,'baseline (teste)',color=MUT,fontsize=10)
ax.set(xscale='log',xlabel='alpha',ylabel='F1 macro (validação)',ylim=(0,0.5)); ax.legend(loc='lower left'); fig.tight_layout(); fig.savefig(out('alpha.png'),dpi=200)
# 4 lr curves
fams=['LR sem regularizacao','LR Ridge','LR Lasso','LR Elastic Net']; nm=['Sem regularização','Ridge (α=1e-4)','Lasso (α=1e-5)','Elastic Net (α=1e-4)']
fig,axs=plt.subplots(2,2,figsize=(8.6,4.6),sharex=True,sharey=True)
for ax,f,n in zip(axs.ravel(),fams,nm):
    gg=res[(res.experimento=='curva_lr')&(res.familia==f)].groupby('n_treino')[['treino_f1_macro','validacao_f1_macro']].agg(['mean','std'])
    for col,c,lab,ls in [('treino_f1_macro',MUT,'Treino','--'),('validacao_f1_macro',ACC,'Validação','-')]:
        m,s=gg[col]['mean'],gg[col]['std']; ax.plot(gg.index,m,ls,marker='o',ms=4,color=c,label=lab); ax.fill_between(gg.index,m-s,m+s,color=c,alpha=.2)
    ax.set_title(n,fontsize=13); ax.set_ylim(0.2,0.55)
axs[0,0].legend(loc='upper right',fontsize=11)
for ax in axs[1]: ax.set_xlabel('respondentes de treino')
for ax in axs[:,0]: ax.set_ylabel('F1 macro')
fig.tight_layout(); fig.savefig(out('curvas_lr.png'),dpi=200)
# 5 ad
t=res[res.experimento=='curva_ad'].sort_values('max_depth'); fig,ax=plt.subplots(figsize=(6.2,3.8))
ax.plot(t.max_depth,t.treino_f1_macro,'--o',color=MUT,ms=4,label='Treino'); ax.plot(t.max_depth,t.validacao_f1_macro,'-o',color=ACC2,ms=4,label='Validação')
b=t.loc[t.validacao_f1_macro.idxmax()]; ax.annotate(f'melhor: prof. {int(b.max_depth)}\nF1 = {b.validacao_f1_macro:.2f}'.replace('.',',',1) if False else f'melhor: prof. {int(b.max_depth)}\nF1 = '+f'{b.validacao_f1_macro:.2f}'.replace('.',','),(b.max_depth,b.validacao_f1_macro),xytext=(6.5,0.18),arrowprops=dict(arrowstyle='->',color=FG),fontsize=12)
ax.set(xlabel='profundidade máxima',ylabel='F1 macro',ylim=(0,0.7),xticks=range(1,13)); ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(out('curva_ad.png'),dpi=200)
# 6 consistencia
c=res[((res.experimento=='curva_lr')&(res.familia=='LR Ridge')&res.fracao_treino.isin([.3,.6,.9]))|(res.experimento=='consistencia_ad')]
gg=c.groupby(['familia','fracao_treino']).validacao_f1_macro.agg(['mean','std']).reset_index()
fig,ax=plt.subplots(figsize=(6.2,3.8)); x=np.arange(3); w=.36
for i,(f,col,lab) in enumerate([('LR Ridge',ACC,'LR-GD Ridge'),('AD propria',ACC2,'AD própria (prof. 5)')]):
    s=gg[gg.familia==f]; ax.bar(x+(i-.5)*w,s['mean'],w,yerr=s['std'],color=col,label=lab,capsize=4,ecolor=FG)
    for xi,m in zip(x,s['mean']): ax.text(xi+(i-.5)*w,m+.03,f'{m:.2f}'.replace('.',','),ha='center',fontsize=11)
ax.set_xticks(x,['30%\n(1.080)','60%\n(2.160)','90%\n(3.240)']); ax.set(ylabel='F1 macro (validação)',ylim=(0,0.6)); ax.grid(axis='x',visible=False); ax.legend(loc='upper left',ncol=2,fontsize=11)
fig.tight_layout(); fig.savefig(out('consistencia.png'),dpi=200)
# 7 confusion
from matplotlib.colors import LinearSegmentedColormap
cm=LinearSegmentedColormap.from_list('d',[BG,'#1E4E79',ACC])
titles=['LR sem regularização','LR Elastic Net','AD própria (prof. 5)']; lab=['1–6','7','8','9','10']
fig,axs=plt.subplots(1,3,figsize=(12,4.6))
for i,(ax,tt) in enumerate(zip(axs,titles),1):
    v=pd.read_csv(R+f'confusao_{i}_contagens.csv',index_col=0).values
    ax.imshow(v,cmap=cm,vmin=0,vmax=260); ax.grid(False)
    ax.set_xticks(range(5),lab); ax.set_yticks(range(5),lab if i==1 else ['']*5)
    for a_ in range(5):
        for b_ in range(5): ax.text(b_,a_,v[a_,b_],ha='center',va='center',fontsize=12,color=FG)
    ax.set_title(tt+'\nacurácia '+f'{np.trace(v)/v.sum():.1%}'.replace('.',','),fontsize=13); ax.set_xlabel('nota predita')
    for s in ax.spines.values(): s.set_visible(False)
axs[0].set_ylabel('nota real'); [a.tick_params(length=0) for a in axs]
fig.subplots_adjust(left=.05,right=.99,top=.86,bottom=.12,wspace=.05); fig.savefig(out('confusao.png'),dpi=200)
# 8 recall por classe LR vs AD
pc=pd.read_csv(R+'metricas_teste_por_classe.csv')
lr=pc[pc.modelo_id=='curva_lr|LR sem regularizacao|seed=42|fraction=1.0|depth=None'].set_index('classe').reindex(CL)
ad=pc[pc.modelo_id=='curva_ad|AD propria|seed=42|fraction=1.0|depth=5'].set_index('classe').reindex(CL)
fig,ax=plt.subplots(figsize=(6.2,3.8)); x=np.arange(5); w=.38
for k,(dfc,col,l) in enumerate([(lr,ACC,'LR sem regularização'),(ad,ACC2,'AD própria')]):
    ax.bar(x+(k-.5)*w,dfc.recall,w,color=col,label=l)
    for xi,v in zip(x,dfc.recall): ax.text(xi+(k-.5)*w,v+.015,f'{v:.0%}',ha='center',fontsize=10)
ax.set_xticks(x,['Baixo\n1–6','Alto\n7','Muito alto\n8','Altíssimo\n9','Máximo\n10']); ax.set(ylabel='recall no teste',ylim=(0,1)); ax.grid(axis='x',visible=False); ax.legend(loc='upper center',ncol=2,fontsize=11)
fig.tight_layout(); fig.savefig(out('recall_classes.png'),dpi=200)
# 9 distância dos erros AD
v=pd.read_csv(R+'confusao_3_contagens.csv',index_col=0).values
dist={k:int(sum(v[i,j] for i in range(5) for j in range(5) if abs(i-j)==k)) for k in range(5)}
fig,ax=plt.subplots(figsize=(5,3.4)); ax.bar([str(k) for k in dist],dist.values(),color=[ACC,ACC,MUT,MUT,MUT])
for k,val in dist.items(): ax.text(k,val+8,str(val),ha='center',fontsize=11)
ax.set(xlabel='distância entre classe real e predita',ylabel='respondentes (teste)',ylim=(0,620)); ax.grid(axis='x',visible=False)
fig.tight_layout(); fig.savefig(out('distancia_erros.png'),dpi=200)
print(dist)
# 10 consistência: 4 métricas no teste
mt=pd.read_csv(R+'metricas_teste_todos_casos.csv')
c=mt[((mt.experimento=='curva_lr')&(mt.familia=='LR Ridge')&mt.fracao_treino.isin([.3,.6,.9]))|(mt.experimento=='consistencia_ad')]
mets=[('precision_macro','Precision macro'),('recall_macro','Recall macro'),('f1_micro','F1 micro (= acurácia)'),('f1_macro','F1 macro')]
fig,axs=plt.subplots(2,2,figsize=(8.6,4.7),sharex=True,sharey=True); x=np.arange(3); w=.36
for ax,(col,tt) in zip(axs.ravel(),mets):
    gg=c.groupby(['familia','fracao_treino'])[col].agg(['mean','std']).reset_index()
    for i,(f,cc,lab) in enumerate([('LR Ridge',ACC,'LR-GD Ridge'),('AD propria',ACC2,'AD própria (prof. 5)')]):
        s=gg[gg.familia==f]; ax.bar(x+(i-.5)*w,s['mean'],w,yerr=s['std'],color=cc,label=lab,capsize=3,ecolor=FG)
        for xi,m in zip(x,s['mean']): ax.text(xi+(i-.5)*w,m+.035,f'{m:.2f}'.replace('.',','),ha='center',fontsize=9.5)
    ax.set_title(tt,fontsize=13); ax.set_ylim(0,0.6); ax.grid(axis='x',visible=False)
for ax in axs[1]: ax.set_xticks(x,['30%','60%','90%']); ax.set_xlabel('parcela do treino')
for ax in axs[:,0]: ax.set_ylabel('teste')
axs[0,0].legend(loc='upper left',ncol=2,fontsize=10)
fig.tight_layout(); fig.savefig(out('consistencia_4metricas.png'),dpi=200)
# 11 quatro gráficos separados (um por métrica)
for k,(col,tt) in enumerate(mets,1):
    fig,ax=plt.subplots(figsize=(4.4,2.5))
    gg=c.groupby(['familia','fracao_treino'])[col].agg(['mean','std']).reset_index()
    for i,(f,cc,lab) in enumerate([('LR Ridge',ACC,'LR-GD Ridge'),('AD propria',ACC2,'AD própria')]):
        s=gg[gg.familia==f]; ax.bar(x+(i-.5)*w,s['mean'],w,yerr=s['std'],color=cc,label=lab,capsize=3,ecolor=FG)
        for xi,m in zip(x,s['mean']): ax.text(xi+(i-.5)*w,m+.04,f'{m:.2f}'.replace('.',','),ha='center',fontsize=10)
    ax.set_title(tt,fontsize=14); ax.set_ylim(0,0.65); ax.grid(axis='x',visible=False)
    ax.set_xticks(x,['30%','60%','90%']); ax.tick_params(labelsize=11)
    if k==1: ax.legend(loc='upper left',ncol=2,fontsize=9)
    fig.tight_layout(); fig.savefig(out(f'consistencia_{k}_{col}.png'),dpi=200); plt.close(fig)
