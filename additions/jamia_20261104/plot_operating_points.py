from pathlib import Path
import csv,json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

ROOT=Path(__file__).resolve().parent
(ROOT/'Audit').mkdir(exist_ok=True)
for language in ('EN','ZH'):(ROOT/'Figures'/language).mkdir(parents=True,exist_ok=True)

# Three frozen operating points. No curve interpolation or new estimation.
rows=list(csv.DictReader((ROOT/'summary_results/overall_three_schemes.csv').open(encoding='utf-8-sig')))
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],'font.size':8.5,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.right':False,'axes.spines.top':False,'axes.linewidth':.65})
CN=FontProperties(family=['Microsoft YaHei','Noto Sans CJK SC','sans-serif'],size=8.5)
for lang in ('EN','ZH'):
    en=lang=='EN';fig,axes=plt.subplots(1,2,figsize=(7.15,3.15))
    fig.subplots_adjust(left=.09,right=.985,bottom=.20,top=.86,wspace=.34)
    tasks=['US_long_ILI','CN_worry'];colors=['#467A97','#A56C8C']
    for i,(ax,task,color) in enumerate(zip(axes,tasks,colors)):
        ax.set_title((['US illness','China worry'] if en else ['美国疾病活动','中国担忧'])[i],fontproperties=None if en else CN,pad=12)
        ax.annotate(chr(97+i),xy=(0,1),xycoords='axes fraction',xytext=(-27,17),textcoords='offset points',fontweight='bold',fontsize=10)
        for scheme,marker in [('Base_original','o'),('Full_original','s'),('Full_constrained','D')]:
            r=next(z for z in rows if z['task']==task and z['scheme']==scheme)
            x,y=100*float(r['FPR']),100*float(r['TPR'])
            face='white' if scheme!='Full_original' else color
            edge='#626B73' if scheme=='Base_original' else color
            ax.scatter([x],[y],marker=marker,s=40,facecolors=face,edgecolors=edge,linewidths=1.2,zorder=3)
            label={'Base_original':('Base','Base'),'Full_original':('Full','Full'),'Full_constrained':('Constrained Full','受约束Full')}[scheme][0 if en else 1]
            offset={('US_long_ILI','Base_original'):(12,-9),('US_long_ILI','Full_original'):(-26,10),('US_long_ILI','Full_constrained'):(12,-9),('CN_worry','Base_original'):(-26,8),('CN_worry','Full_original'):(12,-11),('CN_worry','Full_constrained'):(12,3)}[(task,scheme)]
            ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',fontproperties=None if en else CN,color='#313A42',fontsize=8.5)
        ax.set_xlabel('False-positive rate (%)' if en else '误报率（%）',fontproperties=None if en else CN,labelpad=8)
        ax.set_ylabel('Sensitivity (%)' if en else '检出率（%）',fontproperties=None if en else CN,labelpad=7)
        ax.set_axisbelow(True)
        if i==0:ax.set_xlim(8.5,14);ax.set_ylim(45,70);ax.set_xticks([9,10,11,12,13,14]);ax.set_yticks([45,50,55,60,65,70])
        else:ax.set_xlim(8,40);ax.set_ylim(0,40);ax.set_xticks([10,20,30,40]);ax.set_yticks([0,10,20,30,40])
    fig.canvas.draw()
    out=ROOT/'Figures'/lang/'Figure_6'
    fig.savefig(str(out)+'.pdf');fig.savefig(str(out)+'.svg');fig.savefig(str(out)+'.png',dpi=400)
    plt.close(fig)
(ROOT/'Audit/figure6_contract.json').write_text(json.dumps({'question':'Does the validation-constrained alert rule improve held-out detection?','claim':'The specified calibration does not establish detection gains; realized false-positive rates differ.','archetype':'quantitative grid','panels':'US illness and China worry complementary outcomes; all three prespecified arms','source':'analysis_false_alarm/results/overall_three_schemes.csv','uncertainty':'Point summaries; paired uncertainty in Table 3; no ROC or joint region','axes':'Different visible ranges explicitly labelled by ticks, no cross-task ranking','backend':'Python established project workflow','exclusion':'None','size_mm':[181.61,80.01]},indent=2),encoding='utf8')
