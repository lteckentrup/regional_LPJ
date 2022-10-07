import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(10.5,6))

fig.subplots_adjust(hspace=0.22)
fig.subplots_adjust(wspace=0.5)
fig.subplots_adjust(right=0.95)
fig.subplots_adjust(left=0.07)
fig.subplots_adjust(bottom=0.15)
fig.subplots_adjust(top=0.95)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

gs=GridSpec(2,4)

ax1=fig.add_subplot(gs[0,:3])
ax2=fig.add_subplot(gs[0,3:])
ax3=fig.add_subplot(gs[1,:3])
ax4=fig.add_subplot(gs[1,3:])

def get(trait,var):
    df = pd.read_csv(trait+'.csv')

    if trait in ('SLA', 'CTON'):
        df_TeBE = df[(df['PGF']=='tree')&(df['PHEN']=='evergreen')&(df['lat']<-23.5)]
        df_TrBE = df[(df['PGF']=='tree')&(df['PHEN']=='evergreen')&(df['lat']>=-23.5)]
        df_TeBS = df[(df['PGF']=='tree')&(df['PHEN']=='deciduous')&(df['lat']<-23.5)]
        df_TrBS = df[(df['PGF']=='tree')&(df['PHEN']=='deciduous')&(df['lat']>=-23.5)]

        df_grass = df[(df['PGF']=='herb')|(df['PGF']=='graminoid')]
        df_C3_tropical = df_grass[(df_grass['PHOT']=='c3')&(df_grass['lat']>=-23.5)]
        df_C3_temperate = df_grass[(df_grass['PHOT']=='c3')&(df_grass['lat']<-23.5)]
        df_C4_tropical = df_grass[(df_grass['PHOT']=='c4')&(df_grass['lat']>=-23.5)]
        df_C4_temperate = df_grass[(df_grass['PHOT']=='c4')&(df_grass['lat']<-23.5)]
    else:
        df_tree = df[(df['PGF']=='tree')]
        df_shrub = df[(df['PGF']=='shrub')]

    if trait == 'SLA':
        return(df_TeBE.value.astype(float),
               df_TrBE.value.astype(float),
               df_TeBS.value.astype(float),
               df_TrBS.value.astype(float),
               df_C3_temperate.value.astype(float),
               df_C3_tropical.value.astype(float),
               df_C4_temperate.value.astype(float),
               df_C4_tropical.value.astype(float)
               )
    elif var == 'CTON':
        return(df_TeBE[var],
               df_TrBE[var],
               df_TrBS[var],
               df_TrBS[var],
               df_C3_temperate[var],
               df_C3_tropical[var],
               df_C4_temperate[var],
               df_C4_tropical[var]
               )
    elif var == 'wooddens':
        return(df_tree[var], df_shrub[var])
    elif var == 'klatosa':
        return(1/df_tree.value.astype(float), 1/df_shrub.value.astype(float))

def boxplot(trait,var,axis):
    if trait in ('SLA', 'CTON'):
        (TeBE,TrBE,TeBS,TrBS,C3_temperate,
         C3_tropical,C4_temperate,C4_tropical) = get(trait,var)
        data = [TeBE,TrBE,TeBS,TrBS,C3_temperate,C3_tropical,
                C4_temperate,C4_tropical]
        colors = ['#00678a','#984464','#00678a','#984464','#00678a','#984464',
                  '#00678a','#984464']
        width = 0.5
    else:
        tree, shrub = get(trait,var)
        data = [tree, shrub]
        colors = ['#5eccab','#c0affb']
        width = 0.5

    # Creating plot
    bp = axis.boxplot(data,showfliers=False,widths=width,patch_artist=True)

    for box,color in zip(bp['boxes'],colors):
        # change outline color
        box.set(color=color, linewidth=2)
        # change fill color
        box.set(facecolor = color)

    for median in bp['medians']:
        median.set(color ='k',linewidth=1.5)

    for i in range(len(colors)):
        y = data[i]
        x = np.random.normal(1+i, 0.06, size=len(y))
        axis.scatter(x, y, color='tab:grey', s=0.5,alpha=0.3,zorder=10)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)
    if axis == ax3:
        axis.legend([bp['boxes'][0], bp['boxes'][1]], ['Temperate', 'Tropical'],
                    loc='lower right',bbox_to_anchor=(0.7, -0.4),ncol=2,
                    frameon=False)
    elif axis == ax4:
        axis.legend([bp['boxes'][0], bp['boxes'][1]], ['Tree', 'Shrub'],
                    loc='lower right',bbox_to_anchor=(-0.15, -0.4),ncol=2,
                    frameon=False)

    if trait in ('SLA', 'CTON'):
        axis.axvline(2.5,color='k',alpha=0.7,lw=0.5)
        axis.axvline(4.5,color='k',alpha=0.7,lw=0.5)
        axis.axvline(6.5,color='k',alpha=0.7,lw=0.5)


boxplot('CTON','CTON',ax1)
boxplot('wooddens','wooddens',ax2)
boxplot('SLA','SLA',ax3)
boxplot('klatosa','klatosa',ax4)

ax1.set_ylabel('CTON [-]')
ax3.set_ylabel('SLA [m$^{2}$ kg$^{-1}$]')
ax2.set_ylabel(r'$\mathrm{\rho_{wood}}$[kgC m$^{-3}$]')
ax4.set_ylabel('LA:SA[-]')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')
ax4.set_title('d)', loc='left')

ax1.set_title('Leaf carbon to leaf nitrogen')
ax3.set_title('Specific leaf area')
ax2.set_title('Wood density')
ax4.set_title('LA:SA')

ax1.set_ylim(-10,110)
ax3.set_ylim(-10,90)
ax4.set_ylim(-500,18000)

ax1.set_xticks([1.5,3.5,5.5,7.5])
ax1.set_xticklabels([])

ax2.set_xticks([1,2])
ax2.set_xticklabels([])

ax3.set_xticks([1.5,3.5,5.5,7.5])
ax3.set_xticklabels(['Evergreen', 'Deciduous', 'C$_3$ grass', 'C$_4$ grass'])

ax4.set_xticks([1,2])
ax4.set_xticklabels(['Tree', 'Shrub'])

# show plot
# plt.show()
plt.savefig('plant_traits.png',dpi=300)
