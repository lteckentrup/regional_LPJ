import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from pylab import text
from scipy import stats
import math

fig = plt.figure(figsize=(13,4))

fig.subplots_adjust(hspace=0.25)
fig.subplots_adjust(wspace=0.1)
fig.subplots_adjust(right=0.97)
fig.subplots_adjust(left=0.07)
fig.subplots_adjust(bottom=0.27)
fig.subplots_adjust(top=0.88)

plt.rcParams['text.usetex'] = False
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['font.size'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

ax1=fig.add_subplot(1,3,1)
ax2=fig.add_subplot(1,3,2)
ax3=fig.add_subplot(1,3,3)

def Power_function(x, a, b):
    return(a*x**b)

def Michaelis_Menten(x, a, b):
    return((a*x)/(b+x))

def read_in(dataset,region):
    if dataset == 'Tallo':
        ### reference id 69 reports trees with 6m stem diameter??
        df_tallo = pd.read_csv('Tallo.csv')
        df_oz = df_tallo[(df_tallo['latitude']<-10)&(df_tallo['latitude']>-46)&
                         (df_tallo['longitude']>120)&(df_tallo['longitude']<154)&
                         (df_tallo['height_outlier']=='N')&
                         (df_tallo['height_m']!=0)&
                         (df_tallo['stem_diameter_cm']!=0)&
                         (df_tallo['reference_id'] != 69)]

        df = df_oz.dropna(subset=['stem_diameter_cm','height_m'])
        df = df.rename(columns={'stem_diameter_cm': 'Diameter',
                                'height_m': 'Height',
                                'latitude': 'Latitude'})

    elif dataset == 'Large Tree Survey':
        df = pd.read_csv('data_large_tree_survey.csv')

        ### Drop nan
        df = df[['Diameter','Height','Latitude']]
        df = df.dropna()
        df = df[(df['Diameter']!=0)&(df['Height']!=0)]

    elif dataset == 'AusTraits':
        df = pd.read_csv('tree_allom.csv')
        df = df.rename(columns={'DBH': 'Diameter'})

    if dataset in ('Tallo', 'Large Tree Survey'):
        if region == 'Tropical':
            df_region = df[df['Latitude']>-23.5]
        elif region == 'Temperate':
            df_region = df[df['Latitude']<-23.5]
        return(df_region['Diameter']/100,df_region['Height'])

    else:
        df_region = df
        return(df_region['Diameter'],df_region['Height'])

def fit_data(fit,dataset,region):
    Diameter, Height = read_in(dataset,region)

    if fit == 'Power Function':
        pars, cov = curve_fit(f=Power_function,
                              xdata=Diameter,
                              ydata=Height,
                              p0=[40,0.5])
    elif fit == 'Michaelis-Menten':
        pars, cov = curve_fit(f=Michaelis_Menten,
                              xdata=Diameter,
                              ydata=Height,
                              p0=[80,1])

    elif fit == 'Log. Trafo':
        DBH = np.array(Diameter)
        H = np.array(Height)
        TheilSen = linear_model.TheilSenRegressor(random_state=42)
        TheilSen.fit(np.log10(DBH.reshape((-1, 1))),np.log10(H))

        pars = [TheilSen.coef_,TheilSen.intercept_]

    return(pars)

def get_data_plot(axis,dataset,region):
    Diameter, Height = read_in(dataset,region)
    a_PF, b_PF = fit_data('Power Function',dataset,region)
    a_MM, b_MM = fit_data('Michaelis-Menten',dataset,region)
    a_LR, b_LR = fit_data('Log. Trafo',dataset,region)

    Diameter_sort = np.sort(Diameter, axis=None)
    Height_def = 40*Diameter_sort**0.5
    Height_PF = a_PF*Diameter_sort**b_PF
    Height_MM = (a_MM*Diameter_sort)/(b_MM+Diameter_sort)
    Height_LR = 10**(a_LR*np.log10(Diameter_sort)+b_LR)

    return(Diameter, Height, Diameter_sort, Height_def, Height_PF, Height_MM, Height_LR)

def plot_figure_scatter(axis,dataset):
    Diameter_temp, Height_temp, Diameter_sort_temp, Height_def_temp, Height_PF_temp, Height_MM_temp, Height_LR_temp = get_data_plot(axis,dataset,'Temperate')
    Diameter_trop, Height_trop, Diameter_sort_trop, Height_def_trop, Height_PF_trop, Height_MM_trop, Height_LR_trop = get_data_plot(axis,dataset,'Tropical')

    if dataset in ('Tallo', 'Large Tree Survey'):
        axis.scatter(Diameter_temp,Height_temp,c='#b2b2b2',s=2,label='Temperate (Obs.)')
        axis.scatter(Diameter_trop,Height_trop,c='#666666',s=2,label='Tropical (Obs.)')

        axis.plot(Diameter_sort_temp,Height_def_temp,color='#222255',lw='3',
                  label='Temperate (default)')
        axis.plot(Diameter_sort_trop,Height_def_trop,color='#72190E',lw='3',
                  label='Tropical (default)')
        axis.plot(Diameter_sort_temp,Height_PF_temp,color='#00678a',
                  label='Temperate (PF)')
        axis.plot(Diameter_sort_trop,Height_PF_trop,color='#DC050C',
                  label='Tropical (PF)')
        axis.plot(Diameter_sort_temp,Height_MM_temp,color='#5eccab',
                  label='Temperate (MM)')
        axis.plot(Diameter_sort_trop,Height_MM_trop,color='#EE8026',
                  label='Tropical (MM)')
        axis.plot(Diameter_sort_temp,Height_LR_temp,color='#c0affb',
                  label='Temperate (LR)')
        axis.plot(Diameter_sort_trop,Height_LR_trop,color='#F7CB45',
                  label='Tropical (LR)')

        rval_def_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_def_temp)).mean()))
        rval_def_trop = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_trop,Height_def_trop)).mean()))
        rval_PF_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_PF_temp)).mean()))
        rval_PF_trop = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_trop,Height_PF_trop)).mean()))
        rval_MM_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_MM_temp)).mean()))
        rval_MM_trop = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_trop,Height_MM_trop)).mean()))
        rval_LR_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_LR_temp)).mean()))
        rval_LR_trop = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_trop,Height_LR_trop)).mean()))

        text(0.02, 0.99, 'Temperate', color='k',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.93, 'RMSE = '+rval_def_temp, color='#222255',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.87, 'RMSE = '+rval_PF_temp, color='#00678a',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.81, 'RMSE = '+rval_MM_temp, color='#5eccab',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.75, 'RMSE = '+rval_LR_temp, color='#c0affb',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)

        if dataset == 'Tallo':
            pos_r = 0.72
        elif dataset == 'Large Tree Survey':
            pos_r = 0.7
        text(pos_r, 0.28, 'Tropical', color='k',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(pos_r, 0.22, 'RMSE = '+rval_def_trop, color='#72190E',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(pos_r, 0.16, 'RMSE = '+rval_PF_trop, color='#DC050C',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(pos_r, 0.1, 'RMSE = '+rval_MM_trop, color='#EE8026',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(pos_r, 0.04, 'RMSE = '+rval_LR_trop, color='#F7CB45',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)

    else:
        axis.scatter(Diameter_temp,Height_temp,c='#666666',s=2,label='_nolegend_')

        axis.plot(Diameter_sort_temp,Height_def_temp,color='#6699CC',lw='3',label='default')
        axis.plot(Diameter_sort_temp,Height_PF_temp,color='#44BB99',label='PF')
        axis.plot(Diameter_sort_temp,Height_MM_temp,color='#AAAA00',label='MM')
        axis.plot(Diameter_sort_temp,Height_LR_temp,color='#BBCC33',label='LR')

        rval_def_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_def_temp)).mean()))
        rval_PF_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_PF_temp)).mean()))
        rval_MM_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_MM_temp)).mean()))
        rval_LR_temp = '{:.2f}'.format(math.sqrt(np.square(np.subtract(Height_temp,Height_LR_temp)).mean()))

        text(0.02, 0.99, 'Total', color='k',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.93, 'RMSE = '+rval_def_temp,
             color='#6699CC',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.87, 'RMSE = '+rval_PF_temp,
             color='#44BB99',
             horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.81, 'RMSE = '+rval_MM_temp,
             color='#AAAA00', horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)
        text(0.02, 0.75, 'RMSE = '+rval_LR_temp,
             color='#BBCC33', horizontalalignment='left',
             verticalalignment='center',
             transform = axis.transAxes)

plot_figure_scatter(ax1,'Tallo')
plot_figure_scatter(ax2,'Large Tree Survey')
plot_figure_scatter(ax3,'AusTraits')

for a in (ax1,ax2,ax3):
    a.spines['right'].set_visible(False)
    a.spines['top'].set_visible(False)

    a.set_xlim(-0.1,5)
    a.set_ylim(-1,124)

ax1.legend(loc='best',bbox_to_anchor=(3.1, -0.2), ncol=5,frameon=False)
ax3.legend(loc='upper right',bbox_to_anchor=(1, 1.03),ncol=1,frameon=False, 
           labelspacing = 0.1)
ax1.set_title('Tallo\n(n=64227)')
ax2.set_title('Large Tree Survey\n(n=4694)')
ax3.set_title('AusTraits\n(n=343)')

ax1.set_title('a)', loc='left')
ax2.set_title('b)', loc='left')
ax3.set_title('c)', loc='left')

ax1.set_ylabel('Tree height [m]')

ax1.set_xlabel('Diameter [m]')
ax2.set_xlabel('Diameter [m]')
ax3.set_xlabel('Diameter [m]')

ax2.set_yticklabels([])
ax3.set_yticklabels([])

# plt.show()
plt.savefig('height_dbh.png',dpi=300)
