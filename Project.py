import numpy as np
import pandas as pd
import seaborn as sns;
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')
sns.set(context="paper", font="monospace")

maize = pd.read_csv('maize.csv')
soy = pd.read_csv('soybean.csv')
matplotlib.pyplot.close("all")
#plt.ioff()
#Part 1

prinos = pd.concat((maize['Yield'],soy['Yield']), axis = 1)
prinos.columns = ('Maize', 'Soybean')
m_m = prinos['Maize'].mean()     #Maize mean yield
m_g = prinos['Maize'].std()      #Maize standard deviation
s_m = prinos['Soybean'].mean()  #Soybean mean yield
s_g = prinos['Soybean'].std()   #Maize standard deviation
prinos.plot.hist(alpha=0.8, bins = 40, figsize=(18.5, 10.5))
plt.axvline(m_m, color='k', linestyle='solid', linewidth=3)
plt.axvline(s_m, color='k', linestyle='solid', linewidth=3)
plt.legend(fontsize = 20)


plt.xlabel('[kg]', fontsize = 25)
plt.ylabel('')
plt.xticks(fontsize = 20)
plt.yticks(fontsize = 20)
plt.savefig('part1.png', bbox_inches='tight', dpi=450)

#Part 2222222222222222222222222222222222222222222222222222222222222222222222222

mt = {'Tsr_04':'April','Tsr_05':'May','Tsr_06':'June','Tsr_07':'July','Tsr_08':'August','Tsr_09':'September','Tsr_10':'October'}
mp = {'Pmm_04':'April','Pmm_05':'May','Pmm_06':'June','Pmm_07':'July','Pmm_08':'August','Pmm_09':'September','Pmm_10':'October'}

mSr = maize.mean()                      #all means
mSD = maize.std()                       #all standard deviations
mTAvg = pd.DataFrame(mSr[14:21])
mTSD = pd.DataFrame(mSD[14:21])         #temperature errors
mP = pd.DataFrame(mSr[21:28])           #precipitation averages
mPSD = pd.DataFrame(mSD[21:28])         #precipitation errors
mTAvg.rename(index = mt, inplace = True)
mP.rename(index = mp, inplace = True)
mTSD.rename(index = mt, inplace = True)
mPSD.rename(index = mp, inplace = True)


fig1, ax1 = plt.subplots(nrows=2, ncols=1, sharex = True,figsize=(18.5, 10.5))
mTAvg.plot(ax=ax1[0], yerr = mTSD,legend = None, color = 'r', fontsize = 20)
mP.plot(ax=ax1[1], yerr = mPSD, color = 'b', legend = None, fontsize = 20)

ax1[0].set_ylabel('Average temperature [°C]', fontsize = 20)
ax1[1].set_ylabel('Total precipitation [mm]', fontsize = 20)

plt.savefig('part2.png', bbox_inches='tight', dpi=450)

#Part 3

maizeCorr = maize.corr()                         #the Yield correlations will be used later
maizeACorr = maizeCorr.drop('Yield',1)           #dropping the Yield column
maizeACorr = maizeACorr.drop('Yield')            #dropping the Yield row
soyCorr = soy.corr()
soyACorr = soyCorr.drop('Yield',1)
soyACorr = soyACorr.drop('Yield')

# Draw the heatmaps using seaborn


f3m, ax3m = plt.subplots(figsize=(18.5, 10.5))
sns.set(font_scale=2)
sns.heatmap(maizeACorr, vmin = -1,vmax=1, square=True)
ax3m.set_title('Maize', fontsize = 30)
ax3m.tick_params(labelsize=16)
plt.yticks(rotation=0) 
plt.xticks(rotation = 90)
f3m.tight_layout()
f3m.savefig('part3m.png',bbox_inches='tight', dpi = 450)

f3s, ax3s = plt.subplots(figsize = (18.5, 10.5))
sns.heatmap(soyACorr, vmin = -1, vmax = 1, square = True)
#ax3s.set_title('Soybean', fontsize= 30)
ax3s.tick_params(labelsize = 16)
plt.yticks(rotation = 0)
plt.xticks(rotation = 90)
sns.set(font_scale=1)
f3s.tight_layout()
f3s.savefig('part3s.png',bbox_inches='tight', dpi=450)

#Part 4444444444444444444444444444444444444444444444444444444444444444444444444

fig4m, ax4m = plt.subplots(nrows=2, ncols=1, figsize=(18.5, 10.5))
ax4m[0].tick_params(labelsize=20)
ax4m[1].tick_params(labelsize=20)
ax4m[0].set_title('Maize', fontsize = 30)
ax4m[0].set_ylabel('Temperature[°C]', fontsize = 25)
maizeR = maize.drop('Yield', 1)
maizeT1 = maizeR.iloc[:,0:7]
maizeT2 = maizeR.iloc[:,7:14]
maizeT3 = maizeR.iloc[:,14:21]
maizeT3.rename(columns=mt,inplace=True)
maizeT1.boxplot(ax=ax4m[0])
maizeT2.boxplot(ax=ax4m[0])
maizeT3.boxplot(ax=ax4m[0],fontsize = 12)
maizeR2 = maizeR.iloc[:,21::]
maizeR2.boxplot(ax=ax4m[1], rot = -45, fontsize = 15)

fig4s, ax4s = plt.subplots(nrows=2, ncols =1, figsize=(18.5, 10.5))
ax4s[0].tick_params(labelsize=20)
ax4s[1].tick_params(labelsize=20)
ax4s[0].set_title('Soybean', fontsize = 30)
ax4s[0].set_ylabel('Temperature[°C]', fontsize = 25)
soyR = soy.drop('Yield',1)
soyT1 = soyR.iloc[:,0:7]
soyT2 = soyR.iloc[:,7:14]
soyT3 = soyR.iloc[:,14:21]
soyT3.rename(columns=mt, inplace = True)
soyT1.boxplot(ax=ax4s[0])
soyT2.boxplot(ax=ax4s[0])
soyT3.boxplot(ax=ax4s[0], fontsize = 20)
soyR2 = soyR.iloc[:,21::]
soyR2.boxplot(ax=ax4s[1],rot = -45, fontsize = 15)

fig4s.savefig('part4m.png',bbox_inches='tight', dpi = 450)
fig4m.savefig('part4s.png',bbox_inches='tight', dpi = 450)

#PART55555555555555555555555555555555555555555555555555555555555555555555555555

maizeYCorr = maizeCorr['Yield']
soyYCorr = soyCorr['Yield']

fig5m, ax5m = plt.subplots(nrows = 2, ncols = 4, sharey = True, figsize = (18.5, 10.5))
fig5m.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
ax5m[0,0].tick_params(labelsize=15)
ax5m[0,1].tick_params(labelsize=15)
ax5m[0,2].tick_params(labelsize=15)
ax5m[0,3].tick_params(labelsize=15)
ax5m[1,0].tick_params(labelsize=15)
ax5m[1,1].tick_params(labelsize=15)
ax5m[1,2].tick_params(labelsize=15)
ax5m[1,3].tick_params(labelsize=15)

ax5m[0,0].set_title(str(maizeYCorr['Pmm_06']), fontsize = 20)
ax5m[0,1].set_title(str(maizeYCorr['ETR_06']), fontsize = 20)
ax5m[0,2].set_title(str(maizeYCorr['Pmm_08']), fontsize = 20)
ax5m[0,3].set_title(str(maizeYCorr['ETR_08']), fontsize = 20)
ax5m[1,0].set_title(str(maizeYCorr['Tsr_08']), fontsize = 20)
ax5m[1,1].set_title(str(maizeYCorr['Tsr_06']), fontsize = 20)
ax5m[1,2].set_title(str(maizeYCorr['Tsr_09']), fontsize = 20)
ax5m[1,3].set_title(str(maizeYCorr['Tmin_07']), fontsize = 20)

fig5m.suptitle('Maize', fontsize = 50)

sns.regplot(x=maize['Pmm_06'], y = maize['Yield'],ax = ax5m[0,0], truncate = True)
sns.regplot(x = maize['ETR_06'],y = maize['Yield'],ax = ax5m[0,1], truncate=True)
sns.regplot(x = maize['Pmm_08'],y = maize['Yield'],ax = ax5m[0,2], truncate=True)
sns.regplot(x = maize['ETR_08'],y = maize['Yield'],ax = ax5m[0,3],truncate=True)
sns.regplot(x = maize['Tsr_08'],y = maize['Yield'],ax = ax5m[1,0],truncate=True)
sns.regplot(x = maize['Tsr_06'],y = maize['Yield'],ax = ax5m[1,1],truncate=True)
sns.regplot(x = maize['Tsr_09'],y = maize['Yield'],ax = ax5m[1,2],truncate=True)
sns.regplot(x = maize['Tmin_07'],y = maize['Yield'],ax = ax5m[1,3],truncate=True)

ax5m[0,0].set_xlabel(r'$P_{06}$ [mm]', fontsize = 25)
ax5m[0,1].set_xlabel(r'$E_{TR}06$', fontsize = 25)
ax5m[0,2].set_xlabel(r'$P_{08}$ [mm]', fontsize = 25)
ax5m[0,3].set_xlabel(r'$E_{TR}08$', fontsize = 25)
ax5m[1,0].set_xlabel(r'$T_{sr}08$[°C]', fontsize = 25)
ax5m[1,1].set_xlabel(r'$T_{sr}06$ [°C]', fontsize = 25)
ax5m[1,2].set_xlabel(r'$T_{sr}09$ [°C]', fontsize = 25)
ax5m[1,3].set_xlabel(r'$T_{min}07$ [°C]', fontsize = 25)

ax5m[0,0].set_ylabel('Yield', fontsize = 25)
ax5m[0,1].set_ylabel(' ')
ax5m[0,2].set_ylabel(' ')
ax5m[0,3].set_ylabel(' ')
ax5m[1,0].set_ylabel('Yield', fontsize = 25)
ax5m[1,1].set_ylabel(' ')
ax5m[1,2].set_ylabel(' ')
ax5m[1,3].set_ylabel(' ')

ax5m[0,0].set_xlim([-10,250])
ax5m[0,1].set_xlim([10,130])
ax5m[0,2].set_xlim([-10,160])
ax5m[0,3].set_xlim([-10,130])

ax5m[1,0].set_xlim([19,25])
ax5m[1,1].set_xlim([17.5,25])
ax5m[1,2].set_xlim([14,19.5])
ax5m[1,3].set_xlim([13.5,18.5])

ax5m[0,1].set_ylim([2000,7500])
ax5m[1,1].set_ylim([2000,7500])

fig5s, ax5s = plt.subplots(nrows = 2, ncols = 4, sharey = True,figsize=(18.5, 10.5))
fig5s.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.3)
ax5s[0,0].tick_params(labelsize=15)
ax5s[0,1].tick_params(labelsize=15)
ax5s[0,2].tick_params(labelsize=15)
ax5s[0,3].tick_params(labelsize=15)
ax5s[1,0].tick_params(labelsize=15)
ax5s[1,1].tick_params(labelsize=15)
ax5s[1,2].tick_params(labelsize=15)
ax5s[1,3].tick_params(labelsize=15)

ax5s[0,0].set_title(str(soyYCorr['Pmm_06']), fontsize = 20)
ax5s[0,1].set_title(str(soyYCorr['ETR_06']), fontsize = 20)
ax5s[0,2].set_title(str(soyYCorr['Pmm_08']), fontsize = 20)
ax5s[0,3].set_title(str(soyYCorr['ETR_08']), fontsize = 20)
ax5s[1,0].set_title(str(soyYCorr['Tsr_08']), fontsize = 20)
ax5s[1,1].set_title(str(soyYCorr['Tsr_06']), fontsize = 20)
ax5s[1,2].set_title(str(soyYCorr['Pmm_10']), fontsize = 20)
ax5s[1,3].set_title(str(soyYCorr['Tmin_10']), fontsize = 20)

fig5s.suptitle('Soybean', fontsize = 30)

sns.regplot(x=soy['Pmm_06'], y = soy['Yield'],ax = ax5s[0,0], truncate = True)
sns.regplot(x = soy['ETR_06'],y = soy['Yield'],ax = ax5s[0,1], truncate=True)
sns.regplot(x = soy['Pmm_08'],y = soy['Yield'],ax = ax5s[0,2], truncate=True)
sns.regplot(x = soy['ETR_08'],y = soy['Yield'],ax = ax5s[0,3],truncate=True)
sns.regplot(x = soy['Tsr_08'],y = soy['Yield'],ax = ax5s[1,0],truncate=True)
sns.regplot(x = soy['Tsr_06'],y = soy['Yield'],ax = ax5s[1,1],truncate=True)
sns.regplot(x = soy['Pmm_10'],y = soy['Yield'],ax = ax5s[1,2],truncate=True)
sns.regplot(x = soy['Tmin_10'],y = soy['Yield'],ax = ax5s[1,3],truncate=True)

ax5s[0,0].set_xlabel(r'$P_{06}$ [mm]', fontsize = 25)
ax5s[0,1].set_xlabel(r'$E_{TR}06$', fontsize = 25)
ax5s[0,2].set_xlabel(r'$P_{08}$ [mm]', fontsize = 25)
ax5s[0,3].set_xlabel(r'$E_{TR}08$', fontsize = 25)
ax5s[1,0].set_xlabel(r'$T_{sr}08$ [°C]', fontsize = 25)
ax5s[1,1].set_xlabel(r'$T_{sr}06$ [°C]', fontsize = 25)
ax5s[1,2].set_xlabel(r'$P_{mm}10$ [mm]', fontsize = 25)
ax5s[1,3].set_xlabel(r'$T_{min}10$ [°C]', fontsize = 25)

ax5s[0,0].set_ylabel('Yield', fontsize = 25)
ax5s[0,1].set_ylabel(' ')
ax5s[0,2].set_ylabel(' ')
ax5s[0,3].set_ylabel(' ')
ax5s[1,0].set_ylabel('Yield', fontsize = 25)
ax5s[1,1].set_ylabel(' ')
ax5s[1,2].set_ylabel(' ')
ax5s[1,3].set_ylabel(' ')

ax5s[0,0].set_xlim([-10,250])
ax5s[0,1].set_xlim([10,130])
ax5s[0,2].set_xlim([-10,160])
ax5s[0,3].set_xlim([-10,120])

ax5s[1,0].set_xlim([19,25])
ax5s[1,1].set_xlim([17.5,25])
ax5s[1,2].set_xlim([0,150])
ax5s[1,3].set_xlim([4.5,10])

ax5s[0,1].set_ylim([750,3300])
ax5s[1,1].set_ylim([750,3300])

fig5m.savefig('part5m.png',bbox_inches='tight', dpi = 450)
fig5s.savefig('part5s.png',bbox_inches='tight', dpi = 450)

#Part 6

etM = maize.iloc[:,28:-1]
etpM = np.array(etM.iloc[:,0:5])
etrM = np.array( etM.iloc[:,5:10])
col = ['Wa_05','Wa_06','Wa_07','Wa_08', 'Wa_09']
etwM = etpM - etrM
etwM = pd.DataFrame(data=etwM,columns = col)
etwM['Yield'] = maize['Yield']
mwr = etwM.corr()

fig6 = plt.figure()
plt.suptitle('Maize', fontsize = 20)
ax3mt = plt.subplot2grid((2,3), (0,0), colspan=3)
ax3mp = plt.subplot2grid((2,3), (1,0))
ax3metp = plt.subplot2grid((2,3), (1, 1))
ax3metr = plt.subplot2grid((2,3), (1, 2))
ax3mt.tick_params(labelsize=10)
ax3mp.tick_params(labelsize=10)
ax3metp.tick_params(labelsize=10)
ax3metr.tick_params(labelsize=10)

mer = {'ETR_05':'May','ETR_06':'June','ETR_07':'July','ETR_08':'August', 'ETR_09':'September'}
mep = {'ETP_05':'May','ETP_06':'June','ETP_07':'July','ETP_08':'August', 'ETP_09':'September'}

ax3mt.set_ylabel('Temperature [°C]', fontsize = 12)
ax3mp.set_ylabel('Precipitation [mm]',fontsize = 12)
ax3mp.set_ylim([0,210])
ax3metp.set_ylabel('Potential evapotranspiration', fontsize = 12)
ax3metr.set_ylabel('Real evapotranspiration', fontsize = 12)

maizeP = maizeR.iloc[:,21:28]
maizeETP = maizeR.iloc[:,28:33]
maizeETR = maizeR.iloc[:,33::]
maizeP.rename(columns = mp, inplace = True)
maizeETP.rename(columns = mep, inplace = True)
maizeETR.rename(columns = mer, inplace = True)
maizeT1.boxplot(ax=ax3mt)
maizeT2.boxplot(ax=ax3mt)
maizeT3.boxplot(ax=ax3mt,fontsize = 12)
maizeP.boxplot(ax=ax3mp, rot = -45, fontsize = 11)
maizeETP.boxplot(ax=ax3metp, rot = -60, fontsize = 11)
maizeETR.boxplot(ax=ax3metr, rot = -45, fontsize = 11)

plt.tight_layout(pad=2.6, w_pad=0.5, h_pad=1.0)
plt.savefig('part6m.png',dpi =450, figsize=(5,10))
plt.show()
#PART77777777777777777777777777777777777777777777777777777777777777777777777777

print('Done!')