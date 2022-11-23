import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = "Times New Roman"

# Prepare data
XG1 = np.array([98.76, 98.75, 98.69]) - 97.79
RF1 = np.array([98.33, 98.19, 98.33]) - 97.91
DT1 = np.array([98.05, 97.95, 97.79]) - 98.23
ET1 = np.array([98.31, 98.21, 98.15]) - 97.53
ST1 = np.array([98.69, 98.69, 98.69]) - 98.69
dataIoT20 = np.hstack([XG1, RF1, DT1, ET1, ST1])

XG2 = np.array([99.91, 99.91, 99.90]) - 99.83
RF2 = np.array([99.88, 99.87, 99.89]) - 99.88
DT2 = np.array([99.67, 99.74, 99.72]) - 99.79
ET2 = np.array([99.87, 99.86, 99.86]) - 99.87
ST2 = np.array([99.88, 99.89, 99.88]) - 99.88
dataNSL = np.hstack([XG2, RF2, DT2, ET2, ST2])

XG3 = np.array([99.82, 99.83, 99.81]) - 99.43
RF3 = np.array([99.80, 99.79, 99.75]) - 99.70
DT3 = np.array([99.73, 99.69, 99.68]) - 99.73
ET3 = np.array([99.82, 99.78, 99.81]) - 99.79
ST3 = np.array([99.82, 99.82, 99.82]) - 99.82
data2017 = np.hstack([XG3, RF3, DT3, ET3, ST3])

XG4 = np.array([99.62, 99.62, 99.57]) - 99.51
RF4 = np.array([99.62, 99.61, 99.60]) - 99.60
DT4 = np.array([99.57, 99.56, 99.53]) - 99.55
ET4 = np.array([99.55, 99.55, 99.55]) - 99.53
ST4 = np.array([99.62, 99.60, 99.60]) - 99.61
dataCup99 = np.hstack([XG4, RF4, DT4, ET4, ST4])

XG5 = np.array([93.73, 93.99, 93.61]) - 94.14
RF5 = np.array([94.13, 94.10, 93.86]) - 93.34
DT5 = np.array([93.85, 93.71, 93.29]) - 92.70
ET5 = np.array([93.93, 93.93, 93.47]) - 93.08
ST5 = np.array([93.72, 93.71, 93.71]) - 93.69
dataNB15 = np.hstack([XG5, RF5, DT5, ET5, ST5])

data = np.array([dataIoT20, dataNSL, data2017, dataCup99, dataNB15])
data_shape = np.shape(data)

# Take negative and positive data apart and cumulate
def get_cumulated_array(data, **kwargs):
    cum = data.clip(**kwargs)
    cum = np.cumsum(cum, axis=0)
    d = np.zeros(np.shape(data))
    d[1:] = cum[:-1]
    return d

cumulated_data = get_cumulated_array(data, min=0)
cumulated_data_neg = get_cumulated_array(data, max=0)

# Re-merge negative and positive data.
row_mask = (data<0)
cumulated_data[row_mask] = cumulated_data_neg[row_mask]
data_stack = cumulated_data

# Plot
fig = plt.figure(figsize=(10,5),dpi=80)
ax1 =fig.add_subplot(1,1,1)
cols = ["g","y","b","c","r"]

width = 1
label = ['IoT20', 'NSL-KDD', 'CICIDS-2017','KDD-Cup99', 'UNSW-NB15']
xticklabels = ['BO-TPE', 'PSO', 'GA']*5
x = np.linspace(-5, 25, 100)
ax1.fill_between(x, 2.0, 0, color='#f9d4f2',alpha=0.4)
ax1.fill_between(x, 0, -1.0, color='#b8f8bb',alpha=0.4)
x = []
for i in range(5):
    for j in range(3):
        y = i*5+0.5*width+1.3*j
        x.append(y)

for i in np.arange(0, data_shape[0]):
    ax1.bar(np.array(x), data[i], bottom=data_stack[i], color=cols[i],label=label[i])


# 'Increase' and 'Decrease' Annotations
ax1.text(13,1.5, 'INCREASE',color='#ff8dee', alpha=0.8, horizontalalignment='right', verticalalignment='center', fontdict={'size':18, 'weight':700})
ax1.text(13,-0.8, 'DECREASE',color='#92fa8f',horizontalalignment='left', verticalalignment='center', fontdict={'size':18, 'weight':700})

# Decorations
ax1.set_ylim(-1.0, 2.0)
ax1.set_xlim(-1, 25)
ax1.axhline(y=0, color='black', lw=0.85)
ax1.set_xticks(np.array(x))
ax1.set_xticklabels(xticklabels)
ax1.text(1.8, -1.25, 'XGBoost', ha='center', fontsize=13, c='black')
ax1.text(6.8, -1.25, 'RF', ha='center', fontsize=13, c='black')
ax1.text(11.8, -1.25, 'DT', ha='center', fontsize=13, c='black')
ax1.text(16.8, -1.25, 'ET', ha='center', fontsize=13, c='black')
ax1.text(21.8, -1.25, 'Stacking', ha='center', fontsize=13, c='black')
ax1.legend(title='Dataset:')
ax1.set( ylabel='Accuracy (%)',title='Accuracy Change After Hyoer-parameter Optimization')
plt.show()