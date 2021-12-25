import numpy as np
from pyESN import ESN
import pandas as pd
from matplotlib import pyplot as plt

dataframe = pd.read_hdf('data/bike-pickup.h5')
data = dataframe.values
print(data)
print(dataframe.shape)

esn = ESN(n_inputs = 1,
          n_outputs = 1,
          n_reservoir = 500,
          spectral_radius = 1.5,
          random_state = 42)

trainlen = 3000 * 221
future = 1368 * 221
#inputdata  = data[:, 0].flatten()
inputdata = data.flatten()
print(data[:trainlen, :1].shape)
print(data[trainlen:trainlen+future].shape)
print(inputdata.shape)



pred_training = esn.fit(np.ones(trainlen), inputdata[:trainlen])

prediction = esn.predict(np.ones(future))
print("rmse test error: \n"+str(np.sqrt(np.mean((prediction.flatten() - inputdata[trainlen:trainlen+future])**2))))

plt.figure(figsize=(11, 1.5))
plt.plot(range(0, trainlen+future), inputdata[0:trainlen+future], 'k', label="target system")
plt.plot(range(trainlen, trainlen+future), prediction, 'r', label="free running ESN")
plt.show()

'''
loadData = np.load('taxi/train.npz')
print(loadData.values())
print(loadData.files)
x = loadData['x']
print(x.shape)
y = loadData['y']
print(y.shape)
xoff = loadData['x_offsets']
print(xoff.shape)
#testData = np.load('taxi/test.npz')
#valData = np.load('taxi/val.npz')

data = np.load('mackey_glass_t17.npy') #  http://minds.jacobs-university.de/mantas/code


print(data[:2000].shape)
'''


###
