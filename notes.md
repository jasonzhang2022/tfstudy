+ monitor the loss durin the training
  + try different features to see which is has more impact
  + divide the training in many peroid. Each period, we calculate the RMSE(loss). Plot a loss curve to make sure the
    loss decreases and eventually become flat.
  + adjust batch size, learning rate, steps
+ check features
  + synthentic features if your sense correlations
  + numeric or categorical features
  + plot feature to label to see whether it match your expetaction
  + use histogram to remove outlier. You discount outlier
  + plot features in training set, valiation set to make sure  samples in both sets are similar
  + check eacg features value to make sure you understand what it is.
  + feature scaling if some feature is too big.
    + converge quickly
    + avoid NaN
    + avoid too much weight for this feature
  + avoid use descrecret values that appear a few time in sample
  
For simplicity's sake in the latitude example, we used whole numbers as bin boundaries. 
Had we wanted finer-grain resolution, we could have split bin boundaries at, say, 
every tenth of a degree. Adding more bins enables the model to learn different behaviors 
from latitude 37.4 than latitude 37.5, 
***but only if there are sufficient examples at each tenth of a latitude***

Another approach is to bin by quantile, 
which ensures that the number of examples in each bucket is equal. 
***Binning by quantile completely removes the need to worry about outliers.***

