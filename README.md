# blackjack // CIS667
Playing blackjack with machine learning

Module requirement:
  import matplotlib.pyplot as plt
  import seaborn as sns
  import sklearn.metrics as metrics
  from keras import regularizers
  from keras.models import Sequential
  from keras.layers import Dense, LSTM, Flatten, Dropout

functions:
1) get_aces_values
  According to the rule of blackjack, 'A' can be represented as 1 or 11. 
  Therefore, it's different with other cards that we have to design a list of set 
    to represent different situations.
    
2) make_desk
  This function simulates different situations of the desk, using a for-loop to deal with 1 to 5 packs of card.
  
3) total_up
  This function returns the total value of hand cards.
  

Model from Yunxia Zhao:
  6 layers NN
  loss='binary_crossentropy', optimizer='sgd'
  
Model from Le Liu:
  5 layers NN
  loss='binary_crossentropy', optimizer='adam'
  
