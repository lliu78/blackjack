import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout

# aces can only in one value: all '1'/ all '11'
def get_ace_values(num_aces):
    temp_list = []
    for i in range(num_aces):
        temp_list.append([1,11])
    sum_array = np.zeros((2, len(temp_list)))
    for i in range(len(temp_list)):
        sum_array[0, i] = 1
        sum_array[1, i] = 11
    return [int(s) for s in np.sum(sum_array, axis=1)]

# Make a deck
def make_decks(num_decks, card_types):
    new_deck = []
    for i in range(num_decks):
        for j in range(4):
            new_deck.extend(card_types)
    random.shuffle(new_deck)
    return new_deck

# Total up value of hand
def total_up(hand):
    aces = 0
    total = 0
    
    for card in hand:
        if card != 'A':
            total += card
        else:
            aces += 1
    
    # Call function ace_values to produce list of possible values for aces in hand
    ace_value_list = get_ace_values(aces)
    final_totals = [i+total for i in ace_value_list if i+total<=21]
    
    if final_totals == []:
        return min(ace_value_list) + total
    else:
        return max(final_totals)
    
# Play a game of blackjack (after the cards are dealt)
def play_game(dealer_card, player_card, blackjack, player_card_results, cards_home, hit_stay):
    action = 0
    # Dealer checks for 21
    if set(dealer_card) == blackjack:
        for player in range(players):
            if set(player_card[player]) != blackjack:
                player_card_results[0,player] = -1
            else:
                player_card_results[0,player] = 0
    else:
        for player in range(players):
            # Players check for 21
            if set(player_card[player]) == blackjack:
                player_card_results[0,player] = 1
            else:
                # Hit randomly, check for busts
                if (hit_stay >= 0.5) and (total_up(player_card[player]) != 21):
                    player_card[player].append(cards_home.pop(0))
                    action = 1
                    live_total.append(total_up(player_card[player]))
                    if total_up(player_card[player]) > 21:
                        player_card_results[0,player] = -1

    # Dealer hits based on the rules
    while total_up(dealer_card) < 17:
        dealer_card.append(cards_home.pop(0))
    # Compare dealer hand to players hand but first check if dealer busted
    if total_up(dealer_card) > 21:
        for player in range(players):
            if player_card_results[0,player] != -1:
                player_card_results[0,player] = 1
    else:
        for player in range(players):
            if total_up(player_card[player]) > total_up(dealer_card):
                if total_up(player_card[player]) <= 21:
                    player_card_results[0,player] = 1
            elif total_up(player_card[player]) == total_up(dealer_card):
                player_card_results[0,player] = 0
            else:
                player_card_results[0,player] = -1
                
    return player_card_results, cards_home, action

stacks = 5000
players = 1
num_decks = 1

card_types = ['A',2,3,4,5,6,7,8,9,10,10,10,10]

dealer_card_feature = []
player_card_feature = []
player_live_total = []
player_live_action = []
player_results = []

for stack in range(stacks):
    blackjack = set(['A',10])
    dealer_cards = make_decks(num_decks, card_types)
    while len(dealer_cards) > 20:
        
        curr_player_results = np.zeros((1,players))
        
        dealer_hand = []
        player_hands = [[] for player in range(players)]
        live_total = []
        live_action = []

        # Deal FIRST card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        # Deal SECOND card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        
        # Record the player's live total after cards are dealt
        live_total.append(total_up(player_hands[player]))
        
        if stack < 2500:
            hit_stay = 1
        else:
            hit_stay = 0
        curr_player_results, dealer_cards, action = play_game(dealer_hand, player_hands, 
                                                              blackjack, curr_player_results, 
                                                              dealer_cards, hit_stay)
        
        # Track features
        dealer_card_feature.append(dealer_hand[0])
        player_card_feature.append(player_hands)
        player_results.append(list(curr_player_results[0]))
        player_live_total.append(live_total)
        player_live_action.append(action)

model_df = pd.DataFrame()
model_df['dealer_card'] = dealer_card_feature
model_df['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
model_df['hit?'] = player_live_action

has_ace = []
for i in player_card_feature:
    if ('A' in i[0][0:2]):
        has_ace.append(1)
    else:
        has_ace.append(0)
model_df['has_ace'] = has_ace

dealer_card_num = []
for i in model_df['dealer_card']:
    if i=='A':
        dealer_card_num.append(11)
    else:
        dealer_card_num.append(i)
model_df['dealer_card_num'] = dealer_card_num

model_df['Y'] = [i[0] for i in player_results]
lose = []
for i in model_df['Y']:
    if i == -1:
        lose.append(1)
    else:
        lose.append(0)
model_df['lose'] = lose

correct = []
for i, val in enumerate(model_df['lose']):
    if val == 1:
        if player_live_action[i] == 1:
            correct.append(0)
        else:
            correct.append(1)
    else:
        if player_live_action[i] == 1:
            correct.append(1)
        else:
            correct.append(0)
			model_df['correct_action'] = correct

sum(pd.DataFrame(player_results)[0].value_counts())


stacks = 5000
players = 1
num_decks = 1

card_types = ['A',2,3,4,5,6,7,8,9,10,10,10,10]

dealer_card_feature = []
player_card_feature = []
player_results = []

for stack in range(stacks):
    blackjack = set(['A',10])
    dealer_cards = make_decks(num_decks, card_types)
    while len(dealer_cards) > 20:
        
        curr_player_results = np.zeros((1,players))
        
        dealer_hand = []
        player_hands = [[] for player in range(players)]

        # Deal FIRST card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))
        # Deal SECOND card
        for player, hand in enumerate(player_hands):
            player_hands[player].append(dealer_cards.pop(0))
        dealer_hand.append(dealer_cards.pop(0))

        # Dealer checks for 21
        if set(dealer_hand) == blackjack:
            for player in range(players):
                if set(player_hands[player]) != blackjack:
                    curr_player_results[0,player] = -1
                else:
                    curr_player_results[0,player] = 0
        else:
            for player in range(players):
                # Players check for 21
                if set(player_hands[player]) == blackjack:
                    curr_player_results[0,player] = 1
                else:
                    # Hit only when we know we will not bust
                    while total_up(player_hands[player]) <= 11:
                        player_hands[player].append(dealer_cards.pop(0))
                        if total_up(player_hands[player]) > 21:
                            curr_player_results[0,player] = -1
                            break
        
        # Dealer hits based on the rules
        while total_up(dealer_hand) < 17:
            dealer_hand.append(dealer_cards.pop(0))
        # Compare dealer hand to players hand but first check if dealer busted
        if total_up(dealer_hand) > 21:
            for player in range(players):
                if curr_player_results[0,player] != -1:
                    curr_player_results[0,player] = 1
        else:
            for player in range(players):
                if total_up(player_hands[player]) > total_up(dealer_hand):
                    if total_up(player_hands[player]) <= 21:
                        curr_player_results[0,player] = 1
                elif total_up(player_hands[player]) == total_up(dealer_hand):
                    curr_player_results[0,player] = 0
                else:
                    curr_player_results[0,player] = -1
        #print('player: ' + str(total_up(player_hands[player])),
        #      'dealer: ' + str(total_up(dealer_hand)),
        #      'result: ' + str(curr_player_results)
        #     )    
        
        # Track features
        dealer_card_feature.append(dealer_hand[0])
        player_card_feature.append(player_hands)
        player_results.append(list(curr_player_results[0]))

model_df_naive = pd.DataFrame()
model_df_naive['dealer_card'] = dealer_card_feature
model_df_naive['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
model_df_naive['Y'] = [i[0] for i in player_results]

lose = []
for i in model_df_naive['Y']:
    if i == -1:
        lose.append(1)
    else:
        lose.append(0)
model_df_naive['lose'] = lose

has_ace = []
for i in player_card_feature:
    if ('A' in i[0][0:2]):
        has_ace.append(1)
    else:
        has_ace.append(0)
model_df_naive['has_ace'] = has_ace

dealer_card_num = []
for i in model_df_naive['dealer_card']:
    if i=='A':
        dealer_card_num.append(11)
    else:
        dealer_card_num.append(i)
		model_df_naive['dealer_card_num'] = dealer_card_num

# Train a neural net to play blackjack

# Set up variables for neural net
feature_list = [i for i in model_df.columns if i not in ['dealer_card',
                                                         'Y','lose',
                                                         'correct_action']]
print(feature_list)
train_X = np.array(model_df[feature_list])
train_Y = np.array(model_df['correct_action']).reshape(-1,1)

"""
plt.figure()
plt.scatter(train_X,train_Y)
plt.xlabel('train-X')
plt.ylabel('train-Y')
plt.title('Input data')
plt.show()
"""

# Set up a neural net with 5 layers
model = Sequential()
model.add(Dense(8))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(4))
model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01),activity_regularizer=regularizers.l1(0.001)))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_Y, epochs=10, batch_size=64, verbose=1)

pred_Y_train = model.predict(train_X)
actuals = train_Y[:,-1]


# Plot ROC Curve

fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_train)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10,8))
plt.plot(fpr, tpr, label = ('ROC AUC = %0.3f' % roc_auc))

plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
ax.set_xlabel("False Positive Rate",fontsize=16)
ax.set_ylabel("True Positive Rate",fontsize=16)
plt.setp(ax.get_legend().get_texts(), fontsize=16)

plt.savefig(fname='roc_curve_blackjack', dpi=150)
plt.show()


# Given the relevant inputs, the function below uses the neural net to make a prediction
# and then based on that prediction, decides whether to hit or stay

def model_decision(model, player_sum, has_ace, dealer_card_num):
    input_array = np.array([player_sum, 0, has_ace, dealer_card_num]).reshape(1,-1)
    predict_correct = model.predict(input_array)
    if predict_correct >= 0.52:
        return 1
    else:
	return 0

print('Random: ' + str(round(model_df[model_df['Y']==1].shape[0]/model_df.shape[0], 4)))
print('Random: ' + str(round(model_df_naive[model_df_naive['Y']==1].shape[0]/model_df_naive.shape[0], 4)))

print('Total hit frequency: ' +\
str(round(model_df_naive[model_df_naive['hit?']==1].shape[0]/np.sum(model_df_naive.shape[0]), 4)))
