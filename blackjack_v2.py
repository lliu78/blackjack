import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, LSTM, Flatten, Dropout

def get_ace_values(num_aces):
    '''
    ace_num: int, the number of card "Ace" in hand
    return: list, all possible value could be constructed by this number of Ace
    '''
    permutations = []
    for i in range(num_aces + 1):
        ace_to_one = num_aces - i;
        ace_to_eleven = i;
        permutations.append(ace_to_eleven * 11 + ace_to_one);
    return permutations

def make_decks(deck_num, deck_cards):
  '''
  deck_num: int, the numbers of decks included in game
  deck_cards: list, cards points included in each deck
  return: list, a shuffled stack of cards with deck_num decks.
  reference: https://github.com/yiuhyuk/blackjack
  '''
  shuffled_deck = []
  for i in range(deck_num):
    # each deck contain four different color cards
    for j in range(4):
      shuffled_deck.extend(deck_cards)
  random.shuffle(shuffled_deck)
  return shuffled_deck

def total_up(hand):
    '''
    hand: list, a list of cars point in players'/dealer's hand
    return: int, the best point could get outof cards in hand
    '''
    non_ace_count = 0
    ace_count = 0
    for card in hand:
        if card == 'A':
            ace_count += 1
        else:
            non_ace_count += card

    ace_values = get_ace_values(ace_count)
    total_points = []

    for a in ace_values:
        if a + non_ace_count < 21:
            total_points.append(a + non_ace_count)

    if len(total_points) == 0:
        return min(ace_values) + non_ace_count
    else:
        return max(total_points)


# Experimental results:: Le Liu
# reference: https://github.com/yiuhyuk/blackjack
for i in range(5):
    stacks = 5000
    players_num = 1
    decks_num = i+1

    card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    dealer_card_feature = []
    player_card_feature = []
    player_live_actions = []
    player_results = []

    print("Start Game with")

    for stack in range(stacks):
        blackjack = set(['A', 10])
        dealer_cards = make_decks(decks_num, card_types)
        while len(dealer_cards) > 20:

            curr_player_results = np.zeros((1, players_num))

            dealer_hand = []
            player_hands = [[] for player in range(players_num)]

            # Deal FIRST card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
            dealer_hand.append(dealer_cards.pop(0))
            # Deal SECOND card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
            dealer_hand.append(dealer_cards.pop(0))

            action = 0

            # Dealer checks for 21
            if set(dealer_hand) == blackjack:
                for player in range(players_num):
                    if set(player_hands[player]) != blackjack:
                        curr_player_results[0, player] = -1
                    else:
                        curr_player_results[0, player] = 0
            else:
                for player in range(players_num):
                    # Players check for 21
                    if set(player_hands[player]) == blackjack:
                        curr_player_results[0, player] = 1
                    else:
                        # Hit only when we know we will not bust
                        while total_up(player_hands[player]) <= 11:
                            action = 1
                            player_hands[player].append(dealer_cards.pop(0))
                            if total_up(player_hands[player]) > 21:
                                curr_player_results[0, player] = -1
                                break

            # Dealer hits based on the rules
            while total_up(dealer_hand) < 17:
                dealer_hand.append(dealer_cards.pop(0))
            # Compare dealer hand to players hand but first check if dealer busted
            if total_up(dealer_hand) > 21:
                for player in range(players_num):
                    if curr_player_results[0, player] != -1:
                        curr_player_results[0, player] = 1
            else:
                for player in range(players_num):
                    if total_up(player_hands[player]) > total_up(dealer_hand):
                        if total_up(player_hands[player]) <= 21:
                            curr_player_results[0, player] = 1
                    elif total_up(player_hands[player]) == total_up(dealer_hand):
                        curr_player_results[0, player] = 0
                    else:
                        curr_player_results[0, player] = -1


            # Track features
            dealer_card_feature.append(dealer_hand[0])
            player_card_feature.append(player_hands)
            player_results.append(list(curr_player_results[0]))
            player_live_actions.append(action)

    model_df_naive = pd.DataFrame()
    model_df_naive['dealer_card'] = dealer_card_feature
    model_df_naive['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
    model_df_naive['hit?'] = player_live_actions
    model_df_naive['Y'] = [i[0] for i in player_results]

    lose = []
    for i in model_df_naive['Y']:
        if i == -1:
            lose.append(1)
        else:
            lose.append(0)
    model_df_naive['lose'] = lose

    correct = []
    for i, val in enumerate(model_df_naive['lose']):
        if val == 1:
            if player_live_actions[i] == 1:
                correct.append(0)
            else:
                correct.append(1)
        else:
            if player_live_actions[i] == 1:
                correct.append(1)
            else:
                correct.append(0)
    model_df_naive['correct_action'] = correct

    has_ace = []
    for i in player_card_feature:
        if ('A' in i[0][0:2]):
            has_ace.append(1)
        else:
            has_ace.append(0)
    model_df_naive['has_ace'] = has_ace

    dealer_card_num = []
    for i in model_df_naive['dealer_card']:
        if i == 'A':
            dealer_card_num.append(11)
        else:
            dealer_card_num.append(i)
    model_df_naive['dealer_card_num'] = dealer_card_num

    # Train a neural net to play blackjack

    # Set up variables for neural net
    feature_list = [i for i in model_df_naive.columns if i not in ['dealer_card',
                                                             'Y', 'lose',
                                                             'correct_action']]
    print(feature_list)
    train_X = np.array(model_df_naive[feature_list])
    train_Y = np.array(model_df_naive['correct_action']).reshape(-1, 1)



    # Set up a neural net with 5 layers
    model = Sequential()
    model.add(Dense(8))
    model.add(Dense(64))
    model.add(Dense(16))
    model.add(Dense(4))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01),
                    activity_regularizer=regularizers.l1(0.001)))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(train_X, train_Y, epochs=100, batch_size=64, verbose=1)
    print(model.summary())
    pred_Y_train = model.predict(train_X)
    actuals = train_Y[:, -1]

    # Plot ROC 

    fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_train)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(fpr, tpr, label=('ROC AUC = %0.3f' % roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize=16)

    plt.savefig(fname='roc_curve_blackjack', dpi=150)
    plt.show()


    # Create a function to decide whether to hit or stay

    def model_decision(model, player_sum, has_ace, dealer_card_num):
        input_array = np.array([player_sum, 0, has_ace, dealer_card_num]).reshape(1, -1)
        predict_correct = model.predict(input_array)
        if predict_correct >= 0.52:
            return 1
        else:
            return 0


    print('Random: ' + str(round(model_df_naive[model_df_naive['Y'] == 1].shape[0] / model_df_naive.shape[0], 4)))

    print('Total hit frequency: ' + \
          str(round(model_df_naive[model_df_naive['hit?'] == 1].shape[0] / np.sum(model_df_naive.shape[0]), 4)))


# Experimental results:: Yunxia Zhao
# reference: https://github.com/yiuhyuk/blackjack
for i in range(5):
    stacks = 5000
    players_num = 1
    decks_num = i + 5

    card_types = ['A', 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]

    dealer_card_feature = []
    player_card_feature = []
    player_live_actions = []
    player_results = []

    print("Start Game with")

    for stack in range(stacks):
        blackjack = set(['A', 10])
        dealer_cards = make_decks(decks_num, card_types)
        while len(dealer_cards) > 20:

            curr_player_results = np.zeros((1, players_num))

            dealer_hand = []
            player_hands = [[] for player in range(players_num)]

            # Deal FIRST card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
            dealer_hand.append(dealer_cards.pop(0))
            # Deal SECOND card
            for player, hand in enumerate(player_hands):
                player_hands[player].append(dealer_cards.pop(0))
            dealer_hand.append(dealer_cards.pop(0))

            action = 0

            # Dealer checks for 21
            if set(dealer_hand) == blackjack:
                for player in range(players_num):
                    if set(player_hands[player]) != blackjack:
                        curr_player_results[0, player] = -1
                    else:
                        curr_player_results[0, player] = 0
            else:
                for player in range(players_num):
                    # Players check for 21
                    if set(player_hands[player]) == blackjack:
                        curr_player_results[0, player] = 1
                    else:
                        # Hit only when we know we will not bust
                        while total_up(player_hands[player]) <= 11:
                            action = 1
                            player_hands[player].append(dealer_cards.pop(0))
                            if total_up(player_hands[player]) > 21:
                                curr_player_results[0, player] = -1
                                break

            # Dealer hits based on the rules
            while total_up(dealer_hand) < 17:
                dealer_hand.append(dealer_cards.pop(0))
            # Compare dealer hand to players hand but first check if dealer busted
            if total_up(dealer_hand) > 21:
                for player in range(players_num):
                    if curr_player_results[0, player] != -1:
                        curr_player_results[0, player] = 1
            else:
                for player in range(players_num):
                    if total_up(player_hands[player]) > total_up(dealer_hand):
                        if total_up(player_hands[player]) <= 21:
                            curr_player_results[0, player] = 1
                    elif total_up(player_hands[player]) == total_up(dealer_hand):
                        curr_player_results[0, player] = 0
                    else:
                        curr_player_results[0, player] = -1


            # Track features
            dealer_card_feature.append(dealer_hand[0])
            player_card_feature.append(player_hands)
            player_results.append(list(curr_player_results[0]))
            player_live_actions.append(action)

    model_df_naive = pd.DataFrame()
    model_df_naive['dealer_card'] = dealer_card_feature
    model_df_naive['player_total_initial'] = [total_up(i[0][0:2]) for i in player_card_feature]
    model_df_naive['hit?'] = player_live_actions
    model_df_naive['Y'] = [i[0] for i in player_results]

    lose = []
    for i in model_df_naive['Y']:
        if i == -1:
            lose.append(1)
        else:
            lose.append(0)
    model_df_naive['lose'] = lose

    correct = []
    for i, val in enumerate(model_df_naive['lose']):
        if val == 1:
            if player_live_actions[i] == 1:
                correct.append(0)
            else:
                correct.append(1)
        else:
            if player_live_actions[i] == 1:
                correct.append(1)
            else:
                correct.append(0)
    model_df_naive['correct_action'] = correct

    has_ace = []
    for i in player_card_feature:
        if ('A' in i[0][0:2]):
            has_ace.append(1)
        else:
            has_ace.append(0)
    model_df_naive['has_ace'] = has_ace

    dealer_card_num = []
    for i in model_df_naive['dealer_card']:
        if i == 'A':
            dealer_card_num.append(11)
        else:
            dealer_card_num.append(i)
    model_df_naive['dealer_card_num'] = dealer_card_num

    # Train a neural net to play blackjack

    # Set up variables for neural net
    feature_list = [i for i in model_df_naive.columns if i not in ['dealer_card',
                                                             'Y', 'lose',
                                                             'correct_action']]
    print(feature_list)
    train_X = np.array(model_df_naive[feature_list])
    train_Y = np.array(model_df_naive['correct_action']).reshape(-1, 1)



    # Set up a neural net with 6 layers
    model = Sequential()
    model.add(Dense(16))
    model.add(Dense(32))
    model.add(Dense(64))
    model.add(Dense(32))
    model.add(Dense(2))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    model.fit(train_X, train_Y, epochs=100, batch_size=32, verbose=1)
    print(model.summary())
    pred_Y_train = model.predict(train_X)
    actuals = train_Y[:, -1]

    # Plot ROC

    fpr, tpr, threshold = metrics.roc_curve(actuals, pred_Y_train)
    roc_auc = metrics.auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.plot(fpr, tpr, label=('ROC AUC = %0.3f' % roc_auc))

    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    ax.set_xlabel("False Positive Rate", fontsize=16)
    ax.set_ylabel("True Positive Rate", fontsize=16)
    plt.setp(ax.get_legend().get_texts(), fontsize=16)

    plt.savefig(fname='roc_curve_blackjack', dpi=150)
    plt.show()


    # Create a function to decide whether to hit or stay

    def model_decision(model, player_sum, has_ace, dealer_card_num):
        input_array = np.array([player_sum, 0, has_ace, dealer_card_num]).reshape(1, -1)
        predict_correct = model.predict(input_array)
        if predict_correct >= 0.52:
            return 1
        else:
            return 0


    print('Random: ' + str(round(model_df_naive[model_df_naive['Y'] == 1].shape[0] / model_df_naive.shape[0], 4)))

    print('Total hit frequency: ' + \
          str(round(model_df_naive[model_df_naive['hit?'] == 1].shape[0] / np.sum(model_df_naive.shape[0]), 4)))
    
