"""
Created on 19.11.2017

@author: rmueller
"""
from collections import deque
import datetime as dt

from definitions import PERIOD_1, PERIOD_2
from evaluating.portfolio_evaluator import PortfolioEvaluator
from model.Portfolio import Portfolio
from model.StockMarketData import StockMarketData
from model.IPredictor import IPredictor
from model.ITrader import ITrader
from model.Order import OrderList
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from model.Order import CompanyEnum
from utils import save_keras_sequential, load_keras_sequential, read_stock_market_data
from logger import logger
from predicting.predictor.reference.perfect_predictor import PerfectPredictor
# from predicting.predictor.reference.nn_binary_predictor import StockANnBinaryPredictor, StockBNnBinaryPredictor
import numpy as np
from random import randint, uniform

TEAM_NAME = "team_black"

MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR = TEAM_NAME + '_dql_trader_perfect'
MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR = TEAM_NAME + '_dql_trader_perfect_nn_binary'
MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR = TEAM_NAME + '_dql_trader_nn_binary'


class TeamBlackDqlTrader(ITrader):
    """
    Implementation of ITrader based on reinforced Q-learning (RQL).
    """
    RELATIVE_DATA_DIRECTORY = 'trading/trader/' + TEAM_NAME + '/' + TEAM_NAME + '_dql_trader_data'

    def __init__(self, stock_a_predictor: IPredictor, stock_b_predictor: IPredictor,
                 load_trained_model: bool=True,
                 train_while_trading: bool=False, network_filename: str=MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR):
        """
        Constructor
        Args:
            stock_a_predictor: Predictor for stock A
            stock_b_predictor: Predictor for stock B
            load_trained_model: Flag to trigger loading an already trained neural network
            train_while_trading: Flag to trigger on-the-fly training while trading
        """
        # Save predictors, training mode and name
        assert stock_a_predictor is not None and stock_b_predictor is not None
        self.stock_a_predictor = stock_a_predictor
        self.stock_b_predictor = stock_b_predictor
        self.train_while_trading = train_while_trading
        self.network_filename = network_filename

        # Parameters for neural network
        self.state_size = 5 # war 2
        self.action_size = 3 # war 10
        self.hidden_size = 20 # war 50

        # Parameters for deep Q-learning
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.min_size_of_memory_before_training = 1000  # should be way bigger than batch_size, but smaller than memory
        self.memory = deque(maxlen=2000)

        # Attributes necessary to remember our last actions and fill our memory with experiences
        self.last_state = None

        # Create main model, either as trained model (from file) or as untrained model (from scratch)
        self.model = None
        if load_trained_model:
            logger.debug(f"DQL Trader: Try to load trained model")
            self.model = load_keras_sequential(self.RELATIVE_DATA_DIRECTORY, self.network_filename)
        if self.model is None:  # loading failed or we didn't want to use a trained model
            self.model = Sequential()
            self.model.add(Dense(self.hidden_size * 2, input_dim=self.state_size, activation='relu'))
            self.model.add(Dense(self.hidden_size, activation='relu'))
            self.model.add(Dense(self.action_size, activation='linear'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.stored_action = -1
        self.stored_portfolio_value = 0.0
        self.np_previous_status = np.array([])
        self.count = [0, 0, 0]

    def doTrade(self, portfolio: Portfolio, current_portfolio_value: float,
                stock_market_data: StockMarketData) -> OrderList:
        """
        Generate action to be taken on the "stock market"
    
        Args:
          portfolio : current Portfolio of this trader
          current_portfolio_value : value of Portfolio at given moment
          stock_market_data : StockMarketData for evaluation

        Returns:
          A OrderList instance, may be empty never None
        """
        # TODO: Build and store current state object

        ## cash %, portfolio a value %, portfolio b %, pred a, pred b

        account_value = portfolio.cash + current_portfolio_value
        a_value = portfolio.get_amount(CompanyEnum.COMPANY_A) * stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_A)
        b_value = portfolio.get_amount(CompanyEnum.COMPANY_B) * stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_B)

        a_value_percent = a_value / account_value
        b_value_percent = b_value / account_value
        cash_percent = portfolio.cash / account_value

        pred_a_value = self.stock_a_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_A])
        pred_b_value = self.stock_b_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_B])

        stock_a_value = stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_A)
        stock_b_value = stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_B)

        increase_a = (pred_a_value - stock_a_value) / stock_a_value
        increase_b = (pred_b_value - stock_b_value) / stock_b_value

        current_status = [[cash_percent, a_value_percent, b_value_percent, increase_a, increase_b]]

        np_current_status = np.array(current_status)

        # TODO: Store experience and train the neural network only if doTrade was called before at least once

        ## calc rewards = was cash + portfolio - (old cash + old portfolio), map auf 1 0 -1, now simple:
        if self.stored_action >= 0:

            if current_portfolio_value > self.stored_portfolio_value:
                reward = 1.0
            elif current_portfolio_value < self.stored_portfolio_value:
                reward = -1.0
            else:
                reward = 0

            reward_array = [0, 0, 0]

            reward_array[self.stored_action] = reward

            np_reward_array = np.array([reward_array])

            ## train again

            # print(np_reward_array)

            self.model.fit(self.np_previous_status, np_reward_array, epochs=1, batch_size=1, verbose=0)

        # TODO: Create actions for current state and decrease epsilon for fewer random actions

        action = -1

        random_value = uniform(0.0, 1.0)
        if random_value > self.epsilon:
            action = randint(0, 2)
        else:
            pred = self.model.predict(np_current_status)
            prediction = pred[0]

            if len(prediction) != 3:
                print("komische prediction")

            action = 2
            max = prediction[action]

            for i in range(3):
                if prediction[i] > max:
                    action = i


        self.count[action] = self.count[action] + 1
        # print("actions")
        # print(self.count)

        self.epsilon = self.epsilon * self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

        order_list = OrderList()

        if action < 0 or action > 2:
            print("komische action")
            print(action)
        else:

            stock_a = portfolio.get_amount(CompanyEnum.COMPANY_A)
            stock_b = portfolio.get_amount(CompanyEnum.COMPANY_B)

            if action == 0:
                # sell a
                if stock_a > 0:
                    order_list.sell(CompanyEnum.COMPANY_A, stock_a)
                # buy b
                count_b = portfolio.cash / stock_b_value
                if count_b > 0:
                    order_list.buy(CompanyEnum.COMPANY_B, int(count_b))
            if action == 1:
                # sell b
                if stock_b > 0:
                    order_list.sell(CompanyEnum.COMPANY_B, stock_b)
                # boy a
                count_a = portfolio.cash / stock_a_value
                if count_a > 0:
                    order_list.buy(CompanyEnum.COMPANY_A, int(count_a))

        # TODO: Save created state, actions and portfolio value for the next call of

        self.stored_action = action
        self.stored_portfolio_value = current_portfolio_value
        self.np_previous_status = np_current_status

        ## save current state as laststate, current portfolio and cash

        return order_list

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this trader.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.network_filename)


# This method retrains the trader from scratch using training data from PERIOD_1 and test data from PERIOD_2
EPISODES = 10
if __name__ == "__main__":
    # Read the training data
    training_data = read_stock_market_data([CompanyEnum.COMPANY_A, CompanyEnum.COMPANY_B], [PERIOD_1])
    test_data = read_stock_market_data([CompanyEnum.COMPANY_A, CompanyEnum.COMPANY_B], [PERIOD_1, PERIOD_2])
    start_training_day, final_training_day = dt.date(2009, 1, 2), dt.date(2011, 12, 29)
    start_test_day, final_test_day = dt.date(2012, 1, 3), dt.date(2015, 12, 30)

    # Define initial portfolio
    name = 'DQL trader portfolio'
    portfolio = Portfolio(10000.0, [], name)

    # Initialize trader: use perfect predictors, don't use an already trained model, but learn while trading
    # trader = DqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), False, True, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)
    # trader = DqlTrader(StockANnPerfectBinaryPredictor(), StockBNnPerfectBinaryPredictor(), False, True, MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR)
    trader = TeamBlackDqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), False, True, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)

    # Start evaluation and train correspondingly; don't display the results in a plot but display final portfolio value
    evaluator = PortfolioEvaluator([trader], False)
    final_values_training, final_values_test = [], []
    for i in range(EPISODES):
        logger.info(f"DQL Trader: Starting training episode {i}")
        all_portfolios_over_time = evaluator.inspect_over_time(training_data, [portfolio],
                                                               date_offset=start_training_day)
        portfolio_over_time = all_portfolios_over_time[name]
        final_values_training.append(
            portfolio_over_time[final_training_day].total_value(final_training_day, training_data))
        trader.save_trained_model()

        # Evaluation over training and visualization
        # trader_test = TeamBlackDqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)
        # trader_test = TeamBlackDqlTrader(StockANnPerfectBinaryPredictor(), StockBNnPerfectBinaryPredictor(), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR)
        trader_test = TeamBlackDqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)
        evaluator_test = PortfolioEvaluator([trader_test], False)
        all_portfolios_over_time = evaluator_test.inspect_over_time(test_data, [portfolio], date_offset=start_test_day)
        portfolio_over_time = all_portfolios_over_time[name]
        final_values_test.append(portfolio_over_time[final_test_day].total_value(final_test_day, test_data))
        logger.info(f"DQL Trader: Finished training episode {i}, "
                    f"final portfolio value training {final_values_training[-1]} vs. "
                    f"final portfolio value test {final_values_test[-1]}")

    from matplotlib import pyplot as plt

    plt.figure()
    plt.plot(final_values_training, color="black")
    plt.plot(final_values_test, color="green")
    plt.title('final portfolio value training vs. final portfolio value test')
    plt.ylabel('final portfolio value')
    plt.xlabel('episode')
    plt.show()
