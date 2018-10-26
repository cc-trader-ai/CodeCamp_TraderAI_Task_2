"""
Created on 19.11.2017

@author: rmueller
"""
from collections import deque
from enum import Enum
import datetime as dt

import numpy as np

from definitions import PERIOD_1, PERIOD_2
from evaluating.portfolio_evaluator import PortfolioEvaluator
from model.Portfolio import Portfolio
from model.StockMarketData import StockMarketData
from model.Order import SharesOfCompany
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
from predicting.predictor.reference.nn_binary_predictor import StockANnBinaryPredictor, StockBNnBinaryPredictor
from predicting.predictor.reference.nn_perfect_binary_predictor import StockANnPerfectBinaryPredictor, StockBNnPerfectBinaryPredictor

TEAM_NAME = "team_red"

MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR = TEAM_NAME + '_dql_trader_perfect'
MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR = TEAM_NAME + '_dql_trader_perfect_nn_binary'
MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR = TEAM_NAME + '_dql_trader_nn_binary'

class TeamRedDqlTrader(ITrader):
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
        self.state_size = 2
        self.action_size = 4
        self.hidden_size = 50

        # Parameters for deep Q-learning
        self.gamma = 0.1
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_epochs = 10
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

        # DONE: Build and store current state object
        
        current_price_a = stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_A)
        current_price_b = stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_B)
        
        predicted_price_a = self.stock_a_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_A])
        predicted_price_b = self.stock_b_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_B])
        
        portfolio_value = portfolio.total_value(
            stock_market_data.get_most_recent_trade_day(), 
            stock_market_data)

        state = State(portfolio_value,
            current_price_a, predicted_price_a, 
            current_price_b, predicted_price_b)

        ## DONE: Train NN with previous action and reward
        if self.last_state:
            self.train_model(state, self.last_state)

        # TODO: Store experience and train the neural network only if doTrade was called before at least once

        # DONE: Create actions for current state and decrease epsilon for fewer random actions
        action = self.decide_action(state)
        orders = self.create_orders(action, portfolio, stock_market_data)

        # DONE: Save created state, actions and portfolio value for the next call of doTrade
        self.last_state = state

        return orders

    def train_model(self, state, last_state):
        prediction_for_current_state = self.predict_actions(state)
        reward = self.calculate_reward(state, last_state)

        model_input = np.array([[
            last_state.get_delta_price_a(), 
            last_state.get_delta_price_b()
        ]])

        expected_predictions = [
            last_state.predicted_actions[TradingAction.SELL_A_AND_B.value],
            last_state.predicted_actions[TradingAction.BUY_A_AND_SELL_B.value],
            last_state.predicted_actions[TradingAction.BUY_B_AND_SELL_A.value],
            last_state.predicted_actions[TradingAction.BUY_A_AND_B.value]
        ]

        last_action_index = last_state.best_action.value
        expected_predictions[last_action_index] = reward + self.gamma * prediction_for_current_state[last_action_index]
        expected_output = np.array([expected_predictions])

        self.model.fit(model_input, expected_output, epochs=self.train_epochs, batch_size=self.batch_size)

    def calculate_reward(self, state, last_state):
        return calculate_delta(state.portfolio_value, last_state.portfolio_value)

    def decide_action(self, state):
        predicted_actions = self.predict_actions(state)
        best_action_index = predicted_actions.index(max(predicted_actions))
        best_action = TradingAction(best_action_index)
        state.predicted_actions = predicted_actions
        state.best_action = best_action
        return best_action

    def create_orders(self, action, portfolio, stock_market_data):
        orders = OrderList()
        if action == TradingAction.SELL_A_AND_B:
            self.sell_shares(CompanyEnum.COMPANY_A, orders, portfolio, stock_market_data)
            self.sell_shares(CompanyEnum.COMPANY_B, orders, portfolio, stock_market_data)
        elif action == TradingAction.BUY_A_AND_SELL_B:
            updated_cash = self.sell_shares(CompanyEnum.COMPANY_B, orders, portfolio, stock_market_data)
            self.buy_shares(CompanyEnum.COMPANY_A, orders, updated_cash, stock_market_data)
        elif action == TradingAction.BUY_B_AND_SELL_A:
            updated_cash = self.sell_shares(CompanyEnum.COMPANY_A, orders, portfolio, stock_market_data)
            self.buy_shares(CompanyEnum.COMPANY_B, orders, updated_cash, stock_market_data)
        elif action == TradingAction.BUY_A_AND_B:
            self.buy_shares(CompanyEnum.COMPANY_A, orders, portfolio.cash / 2, stock_market_data)
            self.buy_shares(CompanyEnum.COMPANY_B, orders, portfolio.cash / 2, stock_market_data)
        return orders

    def sell_shares(self, company, orders, portfolio, stock_market_data):
        updated_cash = portfolio.cash
        company_data = stock_market_data[company]
        price = company_data.get_last()[-1]
        shares = find_shares_of_company(company, portfolio.shares)
        if shares and shares.amount > 0:
            orders.sell(company, shares.amount)
            updated_cash = portfolio.cash + (shares.amount * price)
        return updated_cash

    def buy_shares(self, company, orders, cash, stock_market_data):
        company_data = stock_market_data[company]
        price = company_data.get_last()[-1]
        amount_to_buy = int(cash // price)
        if amount_to_buy > 0:
            orders.buy(company, amount_to_buy)

    def predict_actions(self, state):
        model_input = np.array([[
            state.get_delta_price_a(), state.get_delta_price_b()
        ]])
        return self.model.predict(model_input)[0].tolist()

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this trader.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.network_filename)

class State():
    def __init__(self, portfolio_value, current_price_a, predicted_price_a, current_price_b, predicted_price_b):
        self.portfolio_value = portfolio_value
        self.current_price_a = current_price_a
        self.current_price_b = current_price_b
        self.predicted_price_a = predicted_price_a
        self.predicted_price_b = predicted_price_b
        self.predicted_actions = None
        self.best_action = None

    def get_delta_price_a(self):
        return calculate_delta(self.predicted_price_a, self.current_price_a)

    def get_delta_price_b(self):
        return calculate_delta(self.predicted_price_b, self.current_price_b)

class TradingAction(Enum):
    SELL_A_AND_B = 0
    BUY_A_AND_SELL_B = 1
    BUY_B_AND_SELL_A = 2
    BUY_A_AND_B = 3

def calculate_delta(a, b):
    return (a / b) - 1

def find_shares_of_company(company_enum: CompanyEnum, shares: list) -> SharesOfCompany:
    for shares_of_company in shares:
        if isinstance(shares_of_company, SharesOfCompany) and shares_of_company.company_enum == company_enum:
            return shares_of_company

    return None

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
    trader = TeamRedDqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), False, True, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)
    # trader = TeamRedDqlTrader(StockANnPerfectBinaryPredictor(), StockBNnPerfectBinaryPredictor(), False, True, MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR)
    # trader = TeamRedDqlTrader(StockANnBinaryPredictor(), StockBNnBinaryPredictor(), False, True, MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR)

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
        trader_test = TeamRedDqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)
        # trader_test = TeamRedDqlTrader(StockANnPerfectBinaryPredictor(), StockBNnPerfectBinaryPredictor(), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR)
        # trader_test = TeamRedDqlTrader(StockANnBinaryPredictor(), StockBNnBinaryPredictor(), True, False, MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR)
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
