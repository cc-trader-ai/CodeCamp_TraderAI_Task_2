"""
Created on 19.11.2017

@author: rmueller
"""
from collections import deque
import datetime as dt

import math
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
from predicting.predictor.reference.nn_binary_predictor import StockANnBinaryPredictor, StockBNnBinaryPredictor
import numpy as np

TEAM_NAME = "team_blue"

MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR = TEAM_NAME + '_dql_trader_perfect'
MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR = TEAM_NAME + '_dql_trader_perfect_nn_binary'
MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR = TEAM_NAME + '_dql_trader_nn_binary'


class TeamBlueDqlTrader(ITrader):
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
        self.state_size = 5
        self.action_size = 4
        self.hidden_size = 5

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
            self.model.add(Dense(self.action_size, activation='tanh'))
            logger.info(f"DQL Trader: Created new untrained model")
        assert self.model is not None
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        self.portfolio_value_prev = 10000.0;
        self.pct_a_prev = 0;
        self.pct_b_prev = 0;
        self.pct_cash_prev = 1;
        self.diff_a_prev = 0;
        self.diff_b_prev = 0;

        self.loop_value = 0
        self.idx_prev = 0;
        self.y_prev = np.array([[0, 0, 0, 0]])

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
        s_a_current = stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_A)
        s_b_current = stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_B)

        pct_a = round((s_a_current * portfolio.get_amount(CompanyEnum.COMPANY_A))/portfolio.total_value(stock_market_data.get_most_recent_trade_day(), stock_market_data), 2)
        pct_b = round((s_b_current * portfolio.get_amount(CompanyEnum.COMPANY_B))/portfolio.total_value(stock_market_data.get_most_recent_trade_day(), stock_market_data), 2)
        pct_cash = 1 - pct_a - pct_b;


        s_a_next = self.stock_a_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_A]);
        s_b_next = self.stock_b_predictor.doPredict(stock_market_data[CompanyEnum.COMPANY_B]);

        pct_a_diff = round((s_a_next - s_a_current) / s_a_current, 2);
        pct_b_diff = round((s_b_next - s_b_current) / s_b_current, 2);

        portfolio_value = portfolio.total_value(stock_market_data.get_most_recent_trade_day(), stock_market_data);
        portfolio_diff = round((self.portfolio_value_prev - portfolio_value) / self.portfolio_value_prev, 2);




        #r = -1;
        #if (portfolio_diff < 0):
        #    r = 1;
        #elif (portfolio_diff == 0):
        #    r = 0;

        if (portfolio_diff < 0):
            self.y_prev[0][self.idx_prev] = 1;
        elif (portfolio_diff >= 0):
            self.y_prev[0][self.idx_prev] = -1;




        # TODO: Store experience and train the neural network only if doTrade was called before at least once
        if (self.loop_value > 0):
            self.model.fit(np.array([[self.pct_a_prev, self.pct_b_prev, self.pct_cash_prev, self.diff_a_prev, self.diff_b_prev]]), self.y_prev, epochs=1, batch_size=1)

        # TODO: Create actions for current state and decrease epsilon for fewer random actions

        res = self.model.predict(np.array([[pct_a, pct_b, pct_cash, pct_a_diff, pct_b_diff]])) # [0, 0, 1, 0]
        idx = np.argmax(res);

        ret = OrderList()


        if (idx == 0):
            ret.buy(CompanyEnum.COMPANY_A, math.floor(portfolio.cash / stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_A)))
        elif (idx == 1):
            ret.buy(CompanyEnum.COMPANY_B, math.floor(portfolio.cash / stock_market_data.get_most_recent_price(CompanyEnum.COMPANY_B)))
        elif (idx == 2):
            ret.sell(CompanyEnum.COMPANY_A, portfolio.get_amount(CompanyEnum.COMPANY_A))     
        else:
            ret.sell(CompanyEnum.COMPANY_B, portfolio.get_amount(CompanyEnum.COMPANY_B))



        # TODO: Save created state, actions and portfolio value for the next call of doTrade
        self.portfolio_value_prev = portfolio_value;
        self.pct_a_prev = pct_a;
        self.pct_b_prev = pct_b;
        self.pct_cash_prev = pct_cash;
        self.diff_a_prev = pct_a_diff;
        self.diff_b_prev = pct_a_diff;
        self.idx_prev = idx;
        self.y_prev = res;

        self.loop_value = self.loop_value + 1;

        return ret

    def save_trained_model(self):
        """
        Save the trained neural network under a fixed name specific for this trader.
        """
        save_keras_sequential(self.model, self.RELATIVE_DATA_DIRECTORY, self.network_filename)


# This method retrains the trader from scratch using training data from PERIOD_1 and test data from PERIOD_2
EPISODES = 50
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
    trader = TeamBlueDqlTrader(StockANnBinaryPredictor(), StockBNnBinaryPredictor(), False, True, MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR)

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
        # trader_test = TeamBlueDqlTrader(PerfectPredictor(CompanyEnum.COMPANY_A), PerfectPredictor(CompanyEnum.COMPANY_B), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR)
        # trader_test = TeamBlueDqlTrader(StockANnPerfectBinaryPredictor(), StockBNnPerfectBinaryPredictor(), True, False, MODEL_FILENAME_DQLTRADER_PERFECT_NN_BINARY_PREDICTOR)
        trader_test = TeamBlueDqlTrader(StockANnBinaryPredictor(), StockBNnBinaryPredictor(), True, False, MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR)
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
