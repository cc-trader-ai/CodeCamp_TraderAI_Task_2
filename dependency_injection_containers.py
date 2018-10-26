'''
Created on 09.11.2017

@author: jtymoszuk
'''
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from model.CompanyEnum import CompanyEnum
from predicting.predictor.reference.random_predictor import RandomPredictor
from predicting.predictor.reference.perfect_predictor import PerfectPredictor
from predicting.predictor.reference.nn_binary_predictor import StockANnBinaryPredictor, \
    StockBNnBinaryPredictor
from predicting.predictor.reference.nn_perfect_binary_predictor import StockANnPerfectBinaryPredictor, \
    StockBNnPerfectBinaryPredictor
from trading.trader.reference.simple_trader import SimpleTrader
from trading.trader.reference.buy_and_hold_trader import BuyAndHoldTrader
from predicting.predictor.team_blue.team_blue_predictor import TeamBlueStockAPredictor, \
    TeamBlueStockBPredictor
from predicting.predictor.team_green.team_green_predictor import TeamGreenStockAPredictor, \
    TeamGreenStockBPredictor
from predicting.predictor.team_black.team_black_predictor import TeamBlackStockBPredictor, \
    TeamBlackStockAPredictor
from predicting.predictor.team_red.team_red_predictor import TeamRedStockBPredictor, \
    TeamRedStockAPredictor
from trading.trader.team_blue.team_blue_simple_trader import TeamBlueSimpleTrader
from trading.trader.team_green.team_green_simple_trader import TeamGreenSimpleTrader
from trading.trader.team_black.team_black_simple_trader import TeamBlackSimpleTrader
from trading.trader.team_red.team_red_simple_trader import TeamRedSimpleTrader
from trading.trader.team_blue.team_blue_dql_trader import TeamBlueDqlTrader
from trading.trader.team_blue import team_blue_dql_trader
from trading.trader.team_green.team_green_dql_trader import TeamGreenDqlTrader
from trading.trader.team_green import team_green_dql_trader
from trading.trader.team_black.team_black_dql_trader import TeamBlackDqlTrader
from trading.trader.team_black import team_black_dql_trader
from trading.trader.team_red.team_red_dql_trader import TeamRedDqlTrader
from trading.trader.team_red import team_red_dql_trader


class Predictors(containers.DeclarativeContainer):
    """IoC container of predictor providers."""
 
    """ Random predictor delivering value of last share +- Random[0,1]"""
    RandomPredictor = providers.Factory(RandomPredictor)
    
    """ Perfect predictors knowing future"""
    # Task 0 and Task 2
    PerfectPredictor_stock_a = providers.Factory(PerfectPredictor, CompanyEnum.COMPANY_A)    
    PerfectPredictor_stock_b = providers.Factory(PerfectPredictor, CompanyEnum.COMPANY_B)
    
    """ Predictors based on neural networks, trying to estimate next future value of share"""
    # Currently not in use
    # StockANnValuePredictor = providers.Factory(StockANnValuePredictor)
    # StockBNnValuePredictor = providers.Factory(StockBNnValuePredictor)
    
    """ Binary predictors based on neural networks trained only with historical data (till 2011). 
    Predictors are estimating only if next value will go up or down - returned result is then value of last share +- constant value """
    StockANnBinaryPredictor = providers.Factory(StockANnBinaryPredictor)
    StockBNnBinaryPredictor = providers.Factory(StockBNnBinaryPredictor)
    
    """ Predictors based on neural networks trained only with all available data (till 2017).
    Predictors are estimating only if next value will go up or down - returned result is then value of last share +- constant value """
    StockANnPerfectBinaryPredictor = providers.Factory(StockANnPerfectBinaryPredictor)
    StockBNnPerfectBinaryPredictor = providers.Factory(StockBNnPerfectBinaryPredictor)
    
    """Initial empty Predictors for training purposes"""
    """Team Blue Predictors"""
    # Task 1
    TeamBlueStockAPredictor = providers.Factory(TeamBlueStockAPredictor)
    TeamBlueStockBPredictor = providers.Factory(TeamBlueStockBPredictor)
    
    """Team Green Predictors"""
    # Task 1
    TeamGreenStockAPredictor = providers.Factory(TeamGreenStockAPredictor)
    TeamGreenStockBPredictor = providers.Factory(TeamGreenStockBPredictor)
        
    """Team Black Predictors"""
    # Task 1
    TeamBlackStockAPredictor = providers.Factory(TeamBlackStockAPredictor)
    TeamBlackStockBPredictor = providers.Factory(TeamBlackStockBPredictor)
            
    """Team Red Predictors"""
    # Task 1
    TeamRedStockAPredictor = providers.Factory(TeamRedStockAPredictor)
    TeamRedStockBPredictor = providers.Factory(TeamRedStockBPredictor)

 
class Traders(containers.DeclarativeContainer):
    """IoC container of trader providers."""
    
    """Simple Trader"""
    SimpleTrader_with_perfect_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    SimpleTrader_with_random_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.RandomPredictor,
        stock_b_predictor=Predictors.RandomPredictor
        )
    
    SimpleTrader_with_nn_binary_perfect_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.StockANnPerfectBinaryPredictor,
        stock_b_predictor=Predictors.StockBNnPerfectBinaryPredictor
        )
    
    SimpleTrader_with_nn_binary_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.StockANnBinaryPredictor,
        stock_b_predictor=Predictors.StockBNnBinaryPredictor
        )

    """Buy and Hold Trader"""
    BuyAndHoldTrader = providers.Factory(
        BuyAndHoldTrader
        )
  
    """Traders for training purposes"""
    """Team Blue Traders"""
    # Task 0
    TeamBlueSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamBlueSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    # Task 1
    SimpleTrader_with_team_blue_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamBlueStockAPredictor,
        stock_b_predictor=Predictors.TeamBlueStockBPredictor
        )
    
    # Task 2
    TeamBlueDqlTrader_with_perfect_prediction = providers.Factory(
        TeamBlueDqlTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b,
        network_filename=team_blue_dql_trader.MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR
        )
    
    """Team Green Traders"""
    # Task 0
    TeamGreenSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamGreenSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )

    # Task 1
    SimpleTrader_with_team_green_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamGreenStockAPredictor,
        stock_b_predictor=Predictors.TeamGreenStockBPredictor
        )
    
    # Task 2
    TeamGreenDqlTrader_with_perfect_prediction = providers.Factory(
        TeamGreenDqlTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b,
        network_filename=team_green_dql_trader.MODEL_FILENAME_DQLTRADER_PERFECT_PREDICTOR
        )
        
    """Team Black Traders"""
    # Task 0
    TeamBlackSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamBlackSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    # Task 1
    SimpleTrader_with_team_black_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamBlackStockAPredictor,
        stock_b_predictor=Predictors.TeamBlackStockBPredictor
        )
    
    # Task 2
    TeamBlackDqlTrader_with_perfect_prediction = providers.Factory(
        TeamBlackDqlTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b,
        network_filename=team_black_dql_trader.MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR
        )
            
    """Team Red Traders"""
    # Task 0
    TeamRedSimpleTrader_with_perfect_prediction = providers.Factory(
        TeamRedSimpleTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b
        )
    
    # Task 1
    SimpleTrader_with_team_red_prediction = providers.Factory(
        SimpleTrader,
        stock_a_predictor=Predictors.TeamRedStockAPredictor,
        stock_b_predictor=Predictors.TeamRedStockBPredictor
        )
    
    # Task 2
    TeamRedDqlTrader_with_perfect_prediction = providers.Factory(
        TeamRedDqlTrader,
        stock_a_predictor=Predictors.PerfectPredictor_stock_a,
        stock_b_predictor=Predictors.PerfectPredictor_stock_b,
        network_filename=team_red_dql_trader.MODEL_FILENAME_DQLTRADER_NN_BINARY_PREDICTOR
        )
