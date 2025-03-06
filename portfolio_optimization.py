"""
Portfolio Optimization Module
=============================

This module provides functions for fetching asset prices, calculating asset returns,
and performing portfolio optimization using the efficient frontier.

Version: 1.0.0
Author: [Marios Theocharous / LifeGoals Financial Services]
Date: 2025-02-13

Functions:
----------
- get_asset_prices_yahoo(symbols): Fetches historical prices of given assets using Yahoo Finance.
- get_asset_returns(prices_df, freq): Computes asset returns based on price data.
- efficient_frontier(returns_df): Calculates the efficient frontier, returning optimal portfolio weights.
- get_dataframe(weights, risks, exp_returns, sharpe_ratios): Creates a structured DataFrame for the efficient frontier.
- get_min_risk_portfolio(df): Identifies the portfolio with the lowest risk.
- get_max_exp_return_portfolio(df): Identifies the portfolio with the highest expected return.
- get_max_sharpe_ratio_portfolio(df): Identifies the portfolio with the best Sharpe ratio.
- efficient_frontier_json(df, min_risk, max_return, max_sharpe): Converts the efficient frontier results into JSON format.

Usage:
------
This module is designed to be used within a Flask web application for portfolio optimization.
It requires dependencies such as pandas, numpy, and optionally yfinance.

"""

# Import Required Packages
import os
import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize
from typing import List, Tuple
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# Defined Functions
def get_asset_prices_yahoo(asset_tickers: list[str]) -> pd.DataFrame:
    """
    Get asset prices for the last 5-years for define asset symbols.

    Returns:
        pd.DataFrame: DataFrame containing prices for assets for the last 5-years.
    """
    if not isinstance(asset_tickers, list):
        raise TypeError("Asset tickers must be a list.")

    # Set start date as the date today minus 5-years
    start_date = datetime.now() - relativedelta(years=5)

    # Download asset close prices from Yahoo finance
    prices_df = yf.download(asset_tickers, start_date)['Close']

    return prices_df


def get_asset_prices(asset_tickers: list[str]) -> pd.DataFrame:
    """
    Given a list of asset tickers, read their prices from JSON files and return as a DataFrame.

    Args:
        asset_tickers (List[str]): List of asset tickers (str).

    Returns:
        pd.DataFrame: DataFrame with dates as index and tickers as columns.
    """
    if not isinstance(asset_tickers, list):
        raise TypeError("Asset tickers must be a list.")

    prices_dict = {}

    # Calculate cutoff date five years ago from current date
    cutoff_date = datetime.now() - relativedelta(years=5)

    for ticker in asset_tickers:
        filename = f"{ticker}.json"

        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data = json.load(file)
                for entry in data:
                    date = entry["MarketPriceDateTime"]
                    price_date = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S")
                    if price_date >= cutoff_date:
                        if date not in prices_dict:
                            prices_dict[date] = {}
                        prices_dict[date][ticker] = entry["UnitPrice"]

    # Convert dictionary to DataFrame
    prices_df = pd.DataFrame.from_dict(prices_dict, orient='index')
    prices_df.index = pd.to_datetime(prices_df.index)
    prices_df.sort_index(inplace=True)

    # Drop rows with NaN values
    prices_df.dropna(inplace=True)

    return prices_df


def get_asset_returns(prices_df: pd.DataFrame, frequency: str = 'daily') -> pd.DataFrame:
    """
    Calculate returns from asset prices based on the specified frequency.

    Args:
        prices_df (pd.DataFrame): DataFrame containing daily asset prices.
        frequency (str): Frequency of returns ('daily', 'weekly', 'monthly').

    Returns:
        pd.DataFrame: DataFrame containing returns.
    """
    if frequency == 'daily':
        returns_df = np.log(prices_df / prices_df.shift(1)) * 100
    elif frequency == 'weekly':
        prices_df = prices_df.resample('W-Fri').last()
        returns_df = prices_df.pct_change() * 100
    elif frequency == 'monthly':
        prices_df = prices_df.resample('M').last()
        returns_df = prices_df.pct_change() * 100
    else:
        raise ValueError("Frequency must be 'daily', 'weekly', or 'monthly'.")

    return returns_df.dropna()


def detect_scaling_factor(df: pd.DataFrame) -> int:
    """
    Detect the scaling factor from the frequency of the DataFrame's index.

    Args:
        df (pd.DataFrame): DataFrame with a DateTime index.

    Returns:
        int: Scaling factor of the returns ('252', '52', '12').

    Raises:
        ValueError: If the frequency cannot be determined within expected ranges.
    """
    # Calculate the difference between consecutive dates
    delta = df.index.to_series().diff().mode()[0]

    # Determine frequency based on the delta
    if pd.Timedelta(days=1) <= delta < pd.Timedelta(days=7):
        return int(252)
    elif pd.Timedelta(days=7) <= delta < pd.Timedelta(days=30):
        return int(52)
    elif pd.Timedelta(days=30) <= delta < pd.Timedelta(days=365):
        return int(12)
    else:
        raise ValueError("Unable to determine scaling factor: delta outside expected ranges.")


def get_mean_returns(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the mean returns for each asset.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns for each asset.

    Returns:
        np.ndarray: Array containing the mean returns for each asset.
    """
    # Calculate mean returns and convert to a NumPy array
    return returns_df.mean().to_numpy()


def get_covar_matrix(returns_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate the covariance matrix of the asset returns.

    Args:
        returns_df (pd.DataFrame): DataFrame containing returns for each asset.

    Returns:
        np.ndarray: Covariance matrix of the returns.
    """
    # Calculate covariance matrix and convert to a NumPy array
    return returns_df.cov().to_numpy()


def portfolio_exp_return(weights: np.ndarray, mean_returns: np.ndarray, scaling_factor: int) -> float:
    """
    Calculate the expected annual return of the portfolio.

    Args:
        weights (np.ndarray): Array of portfolio weights for each asset.
        mean_returns (np.ndarray): Array of mean returns for each asset.
        scaling_factor (int): Scaling factor of the mean return.

    Returns:
        float: Expected annual return of the portfolio.
    """
    # Calculate expected return (scaled to annual)
    return scaling_factor * np.dot(weights, mean_returns)


def portfolio_risk(weights: np.ndarray, covar_matrix: np.ndarray, scaling_factor: int) -> float:
    """
    Calculate the annualized risk (standard deviation) of the portfolio.

    Args:
        weights (np.ndarray): Array of portfolio weights for each asset.
        covar_matrix (np.ndarray): Covariance matrix of the asset returns.
        scaling_factor (int): Scaling factor of the risk.

    Returns:
        float: Annualized risk (standard deviation) of the portfolio.
    """
    # Calculate portfolio variance and take the square root to get the standard deviation (scaled to annual)
    return np.sqrt(scaling_factor * np.dot(weights.T, np.dot(covar_matrix, weights)))


def portfolio_sharpe_ratio(exp_return: float, risk: float) -> float:
    """
    Calculate the Sharpe ratio of the portfolio.

    Args:
        exp_return (float): Expected return of the portfolio.
        risk (float): Risk (standard deviation) of the portfolio.

    Returns:
        float: Sharpe ratio of the portfolio.
    """
    return exp_return / risk


def optimization_func(mean_returns: np.ndarray, covar_matrix: np.ndarray, risk_avers_param: float,
                      number_of_assets: int) -> optimize.OptimizeResult:
    """
    Optimize the portfolio weights based on mean returns, covariance matrix, and risk aversion parameter.

    Args:
        mean_returns (np.ndarray): Array of mean daily returns for each asset.
        covar_matrix (np.ndarray): Covariance matrix of the asset returns.
        risk_avers_param (float): Risk aversion parameter (between 0 and 1).
        number_of_assets (int): Number of assets in the portfolio.

    Returns:
        optimize.OptimizeResult: Result of the optimization process, containing optimal weights and other information.
    """

    def objective(weights: np.ndarray, mean_returns: np.ndarray, covar_matrix: np.ndarray,
                  risk_avers_param: float) -> float:
        """
        Objective function to minimize for portfolio optimization.

        Args:
            weights (np.ndarray): Array of portfolio weights for each asset.
            mean_returns (np.ndarray): Array of mean daily returns for each asset.
            covar_matrix (np.ndarray): Covariance matrix of the asset returns.
            risk_avers_param (float): Risk aversion parameter.

        Returns:
            float: Value of the objective function.
        """
        portfolio_variance = np.dot(weights, np.dot(covar_matrix, weights.T))
        portfolio_exp_return = np.dot(mean_returns, weights.T)
        func = risk_avers_param * portfolio_variance - (1 - risk_avers_param) * portfolio_exp_return
        return func

    def constraint_sum_opt_weights(weights: np.ndarray) -> float:
        """
        Constraint function to ensure the sum of portfolio weights equals 1.

        Args:
            weights (np.ndarray): Array of portfolio weights for each asset.

        Returns:
            float: Value of the constraint function.
        """
        return np.sum(weights) - 1

    # Constraint: sum of weights must equal 1
    constraints = ({'type': 'eq', 'fun': constraint_sum_opt_weights})

    # Bounds for weights: between 0.05 and 0.2 for each asset
    bounds = tuple((0.025, 0.5) for _ in range(number_of_assets))

    # Initial guess for weights: equally distributed
    initial_weights = np.repeat(1 / number_of_assets, number_of_assets)

    # Perform the optimization using SLSQP (Sequential Least Squares Programming)
    opt_result = optimize.minimize(
        objective,
        x0=initial_weights,
        args=(mean_returns, covar_matrix, risk_avers_param),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-10
    )

    return opt_result


def efficient_frontier(returns_df: pd.DataFrame) -> Tuple[List[np.ndarray], List[float], List[float], List[float]]:
    """
    Calculate the efficient frontier for a given set of assets.

    Args:
    returns_df (pd.DataFrame): DataFrame containing returns for each asset.

    Returns:
        Tuple[List[np.ndarray], List[float], List[float], List[float]]:
            - List of optimal portfolio weights for each point on the efficient frontier.
            - List of portfolio risks (standard deviations) for each point.
            - List of expected portfolio returns for each point.
            - List of portfolio Sharpe ratios for each point.
    """
    # Compute mean returns and covariance matrix
    asset_names = np.array(returns_df.columns)
    mean_returns = get_mean_returns(returns_df)
    covar_matrix = get_covar_matrix(returns_df)

    # Determine frequency of returns based
    scaling_factor = detect_scaling_factor(returns_df)

    num_points = 1000  # Number of points on the efficient frontier
    num_assets = len(asset_names)  # Number of assets

    efficient_weights = []
    portfolio_risks = []
    portfolio_exp_returns = []
    portfolio_sharpe_ratios = []

    for i in range(num_points):
        # Risk aversion parameter ranges from 0 to 1
        risk_avers_param = i / num_points

        # Optimize portfolio for the current risk aversion parameter
        result = optimization_func(mean_returns, covar_matrix, risk_avers_param, num_assets)
        optimal_weights = np.array(result.x)

        # Round the values to two decimal places
        optimal_weights = np.round(optimal_weights, 2)

        # Normalize to ensure the sum is exactly 1
        optimal_weights /= np.sum(optimal_weights)

        # Final check
        if not np.isclose(np.sum(optimal_weights), 1.0, atol=1e-8):
            raise ValueError("The sum of the optimal weights is not equal to 1 after adjustment.")

        # Store the optimal weights
        efficient_weights.append(optimal_weights)

        # Calculate and store the portfolio risk
        risk = portfolio_risk(optimal_weights, covar_matrix, scaling_factor)
        portfolio_risks.append(risk)

        # Calculate and store the expected portfolio return
        exp_return = portfolio_exp_return(optimal_weights, mean_returns, scaling_factor)
        portfolio_exp_returns.append(exp_return)

        # Calculate and store the portfolio Sharpe ratio
        sharpe_ratio = portfolio_sharpe_ratio(exp_return, risk)
        portfolio_sharpe_ratios.append(sharpe_ratio)

    return efficient_weights, portfolio_risks, portfolio_exp_returns, portfolio_sharpe_ratios


def get_dataframe(efficient_weights: List[np.ndarray], portfolio_risk: List[float], portfolio_exp_return: List[float],
                  portfolio_sharpe_ratio: List[float]) -> pd.DataFrame:
    """
    Create a DataFrame from the lists of efficient weights, portfolio risks, expected returns, and Sharpe ratios.

    Args:
        efficient_weights (List[np.ndarray]): List of optimal weights for each point on the efficient frontier.
        portfolio_risk (List[float]): List of portfolio risks (standard deviations) for each point.
        portfolio_exp_return (List[float]): List of expected portfolio returns for each point.
        portfolio_sharpe_ratio (List[float]): List of portfolio Sharpe ratios for each point.

    Returns:
        pd.DataFrame: DataFrame containing the efficient frontier data.
    """
    dataframe = pd.DataFrame({
        'Weights': efficient_weights,
        'Risk': portfolio_risk,
        'Exp Return': portfolio_exp_return,
        'Sharpe Ratio': portfolio_sharpe_ratio
    })
    return dataframe


def get_min_risk_portfolio(efficient_frontier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the portfolio with the minimum risk from the efficient frontier DataFrame.

    Args:
        efficient_frontier_df (pd.DataFrame): DataFrame containing the efficient frontier data.

    Returns:
        pd.DataFrame: DataFrame containing the portfolio with the minimum risk.
    """
    min_risk_portfolio = efficient_frontier_df.nsmallest(1, 'Risk')
    return min_risk_portfolio


def get_max_exp_return_portfolio(efficient_frontier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the portfolio with the maximum expected return from the efficient frontier DataFrame.

    Args:
        efficient_frontier_df (pd.DataFrame): DataFrame containing the efficient frontier data.

    Returns:
        pd.DataFrame: DataFrame containing the portfolio with the maximum expected return.
    """
    max_exp_return_portfolio = efficient_frontier_df.nlargest(1, 'Exp Return')
    return max_exp_return_portfolio


def get_max_sharpe_ratio_portfolio(efficient_frontier_df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the portfolio with the maximum Sharpe ratio from the efficient frontier DataFrame.

    Args:
        efficient_frontier_df (pd.DataFrame): DataFrame containing the efficient frontier data.

    Returns:
        pd.DataFrame: DataFrame containing the portfolio with the maximum Sharpe ratio.
    """
    max_sharpe_ratio_portfolio = efficient_frontier_df.nlargest(1, 'Sharpe Ratio')
    return max_sharpe_ratio_portfolio


def efficient_frontier_json(efficient_frontier_df: pd.DataFrame, min_risk_portfolio: pd.DataFrame,
                            max_exp_return_portfolio: pd.DataFrame, max_sharpe_ratio_portfolio: pd.DataFrame) -> dict:
    """
    Convert the efficient frontier data and special portfolios to a JSON-serializable dictionary.

    Args:
        efficient_frontier_df (pd.DataFrame): DataFrame containing the efficient frontier data.
        min_risk_portfolio (pd.DataFrame): DataFrame containing the portfolio with the minimum risk.
        max_exp_return_portfolio (pd.DataFrame): DataFrame containing the portfolio with the maximum expected return.
        max_sharpe_ratio_portfolio (pd.DataFrame): DataFrame containing the portfolio with the maximum Sharpe ratio.

    Returns:
        dict: Dictionary containing the portfolios in structured format.
    """

    def convert_df(df):
        """ Convert DataFrame to a JSON-serializable dictionary, ensuring no NumPy arrays. """
        return df.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x).to_dict(orient='records')

    json_data = {
        "Minimum Risk Portfolio": convert_df(min_risk_portfolio),
        "Maximum Return Portfolio": convert_df(max_exp_return_portfolio),
        "Maximum Sharpe Ratio Portfolio": convert_df(max_sharpe_ratio_portfolio),
        "xEfficient Set Portfolios": convert_df(efficient_frontier_df)
    }

    # Save JSON to files (optional)
    for key, value in json_data.items():
        filename = f"{key}.json"
        with open(filename, 'w') as file:
            json.dump(value, file)

    return json_data