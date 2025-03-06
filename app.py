from flask import Flask, request, jsonify
from flask_cors import CORS
import portfolio_optimization  # Import your portfolio optimization functions
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def home():
    return jsonify({"message": "Flask app porfolio.py is running!", "Status": "OK"})

@app.route('/process', methods=['GET'])
def process_data():
    symbols = request.args.get('input', '')  # Get the input from the URL
    symbol_list = symbols.split(',') if symbols else []  # Convert to list
    print(symbol_list)

    if not symbol_list:
        return jsonify({"error": "No symbols provided"}), 400  # Handle empty input case

    try:
        # Run portfolio optimization
        result = optimize_portfolio(symbol_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors gracefully

def optimize_portfolio(symbols):
    """
    Optimizes the portfolio based on the given trading symbols.

    :param symbols: List of stock symbols to include in the optimization.
    :return: JSON response containing the efficient frontier results.
    """
    # Import asset prices and compute returns
    prices_df = portfolio_optimization.get_asset_prices_yahoo(symbols)
    returns_df = portfolio_optimization.get_asset_returns(prices_df, 'weekly')

    # Calculate the efficient frontier
    efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios = portfolio_optimization.efficient_frontier(returns_df)

    # Create DataFrame
    efficient_frontier_df = portfolio_optimization.get_dataframe(efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios)

    # Get special portfolios
    min_risk_portf = portfolio_optimization.get_min_risk_portfolio(efficient_frontier_df)
    max_exp_return_portf = portfolio_optimization.get_max_exp_return_portfolio(efficient_frontier_df)
    max_sharpe_ratio_portf = portfolio_optimization.get_max_sharpe_ratio_portfolio(efficient_frontier_df)

    # Export required data to JSON format
    json_data = portfolio_optimization.efficient_frontier_json(efficient_frontier_df, min_risk_portf, max_exp_return_portf, max_sharpe_ratio_portf)

    return json_data

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000)) #default to port 8000 if port is not set
    app.run(host="0.0.0.0", port=port, debug=True)

