from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import os
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import portfolio_optimization  # Import your portfolio optimization functions

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def home():
    return jsonify({"message": "Flask app portfolio.py is running!", "Status": "OK"})

@app.route('/process', methods=['GET'])
def process_data():
    symbols = request.args.get('input', '')  # Get input from URL
    symbol_list = symbols.split(',') if symbols else []  # Convert to list

    if not symbol_list:
        return jsonify({"error": "No symbols provided"}), 400  # Handle empty input case

    try:
        # Run portfolio optimization and return JSON data
        result = optimize_portfolio(symbol_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors gracefully

@app.route('/plot', methods=['GET'])
def plot_efficient_frontier():
    """
    Generates and returns the efficient frontier plot as a PNG image.
    """
    symbols = request.args.get('input', '')  # Get symbols from URL
    symbol_list = symbols.split(',') if symbols else []  # Convert to list

    if not symbol_list:
        return jsonify({"error": "No symbols provided"}), 400

    try:
        # Fetch portfolio data
        prices_df = portfolio_optimization.get_asset_prices_yahoo(symbol_list)
        returns_df = portfolio_optimization.get_asset_returns(prices_df, 'weekly')

        # Compute efficient frontier
        efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios = portfolio_optimization.efficient_frontier(returns_df)

        # Create Efficient Frontier plot
        plt.figure(figsize=(8, 6))
        plt.scatter(portf_risks, portf_exp_returns, c=portf_sharpe_ratios, cmap='viridis', marker="o")
        plt.colorbar(label="Sharpe Ratio")
        plt.xlabel("Risk (Standard Deviation)")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier")
        plt.grid(True)

        # Save plot to a BytesIO buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")  # ✅ Fixes the `_io.BytesIO` error
        buffer.seek(0)  # Reset buffer position

        return send_file(buffer, mimetype="image/png")  # Return image response

    except Exception as e:
        return jsonify({"error": str(e)}), 500  # Handle errors gracefully

def optimize_portfolio(symbols):
    """
    Optimizes the portfolio and returns efficient frontier data in JSON format.
    """
    # Import asset prices and compute returns
    prices_df = portfolio_optimization.get_asset_prices_yahoo(symbols)
    returns_df = portfolio_optimization.get_asset_returns(prices_df, 'weekly')

    # Calculate efficient frontier
    efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios = portfolio_optimization.efficient_frontier(returns_df)

    # Convert results into a DataFrame
    efficient_frontier_df = portfolio_optimization.get_dataframe(efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios)

    # Get special portfolios
    min_risk_portf = portfolio_optimization.get_min_risk_portfolio(efficient_frontier_df)
    max_exp_return_portf = portfolio_optimization.get_max_exp_return_portfolio(efficient_frontier_df)
    max_sharpe_ratio_portf = portfolio_optimization.get_max_sharpe_ratio_portfolio(efficient_frontier_df)

    # Export data as JSON
    json_data = portfolio_optimization.efficient_frontier_json(efficient_frontier_df, min_risk_portf, max_exp_return_portf, max_sharpe_ratio_portf)

    return json_data

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # Default to port 8000 if not set
    app.run(host="0.0.0.0", port=port, debug=True)
