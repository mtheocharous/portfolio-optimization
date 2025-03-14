from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import portfolio_optimization  # Import your portfolio optimization functions
import os
import matplotlib.pyplot as plt
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def home():
    return jsonify({"message": "Flask app portfolio.py is running!", "Status": "OK"})

@app.route('/process', methods=['GET'])
def process_data():
    symbols = request.args.get('input', '')  # Get the input from the URL
    symbol_list = symbols.split(',') if symbols else []  # Convert to list
    print(symbol_list)

    if not symbol_list:
        return jsonify({"error": "No symbols provided"}), 400  # Handle empty input case

    try:
        # Run portfolio optimization and generate plot
        result, plot_img = optimize_portfolio(symbol_list)
        
        # Save the plot as an image
        img_io = io.BytesIO()
        plot_img.save(img_io, 'PNG')
        img_io.seek(0)

        # You can either return the image directly in the response or save it and provide a URL
        return send_file(img_io, mimetype='image/png')

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

    # Plot Efficient Frontier
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot for efficient frontier
    ax.scatter(portf_risks, portf_exp_returns, c=portf_sharpe_ratios, cmap='viridis', marker='o')
    ax.set_xlabel('Risk (Standard Deviation)')
    ax.set_ylabel('Expected Return')
    ax.set_title('Efficient Frontier')

    # Highlight the special portfolios
    ax.scatter(min_risk_portf['Risk'], min_risk_portf['Exp Return'], color='red', label='Min Risk Portfolio')
    ax.scatter(max_exp_return_portf['Risk'], max_exp_return_portf['Exp Return'], color='green', label='Max Exp Return Portfolio')
    ax.scatter(max_sharpe_ratio_portf['Risk'], max_sharpe_ratio_portf['Exp Return'], color='blue', label='Max Sharpe Ratio Portfolio')

    # Add legend
    ax.legend()

    # Save the plot to an image buffer
    canvas = FigureCanvas(fig)
    img_io = io.BytesIO()
    canvas.print_png(img_io)
    img_io.seek(0)

    return json_data, img_io  # Return the JSON data and the image buffer

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))  # default to port 8000 if port is not set
    app.run(host="0.0.0.0", port=port, debug=True)
