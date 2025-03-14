from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import portfolio_optimization  # Import your portfolio optimization functions
import plotly.graph_objects as go
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all domains

@app.route('/')
def home():
    return jsonify({"message": "Flask app is running!", "Status": "OK"})

@app.route('/process', methods=['GET'])
def process_data():
    symbols = request.args.get('input', '')
    symbol_list = symbols.split(',') if symbols else []

    if not symbol_list:
        return jsonify({"error": "No symbols provided"}), 400

    try:
        result = optimize_portfolio(symbol_list)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def optimize_portfolio(symbols):
    """
    Optimizes the portfolio based on the given trading symbols.
    Returns portfolio optimization results in JSON format.
    """
    prices_df = portfolio_optimization.get_asset_prices_yahoo(symbols)
    returns_df = portfolio_optimization.get_asset_returns(prices_df, 'weekly')

    # Compute Efficient Frontier
    efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios = portfolio_optimization.efficient_frontier(returns_df)

    # Convert data to DataFrame
    efficient_frontier_df = portfolio_optimization.get_dataframe(efficient_weights, portf_risks, portf_exp_returns, portf_sharpe_ratios)

    # Get special portfolios
    min_risk_portf = portfolio_optimization.get_min_risk_portfolio(efficient_frontier_df)
    max_exp_return_portf = portfolio_optimization.get_max_exp_return_portfolio(efficient_frontier_df)
    max_sharpe_ratio_portf = portfolio_optimization.get_max_sharpe_ratio_portfolio(efficient_frontier_df)

    # Export required data to JSON format
    json_data = portfolio_optimization.efficient_frontier_json(efficient_frontier_df, min_risk_portf, max_exp_return_portf, max_sharpe_ratio_portf)

    return json_data

@app.route('/plot')
def plot_efficient_frontier():
    symbols = request.args.get('input', '')
    symbol_list = symbols.split(',') if symbols else []

    if not symbol_list:
        return jsonify({"error": "No symbols provided"}), 400

    try:
        # Run optimization
        result = optimize_portfolio(symbol_list)
        efficient_frontier = result["xEfficient Set Portfolios"]

        # Extract data
        risks = [p["Risk"] for p in efficient_frontier]
        returns = [p["Exp Return"] for p in efficient_frontier]
        sharpe_ratios = [p["Sharpe Ratio"] for p in efficient_frontier]

        # Create a Plotly scatter plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=risks, 
            y=returns, 
            mode='markers', 
            marker=dict(size=10, color=sharpe_ratios, colorscale='Viridis', showscale=True),
            name="Efficient Frontier"
        ))
        
        fig.update_layout(
            title="Efficient Frontier",
            xaxis_title="Risk (Standard Deviation)",
            yaxis_title="Expected Return",
            template="plotly_white"
        )

        # Render as an HTML page
        return render_template_string("""
            <html>
            <head>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            </head>
            <body>
                <h1>Efficient Frontier</h1>
                <div id="plotly-div"></div>
                <script>
                    var plotly_data = {{ plot | safe }};
                    Plotly.newPlot('plotly-div', plotly_data.data, plotly_data.layout);
                </script>
            </body>
            </html>
        """, plot=fig.to_json())

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
