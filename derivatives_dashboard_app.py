import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
from scipy.stats import norm
from scipy.interpolate import griddata, RBFInterpolator
from scipy.optimize import minimize_scalar, minimize
import io
import base64
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Derivatives Pricing Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .greeks-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .exotic-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ====================
# PRICING ENGINES
# ====================

class BlackScholesEngine:
    """Enhanced Black-Scholes pricing engine"""
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return BlackScholesEngine._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def price_option(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        d1 = BlackScholesEngine._d1(S, K, T, r, sigma)
        d2 = BlackScholesEngine._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return max(price, 0)

class BinomialTreeEngine:
    """Enhanced Binomial tree with exotic option support"""
    
    @staticmethod
    def price_vanilla_option(S: float, K: float, T: float, r: float, sigma: float, 
                           option_type: str, n_steps: int = 100) -> float:
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        prices = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            prices[i] = S * (u ** (n_steps - i)) * (d ** i)
        
        if option_type.lower() == 'call':
            option_values = np.maximum(prices - K, 0)
        else:
            option_values = np.maximum(K - prices, 0)
        
        for j in range(n_steps - 1, -1, -1):
            for i in range(j + 1):
                hold_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                current_price = S * (u ** (j - i)) * (d ** i)
                
                if option_type.lower() == 'call':
                    exercise_value = max(current_price - K, 0)
                else:
                    exercise_value = max(K - current_price, 0)
                
                option_values[i] = max(hold_value, exercise_value)
        
        return option_values[0]
    
    @staticmethod
    def price_barrier_option(S: float, K: float, T: float, r: float, sigma: float, 
                           barrier: float, option_type: str, barrier_type: str, n_steps: int = 200) -> float:
        """
        Price barrier options using binomial tree
        barrier_type: 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        """
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Build price tree
        price_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        barrier_hit = np.zeros((n_steps + 1, n_steps + 1), dtype=bool)
        
        # Initialize stock prices
        price_tree[0, 0] = S
        
        # Fill price tree and check barrier conditions
        for i in range(n_steps):
            for j in range(i + 1):
                price_up = price_tree[i, j] * u
                price_down = price_tree[i, j] * d
                
                price_tree[i + 1, j] = price_up
                price_tree[i + 1, j + 1] = price_down
                
                # Check barrier conditions
                if barrier_type in ['up-and-out', 'up-and-in']:
                    if price_up >= barrier:
                        barrier_hit[i + 1, j] = True
                    if price_down >= barrier:
                        barrier_hit[i + 1, j + 1] = True
                else:  # down barriers
                    if price_up <= barrier:
                        barrier_hit[i + 1, j] = True
                    if price_down <= barrier:
                        barrier_hit[i + 1, j + 1] = True
                
                # Propagate barrier hit status
                barrier_hit[i + 1, j] = barrier_hit[i + 1, j] or barrier_hit[i, j]
                barrier_hit[i + 1, j + 1] = barrier_hit[i + 1, j + 1] or barrier_hit[i, j]
        
        # Terminal payoffs
        for j in range(n_steps + 1):
            terminal_price = price_tree[n_steps, j]
            
            if option_type.lower() == 'call':
                vanilla_payoff = max(terminal_price - K, 0)
            else:
                vanilla_payoff = max(K - terminal_price, 0)
            
            if barrier_type in ['up-and-out', 'down-and-out']:
                option_tree[n_steps, j] = vanilla_payoff if not barrier_hit[n_steps, j] else 0
            else:  # knock-in
                option_tree[n_steps, j] = vanilla_payoff if barrier_hit[n_steps, j] else 0
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[i, j] = np.exp(-r * dt) * (p * option_tree[i + 1, j] + 
                                                       (1 - p) * option_tree[i + 1, j + 1])
        
        return option_tree[0, 0]
    
    @staticmethod
    def price_asian_option(S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str, n_steps: int = 100) -> float:
        """
        Price arithmetic Asian options using binomial tree
        This is a simplified implementation - full implementation would require
        tracking all possible average paths
        """
        # For simplicity, using geometric Asian approximation
        # In practice, you'd use Monte Carlo for arithmetic Asian options
        
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Adjusted parameters for geometric Asian
        sigma_adj = sigma / np.sqrt(3)
        r_adj = (r + sigma**2 / 6) / 2
        
        # Use Black-Scholes with adjusted parameters
        d1 = (np.log(S / K) + (r_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S * np.exp((r_adj - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((r_adj - r) * T) * norm.cdf(-d1)
        
        return max(price, 0)

class MonteCarloEngine:
    """Monte Carlo pricing engine for exotic options"""
    
    @staticmethod
    def price_asian_option(S: float, K: float, T: float, r: float, sigma: float,
                         option_type: str, n_simulations: int = 10000, n_steps: int = 252) -> Dict:
        """Price arithmetic Asian option using Monte Carlo"""
        dt = T / n_steps
        payoffs = []
        
        for _ in range(n_simulations):
            # Generate price path
            prices = [S]
            for _ in range(n_steps):
                z = np.random.standard_normal()
                price = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                prices.append(price)
            
            # Calculate average price
            avg_price = np.mean(prices)
            
            # Calculate payoff
            if option_type.lower() == 'call':
                payoff = max(avg_price - K, 0)
            else:
                payoff = max(K - avg_price, 0)
            
            payoffs.append(payoff)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': (option_price - 1.96 * std_error, option_price + 1.96 * std_error)
        }
    
    @staticmethod
    def price_barrier_option(S: float, K: float, T: float, r: float, sigma: float,
                           barrier: float, option_type: str, barrier_type: str,
                           n_simulations: int = 10000, n_steps: int = 252) -> Dict:
        """Price barrier option using Monte Carlo"""
        dt = T / n_steps
        payoffs = []
        
        for _ in range(n_simulations):
            # Generate price path
            prices = [S]
            barrier_hit = False
            
            for _ in range(n_steps):
                z = np.random.standard_normal()
                price = prices[-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
                prices.append(price)
                
                # Check barrier condition
                if barrier_type in ['up-and-out', 'up-and-in']:
                    if price >= barrier:
                        barrier_hit = True
                else:  # down barriers
                    if price <= barrier:
                        barrier_hit = True
            
            # Calculate payoff
            final_price = prices[-1]
            if option_type.lower() == 'call':
                vanilla_payoff = max(final_price - K, 0)
            else:
                vanilla_payoff = max(K - final_price, 0)
            
            if barrier_type in ['up-and-out', 'down-and-out']:
                payoff = vanilla_payoff if not barrier_hit else 0
            else:  # knock-in
                payoff = vanilla_payoff if barrier_hit else 0
            
            payoffs.append(payoff)
        
        # Discount to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_simulations)
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': (option_price - 1.96 * std_error, option_price + 1.96 * std_error)
        }

# ====================
# IMPLIED VOLATILITY CALIBRATION
# ====================

class ImpliedVolatilityCalibrator:
    """Calibrate implied volatility from market prices"""
    
    @staticmethod
    def implied_volatility(market_price: float, S: float, K: float, T: float, 
                          r: float, option_type: str, initial_guess: float = 0.2) -> float:
        """Calculate implied volatility using Brent's method"""
        
        def objective(sigma):
            try:
                model_price = BlackScholesEngine.price_option(S, K, T, r, sigma, option_type)
                return (model_price - market_price) ** 2
            except:
                return float('inf')
        
        try:
            result = minimize_scalar(objective, bounds=(0.001, 5.0), method='bounded')
            return result.x if result.success else initial_guess
        except:
            return initial_guess
    
    @staticmethod
    def calibrate_surface(option_data: pd.DataFrame, S: float, r: float) -> Dict:
        """Calibrate implied volatility surface from option chain data"""
        implied_vols = []
        
        for _, row in option_data.iterrows():
            K = row['strike']
            T = row['time_to_maturity']
            market_price = row['market_price']
            option_type = row['option_type']
            
            iv = ImpliedVolatilityCalibrator.implied_volatility(
                market_price, S, K, T, r, option_type
            )
            implied_vols.append(iv)
        
        option_data = option_data.copy()
        option_data['implied_volatility'] = implied_vols
        
        return {
            'calibrated_data': option_data,
            'rmse': np.sqrt(np.mean([(row['market_price'] - 
                                    BlackScholesEngine.price_option(S, row['strike'], 
                                                                   row['time_to_maturity'], r, 
                                                                   row['implied_volatility'], 
                                                                   row['option_type']))**2 
                                   for _, row in option_data.iterrows()]))
        }
    
    @staticmethod
    def interpolate_volatility_surface(strikes: np.ndarray, maturities: np.ndarray, 
                                     implied_vols: np.ndarray, 
                                     method: str = 'rbf') -> callable:
        """Create interpolated volatility surface"""
        
        if method == 'rbf':
            # Use RBF interpolation
            points = np.column_stack([strikes.flatten(), maturities.flatten()])
            values = implied_vols.flatten()
            interpolator = RBFInterpolator(points, values, smoothing=0.1)
            
            def vol_surface(K, T):
                query_points = np.column_stack([np.array(K).flatten(), np.array(T).flatten()])
                return interpolator(query_points).reshape(np.array(K).shape)
            
            return vol_surface
        
        else:
            # Use griddata for linear/cubic interpolation
            points = np.column_stack([strikes.flatten(), maturities.flatten()])
            values = implied_vols.flatten()
            
            def vol_surface(K, T):
                query_points = np.column_stack([np.array(K).flatten(), np.array(T).flatten()])
                interpolated = griddata(points, values, query_points, method=method, fill_value=0.2)
                return interpolated.reshape(np.array(K).shape)
            
            return vol_surface

# ====================
# GREEKS CALCULATION
# ====================

class GreeksCalculator:
    """Enhanced Greeks calculator with finite difference methods"""
    
    @staticmethod
    def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float, 
                        option_type: str) -> Dict[str, float]:
        if T <= 0:
            return {
                'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0
            }
        
        d1 = BlackScholesEngine._d1(S, K, T, r, sigma)
        d2 = BlackScholesEngine._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            delta = norm.cdf(d1)
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            delta = norm.cdf(d1) - 1
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                    + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
    
    @staticmethod
    def finite_difference_greeks(S: float, K: float, T: float, r: float, sigma: float,
                               option_type: str, pricing_func, epsilon: float = 0.01) -> Dict[str, float]:
        """Calculate Greeks using finite differences for exotic options"""
        
        base_price = pricing_func(S, K, T, r, sigma, option_type)
        
        # Delta
        price_up = pricing_func(S * (1 + epsilon), K, T, r, sigma, option_type)
        price_down = pricing_func(S * (1 - epsilon), K, T, r, sigma, option_type)
        delta = (price_up - price_down) / (2 * S * epsilon)
        
        # Gamma
        gamma = (price_up - 2 * base_price + price_down) / ((S * epsilon) ** 2)
        
        # Vega
        price_vol_up = pricing_func(S, K, T, r, sigma * (1 + epsilon), option_type)
        price_vol_down = pricing_func(S, K, T, r, sigma * (1 - epsilon), option_type)
        vega = (price_vol_up - price_vol_down) / (2 * sigma * epsilon) / 100
        
        # Theta (using small time increment)
        time_epsilon = 1/365  # One day
        if T > time_epsilon:
            price_time = pricing_func(S, K, T - time_epsilon, r, sigma, option_type)
            theta = -(price_time - base_price) / time_epsilon * (1/365)
        else:
            theta = 0
        
        # Rho
        price_rate_up = pricing_func(S, K, T, r * (1 + epsilon), sigma, option_type)
        price_rate_down = pricing_func(S, K, T, r * (1 - epsilon), sigma, option_type)
        rho = (price_rate_up - price_rate_down) / (2 * r * epsilon) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }

# ====================
# DATA FUNCTIONS
# ====================

@st.cache_data(ttl=300)
def fetch_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_current_price(symbol: str) -> float:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        return data['Close'].iloc[-1] if not data.empty else 100.0
    except:
        return 100.0

def generate_sample_option_chain(S: float, r: float) -> pd.DataFrame:
    """Generate sample option chain data for calibration demo"""
    np.random.seed(42)  # For reproducible results
    
    strikes = np.array([S * k for k in [0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2]])
    maturities = np.array([0.25, 0.5, 0.75, 1.0])  # 3, 6, 9, 12 months
    
    data = []
    for T in maturities:
        for K in strikes:
            for option_type in ['call', 'put']:
                # Generate synthetic market price with some noise
                true_vol = 0.2 + 0.05 * np.sin(K/S) + 0.02 * np.random.randn()
                true_vol = max(0.1, true_vol)  # Minimum vol
                
                theoretical_price = BlackScholesEngine.price_option(S, K, T, r, true_vol, option_type)
                market_price = theoretical_price * (1 + 0.02 * np.random.randn())  # Add noise
                market_price = max(0.01, market_price)  # Minimum price
                
                data.append({
                    'strike': K,
                    'time_to_maturity': T,
                    'option_type': option_type,
                    'market_price': market_price,
                    'true_vol': true_vol  # For comparison (wouldn't be available in real data)
                })
    
    return pd.DataFrame(data)

# ====================
# VISUALIZATION FUNCTIONS
# ====================

def create_volatility_surface_plot(calibrated_data: pd.DataFrame) -> go.Figure:
    """Create 3D volatility surface plot"""
    # Separate calls and puts
    calls = calibrated_data[calibrated_data['option_type'] == 'call']
    
    if len(calls) == 0:
        return go.Figure()
    
    # Create meshgrid for surface
    strikes = np.sort(calls['strike'].unique())
    maturities = np.sort(calls['time_to_maturity'].unique())
    
    if len(strikes) < 2 or len(maturities) < 2:
        return go.Figure()
    
    # Create volatility matrix
    vol_matrix = np.zeros((len(maturities), len(strikes)))
    
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            mask = (calls['strike'] == K) & (calls['time_to_maturity'] == T)
            if mask.any():
                vol_matrix[i, j] = calls[mask]['implied_volatility'].iloc[0]
    
    fig = go.Figure(data=[go.Surface(
        x=strikes,
        y=maturities,
        z=vol_matrix,
        colorscale='Viridis',
        colorbar=dict(title="Implied Volatility"),
        showscale=True
    )])
    
    fig.update_layout(
        title='Implied Volatility Surface',
        scene=dict(
            xaxis_title='Strike Price',
            yaxis_title='Time to Maturity (Years)',
            zaxis_title='Implied Volatility',
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6))
        ),
        template='plotly_white',
        height=600
    )
    
    return fig

def create_exotic_payoff_diagram(S: float, K: float, barrier: float = None, 
                                option_type: str = 'call', exotic_type: str = 'vanilla') -> go.Figure:
    """Create payoff diagrams for exotic options"""
    spot_range = np.linspace(S * 0.5, S * 1.5, 100)
    
    fig = go.Figure()
    
    if exotic_type == 'vanilla':
        if option_type.lower() == 'call':
            payoffs = np.maximum(spot_range - K, 0)
        else:
            payoffs = np.maximum(K - spot_range, 0)
        title = f'Vanilla {option_type.title()} Payoff'
    
    elif exotic_type == 'barrier':
        if barrier is None:
            barrier = K * 1.2 if option_type == 'call' else K * 0.8
        
        if option_type.lower() == 'call':
            vanilla_payoffs = np.maximum(spot_range - K, 0)
        else:
            vanilla_payoffs = np.maximum(K - spot_range, 0)
        
        # Simplified barrier payoff (knock-out)
        payoffs = np.where(spot_range >= barrier, 0, vanilla_payoffs) if barrier > K else vanilla_payoffs
        title = f'Barrier {option_type.title()} Payoff (Barrier: ${barrier:.2f})'
        
        # Add barrier line
        fig.add_vline(x=barrier, line_dash="dash", line_color="red", 
                     annotation_text=f"Barrier: ${barrier:.2f}")
    
    else:  # Asian approximation
        if option_type.lower() == 'call':
            payoffs = np.maximum(spot_range - K, 0) * 0.8  # Simplified Asian approximation
        else:
            payoffs = np.maximum(K - spot_range, 0) * 0.8
        title = f'Asian {option_type.title()} Payoff (Approximation)'
    
    fig.add_trace(go.Scatter(
        x=spot_range,
        y=payoffs,
        mode='lines',
        name='Payoff',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_vline(x=K, line_dash="dash", line_color="green", 
                 annotation_text=f"Strike: ${K:.2f}")
    fig.add_vline(x=S, line_dash="dash", line_color="orange", 
                 annotation_text=f"Current: ${S:.2f}")
    
    fig.update_layout(
        title=title,
        xaxis_title='Underlying Price at Expiry',
        yaxis_title='Payoff',
        template='plotly_white',
        height=400
    )
    
    return fig

# ====================
# MAIN APPLICATION
# ====================

def main():
    st.markdown('<h1 class="main-header">🏦 Advanced Derivatives Pricing Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Main navigation
    main_tabs = st.tabs(["🔵 Vanilla Options", "🟣 Exotic Options", "📊 Vol Surface Calibration"])
    
    # ========== VANILLA OPTIONS TAB ==========
    with main_tabs[0]:
        # Sidebar for vanilla options
        with st.sidebar:
            st.title("📊 Vanilla Option Parameters")
            
            # Provide a dropdown of common tickers with an 'Other' fallback
            TOP_TICKERS = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',
                'TSLA', 'NVDA', 'JPM', 'V', 'UNH'
            ]

            symbol_choice = st.selectbox("Stock Symbol", options=TOP_TICKERS + ["Other"], index=0)
            if symbol_choice == "Other":
                symbol = st.text_input("Enter ticker symbol", value="")
            else:
                symbol = symbol_choice

            current_price = get_current_price(symbol) if symbol else 100.0
            st.success(f"Current {symbol} price: ${current_price:.2f}" if symbol else "Default price: $100.00")
            
            spot_price = st.number_input("Spot Price ($)", value=float(current_price), min_value=0.01)
            strike_price = st.number_input("Strike Price ($)", value=float(current_price), min_value=0.01)
            
            days_to_expiry = st.slider("Days to Expiry", 1, 365, 30)
            time_to_maturity = days_to_expiry / 365.0
            
            risk_free_rate = st.slider("Risk-free Rate (%)", 0.0, 10.0, 5.0) / 100
            volatility = st.slider("Volatility (%)", 1.0, 100.0, 20.0) / 100
            
            option_type = st.selectbox("Option Type", ["Call", "Put"])
            model_type = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree"])
            
            if model_type == "Binomial Tree":
                n_steps = st.slider("Tree Steps", 10, 500, 100)
        
        # Calculate vanilla option price
        if model_type == "Black-Scholes":
            option_price = BlackScholesEngine.price_option(
                spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type
            )
        else:
            option_price = BinomialTreeEngine.price_vanilla_option(
                spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type, n_steps
            )
        
        # Calculate Greeks
        greeks = GreeksCalculator.calculate_greeks(
            spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type
        )
        
        # Display vanilla option metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <h3>Option Price</h3>
                <h2>${option_price:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown(f'''
            <div class="greeks-card">
                <h3>Delta</h3>
                <h2>{greeks["delta"]:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col3:
            st.markdown(f'''
            <div class="greeks-card">
                <h3>Gamma</h3>
                <h2>{greeks["gamma"]:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        with col4:
            st.markdown(f'''
            <div class="greeks-card">
                <h3>Vega</h3>
                <h2>{greeks["vega"]:.4f}</h2>
            </div>
            ''', unsafe_allow_html=True)
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Theta (per day)", f"{greeks['theta']:.4f}")
        with col6:
            st.metric("Rho", f"{greeks['rho']:.4f}")
        with col7:
            moneyness = spot_price / strike_price
            st.metric("Moneyness", f"{moneyness:.4f}")
        with col8:
            intrinsic = max(0, (spot_price - strike_price) if option_type == "Call" else (strike_price - spot_price))
            time_value = option_price - intrinsic
            st.metric("Time Value", f"${time_value:.4f}")
        
        # Vanilla option analysis tabs
        vanilla_tabs = st.tabs(["📈 Price Analysis", "🔄 Greeks", "📊 Sensitivity", "🔧 Model Comparison"])
        
        with vanilla_tabs[0]:
            price_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
            
            if model_type == "Black-Scholes":
                prices = [BlackScholesEngine.price_option(S, strike_price, time_to_maturity, risk_free_rate, volatility, option_type) 
                         for S in price_range]
            else:
                prices = [BinomialTreeEngine.price_vanilla_option(S, strike_price, time_to_maturity, risk_free_rate, volatility, option_type, n_steps) 
                         for S in price_range]
            
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(x=price_range, y=prices, mode='lines', name='Option Price', line=dict(color='blue', width=3)))
            fig_price.add_vline(x=spot_price, line_dash="dash", line_color="red", annotation_text="Current Spot")
            fig_price.add_vline(x=strike_price, line_dash="dash", line_color="green", annotation_text="Strike")
            fig_price.update_layout(title=f'{option_type} Option Price vs Underlying', 
                                  xaxis_title='Underlying Price', yaxis_title='Option Price', 
                                  template='plotly_white', height=400)
            st.plotly_chart(fig_price, use_container_width=True)
            
            # Payoff diagram
            if option_type.lower() == "call":
                payoffs = np.maximum(price_range - strike_price, 0)
            else:
                payoffs = np.maximum(strike_price - price_range, 0)
            
            fig_payoff = go.Figure()
            fig_payoff.add_trace(go.Scatter(x=price_range, y=payoffs, mode='lines', name='Payoff', line=dict(color='red', width=3)))
            fig_payoff.add_vline(x=strike_price, line_dash="dash", line_color="green", annotation_text="Strike")
            fig_payoff.update_layout(title=f'{option_type} Option Payoff at Expiry',
                                   xaxis_title='Underlying Price', yaxis_title='Payoff',
                                   template='plotly_white', height=400)
            st.plotly_chart(fig_payoff, use_container_width=True)
        
        with vanilla_tabs[1]:
            # Greeks analysis (similar to original implementation)
            greeks_data = {greek: [] for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']}
            
            for S in price_range:
                current_greeks = GreeksCalculator.calculate_greeks(
                    S, strike_price, time_to_maturity, risk_free_rate, volatility, option_type
                )
                for greek in greeks_data:
                    greeks_data[greek].append(current_greeks[greek])
            
            fig_greeks = make_subplots(rows=2, cols=3, subplot_titles=['Delta', 'Gamma', 'Vega', 'Theta', 'Rho'])
            
            colors = ['blue', 'red', 'green', 'purple', 'orange']
            greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']
            positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
            
            for i, (greek, color) in enumerate(zip(greek_names, colors)):
                row, col = positions[i]
                fig_greeks.add_trace(
                    go.Scatter(x=price_range, y=greeks_data[greek], mode='lines', 
                             name=greek.title(), line=dict(color=color, width=2)),
                    row=row, col=col
                )
            
            fig_greeks.update_layout(title='Greeks vs Underlying Price', template='plotly_white', 
                                   height=600, showlegend=False)
            st.plotly_chart(fig_greeks, use_container_width=True)
        
        with vanilla_tabs[2]:
            col_left, col_right = st.columns(2)
            
            with col_left:
                # Volatility sensitivity
                vol_range = np.linspace(0.1, 1.0, 50)
                vol_prices = [BlackScholesEngine.price_option(spot_price, strike_price, time_to_maturity, risk_free_rate, vol, option_type) 
                             for vol in vol_range]
                
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=vol_range*100, y=vol_prices, mode='lines', name='Option Price'))
                fig_vol.add_vline(x=volatility*100, line_dash="dash", line_color="red", annotation_text="Current Vol")
                fig_vol.update_layout(title='Price vs Volatility', xaxis_title='Volatility (%)', 
                                    yaxis_title='Option Price ($)', template='plotly_white')
                st.plotly_chart(fig_vol, use_container_width=True)
            
            with col_right:
                # Time decay
                time_range = np.linspace(0.01, 1.0, 50)
                time_prices = [BlackScholesEngine.price_option(spot_price, strike_price, t, risk_free_rate, volatility, option_type) 
                              for t in time_range]
                
                fig_time = go.Figure()
                fig_time.add_trace(go.Scatter(x=time_range*365, y=time_prices, mode='lines', name='Option Price'))
                fig_time.add_vline(x=time_to_maturity*365, line_dash="dash", line_color="red", annotation_text="Current TTM")
                fig_time.update_layout(title='Price vs Time to Maturity', xaxis_title='Days to Maturity', 
                                     yaxis_title='Option Price ($)', template='plotly_white')
                st.plotly_chart(fig_time, use_container_width=True)
        
        with vanilla_tabs[3]:
            # Model comparison
            steps_range = range(10, 201, 10)
            bs_price = BlackScholesEngine.price_option(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type)
            binomial_prices = [BinomialTreeEngine.price_vanilla_option(spot_price, strike_price, time_to_maturity, risk_free_rate, volatility, option_type, steps) 
                              for steps in steps_range]
            
            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Scatter(x=list(steps_range), y=binomial_prices, mode='lines+markers', 
                                              name='Binomial Tree', line=dict(color='blue')))
            fig_comparison.add_hline(y=bs_price, line_dash="dash", line_color="red", 
                                   annotation_text=f"Black-Scholes: ${bs_price:.4f}")
            fig_comparison.update_layout(title='Model Convergence: Binomial Tree vs Black-Scholes',
                                       xaxis_title='Number of Steps', yaxis_title='Option Price ($)',
                                       template='plotly_white')
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # ========== EXOTIC OPTIONS TAB ==========
    with main_tabs[1]:
        st.header("🟣 Exotic Options Pricing")
        
        exotic_col1, exotic_col2 = st.columns([1, 2])
        
        with exotic_col1:
            st.subheader("Parameters")
            
            # Use same basic parameters from vanilla
            exotic_spot = st.number_input("Spot Price ($)", value=100.0, min_value=0.01, key="exotic_spot")
            exotic_strike = st.number_input("Strike Price ($)", value=100.0, min_value=0.01, key="exotic_strike")
            exotic_days = st.slider("Days to Expiry", 1, 365, 30, key="exotic_days")
            exotic_T = exotic_days / 365.0
            exotic_r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 5.0, key="exotic_r") / 100
            exotic_vol = st.slider("Volatility (%)", 1.0, 100.0, 20.0, key="exotic_vol") / 100
            exotic_option_type = st.selectbox("Option Type", ["Call", "Put"], key="exotic_type")
            
            st.divider()
            
            exotic_type = st.selectbox("Exotic Option Type", ["Asian", "Barrier"])
            
            if exotic_type == "Barrier":
                barrier_type = st.selectbox("Barrier Type", ["Up-and-Out", "Up-and-In", "Down-and-Out", "Down-and-In"])
                barrier_level = st.number_input("Barrier Level ($)", value=exotic_spot * 1.2, min_value=0.01)
                pricing_method = st.selectbox("Pricing Method", ["Binomial Tree", "Monte Carlo"])
                
                if pricing_method == "Monte Carlo":
                    n_simulations = st.slider("Simulations", 1000, 50000, 10000)
            else:  # Asian
                pricing_method = st.selectbox("Pricing Method", ["Binomial Tree (Geometric)", "Monte Carlo (Arithmetic)"])
                if pricing_method == "Monte Carlo (Arithmetic)":
                    n_simulations = st.slider("Simulations", 1000, 50000, 10000)
        
        with exotic_col2:
            st.subheader("Pricing Results")
            
            # Price exotic option
            if exotic_type == "Barrier":
                if pricing_method == "Binomial Tree":
                    exotic_price = BinomialTreeEngine.price_barrier_option(
                        exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol,
                        barrier_level, exotic_option_type, barrier_type.lower().replace("-", "-"), 200
                    )
                    
                    st.markdown(f'''
                    <div class="exotic-card">
                        <h3>{barrier_type} {exotic_option_type} Option</h3>
                        <h2>${exotic_price:.4f}</h2>
                        <p>Binomial Tree (200 steps)</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                else:  # Monte Carlo
                    mc_result = MonteCarloEngine.price_barrier_option(
                        exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol,
                        barrier_level, exotic_option_type, barrier_type.lower().replace("-", "-"), n_simulations
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f'''
                        <div class="exotic-card">
                            <h3>{barrier_type} {exotic_option_type}</h3>
                            <h2>${mc_result["price"]:.4f}</h2>
                            <p>Monte Carlo ({n_simulations:,} sims)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f'''
                        <div class="info-box">
                            <h4>Confidence Interval (95%)</h4>
                            <p>${mc_result["confidence_interval"][0]:.4f} - ${mc_result["confidence_interval"][1]:.4f}</p>
                            <p>Std Error: ${mc_result["std_error"]:.4f}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Compare with vanilla
                vanilla_price = BlackScholesEngine.price_option(
                    exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol, exotic_option_type
                )
                
                st.metric("Vanilla Option Price", f"${vanilla_price:.4f}")
                
                # Calculate finite difference Greeks for barrier option
                def barrier_pricing_func(S, K, T, r, sigma, opt_type):
                    return BinomialTreeEngine.price_barrier_option(
                        S, K, T, r, sigma, barrier_level, opt_type, barrier_type.lower().replace("-", "-"), 100
                    )
                
                barrier_greeks = GreeksCalculator.finite_difference_greeks(
                    exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol, 
                    exotic_option_type, barrier_pricing_func
                )
                
                st.subheader("Barrier Option Greeks (Finite Difference)")
                greek_cols = st.columns(5)
                greek_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
                for i, (name, value) in enumerate(zip(greek_names, barrier_greeks.values())):
                    with greek_cols[i]:
                        st.metric(name, f"{value:.4f}")
                
            else:  # Asian
                if pricing_method == "Binomial Tree (Geometric)":
                    exotic_price = BinomialTreeEngine.price_asian_option(
                        exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol, exotic_option_type, 100
                    )
                    
                    st.markdown(f'''
                    <div class="exotic-card">
                        <h3>Asian {exotic_option_type} Option</h3>
                        <h2>${exotic_price:.4f}</h2>
                        <p>Geometric Average (Analytical)</p>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                else:  # Monte Carlo
                    mc_result = MonteCarloEngine.price_asian_option(
                        exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol, exotic_option_type, n_simulations
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.markdown(f'''
                        <div class="exotic-card">
                            <h3>Asian {exotic_option_type}</h3>
                            <h2>${mc_result["price"]:.4f}</h2>
                            <p>Arithmetic Average (MC)</p>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f'''
                        <div class="info-box">
                            <h4>Confidence Interval (95%)</h4>
                            <p>${mc_result["confidence_interval"][0]:.4f} - ${mc_result["confidence_interval"][1]:.4f}</p>
                            <p>Std Error: ${mc_result["std_error"]:.4f}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                # Compare with vanilla
                vanilla_price = BlackScholesEngine.price_option(
                    exotic_spot, exotic_strike, exotic_T, exotic_r, exotic_vol, exotic_option_type
                )
                
                st.metric("Vanilla Option Price", f"${vanilla_price:.4f}")
        
        # Exotic option payoff diagram
        st.subheader("Payoff Analysis")
        
        if exotic_type == "Barrier":
            payoff_fig = create_exotic_payoff_diagram(
                exotic_spot, exotic_strike, barrier_level, exotic_option_type, 'barrier'
            )
        else:
            payoff_fig = create_exotic_payoff_diagram(
                exotic_spot, exotic_strike, None, exotic_option_type, 'asian'
            )
        
        st.plotly_chart(payoff_fig, use_container_width=True)
    
    # ========== VOLATILITY SURFACE CALIBRATION TAB ==========
    with main_tabs[2]:
        st.header("📊 Implied Volatility Surface Calibration")
        
        cal_col1, cal_col2 = st.columns([1, 2])
        
        with cal_col1:
            st.subheader("Calibration Settings")
            
            # Calibration: same top tickers dropdown with 'Other'
            CAL_TOP_TICKERS = [
                'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',
                'TSLA', 'NVDA', 'JPM', 'V', 'UNH'
            ]

            cal_choice = st.selectbox("Stock Symbol", options=CAL_TOP_TICKERS + ["Other"], index=0, key="cal_symbol_choice")
            if cal_choice == "Other":
                cal_symbol = st.text_input("Enter ticker symbol for calibration", value="", key="cal_symbol")
            else:
                cal_symbol = cal_choice

            cal_spot = get_current_price(cal_symbol) if cal_symbol else 100.0
            st.info(f"Using spot price: ${cal_spot:.2f}")
            
            cal_r = st.slider("Risk-free Rate (%)", 0.0, 10.0, 5.0, key="cal_r") / 100
            
            use_sample_data = st.checkbox("Use Sample Option Chain Data", value=True)
            
            if use_sample_data:
                st.info("Using synthetic option chain data for demonstration")
                
                if st.button("Generate Sample Data"):
                    sample_data = generate_sample_option_chain(cal_spot, cal_r)
                    st.session_state.option_chain_data = sample_data
                
                if 'option_chain_data' in st.session_state:
                    option_data = st.session_state.option_chain_data
                    st.success(f"Loaded {len(option_data)} option contracts")
                    
                    # Show sample of data
                    st.write("Sample Option Chain Data:")
                    st.dataframe(option_data.head(10))
                    
            else:
                st.info("Upload your own option chain CSV file")
                uploaded_file = st.file_uploader("Choose CSV file", type="csv")
                
                if uploaded_file is not None:
                    try:
                        option_data = pd.read_csv(uploaded_file)
                        st.success(f"Loaded {len(option_data)} option contracts")
                        st.dataframe(option_data.head())
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
                        option_data = None
                else:
                    option_data = None
        
        with cal_col2:
            if 'option_chain_data' in st.session_state or (not use_sample_data and 'option_data' in locals() and option_data is not None):
                if use_sample_data:
                    data_to_calibrate = st.session_state.option_chain_data
                else:
                    data_to_calibrate = option_data
                
                st.subheader("Calibration Results")
                
                # Perform calibration
                with st.spinner("Calibrating implied volatility surface..."):
                    calibration_result = ImpliedVolatilityCalibrator.calibrate_surface(
                        data_to_calibrate, cal_spot, cal_r
                    )
                
                calibrated_data = calibration_result['calibrated_data']
                rmse = calibration_result['rmse']
                
                st.success(f"Calibration completed! RMSE: ${rmse:.4f}")
                
                # Show calibration statistics
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                
                with col_stat1:
                    avg_iv = calibrated_data['implied_volatility'].mean()
                    st.metric("Average Implied Vol", f"{avg_iv*100:.2f}%")
                
                with col_stat2:
                    iv_std = calibrated_data['implied_volatility'].std()
                    st.metric("Vol Std Dev", f"{iv_std*100:.2f}%")
                
                with col_stat3:
                    min_iv = calibrated_data['implied_volatility'].min()
                    max_iv = calibrated_data['implied_volatility'].max()
                    st.metric("Vol Range", f"{min_iv*100:.1f}% - {max_iv*100:.1f}%")
                
                # Create volatility surface plot
                vol_surface_fig = create_volatility_surface_plot(calibrated_data)
                st.plotly_chart(vol_surface_fig, use_container_width=True)
                
                # Volatility smile/skew analysis
                st.subheader("Volatility Smile Analysis")
                
                # Group by maturity for smile analysis
                maturities = sorted(calibrated_data['time_to_maturity'].unique())
                
                if len(maturities) >= 2:
                    selected_maturity = st.selectbox("Select Maturity for Smile", 
                                                   [f"{T:.2f} years" for T in maturities])
                    maturity_value = float(selected_maturity.split()[0])
                    
                    smile_data = calibrated_data[
                        (calibrated_data['time_to_maturity'] == maturity_value) & 
                        (calibrated_data['option_type'] == 'call')
                    ].sort_values('strike')
                    
                    if len(smile_data) > 0:
                        fig_smile = go.Figure()
                        
                        fig_smile.add_trace(go.Scatter(
                            x=smile_data['strike'] / cal_spot,  # Moneyness
                            y=smile_data['implied_volatility'] * 100,
                            mode='lines+markers',
                            name=f'T={maturity_value:.2f}Y',
                            line=dict(width=3)
                        ))
                        
                        fig_smile.add_vline(x=1.0, line_dash="dash", line_color="red", 
                                          annotation_text="ATM")
                        
                        fig_smile.update_layout(
                            title=f'Volatility Smile (T={maturity_value:.2f} years)',
                            xaxis_title='Moneyness (K/S)',
                            yaxis_title='Implied Volatility (%)',
                            template='plotly_white',
                            height=400
                        )
                        
                        st.plotly_chart(fig_smile, use_container_width=True)
                
                # Show calibrated data table
                with st.expander("View Calibrated Data"):
                    st.dataframe(calibrated_data)
                
                # Export calibrated data
                if st.button("📥 Download Calibrated Data"):
                    csv = calibrated_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="calibrated_vol_surface.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
            
            else:
                st.info("Please generate sample data or upload option chain data to begin calibration")
    
    # Export functionality in sidebar
    with st.sidebar:
        st.divider()
        st.subheader("📥 Export Options")
        
        

        # PDF export: assemble key charts and parameters into a single PDF
        if st.button("📊 Download PDF Report"):
            # PDF generation requires reportlab and matplotlib
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.pdfgen import canvas
                from reportlab.lib.utils import ImageReader
                import matplotlib.pyplot as plt
                import io

                # Gather parameters (use locals fallback)
                params = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'spot_price': locals().get('spot_price', None),
                    'strike_price': locals().get('strike_price', None),
                    'time_to_maturity': locals().get('time_to_maturity', None),
                    'risk_free_rate': locals().get('risk_free_rate', None),
                    'volatility': locals().get('volatility', None),
                    'option_type': locals().get('option_type', None),
                    'model_type': locals().get('model_type', None)
                }

                # Create Price vs Underlying Matplotlib chart
                fig_bufs = []
                try:
                    S0 = float(params['spot_price']) if params['spot_price'] else 100.0
                    K0 = float(params['strike_price']) if params['strike_price'] else S0
                    T0 = float(params['time_to_maturity']) if params['time_to_maturity'] else 30/365.0
                    r0 = float(params['risk_free_rate']) if params['risk_free_rate'] else 0.05
                    vol0 = float(params['volatility']) if params['volatility'] else 0.2
                    opt_type = params.get('option_type', 'Call') or 'Call'

                    price_range = np.linspace(S0 * 0.7, S0 * 1.3, 200)
                    if params.get('model_type', 'Black-Scholes') == 'Binomial Tree':
                        prices = [BinomialTreeEngine.price_vanilla_option(S, K0, T0, r0, vol0, opt_type, n_steps=100) for S in price_range]
                    else:
                        prices = [BlackScholesEngine.price_option(S, K0, T0, r0, vol0, opt_type) for S in price_range]

                    plt.figure(figsize=(6,3))
                    plt.plot(price_range, prices, color='tab:blue', linewidth=2)
                    plt.axvline(S0, color='red', linestyle='--', label='Spot')
                    plt.axvline(K0, color='green', linestyle='--', label='Strike')
                    plt.title(f'Option Price vs Underlying ({opt_type})')
                    plt.xlabel('Underlying Price')
                    plt.ylabel('Option Price')
                    plt.legend()
                    buf1 = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf1, format='png', dpi=150)
                    plt.close()
                    buf1.seek(0)
                    fig_bufs.append(buf1)
                except Exception:
                    # skip chart if creation fails
                    pass

                # Payoff diagram
                try:
                    spot_range = np.linspace(S0 * 0.5, S0 * 1.5, 200)
                    if opt_type.lower() == 'call':
                        payoffs = np.maximum(spot_range - K0, 0)
                    else:
                        payoffs = np.maximum(K0 - spot_range, 0)

                    plt.figure(figsize=(6,3))
                    plt.plot(spot_range, payoffs, color='tab:red', linewidth=2)
                    plt.axvline(K0, color='green', linestyle='--', label='Strike')
                    plt.axvline(S0, color='orange', linestyle='--', label='Spot')
                    plt.title(f'Payoff at Expiry ({opt_type})')
                    plt.xlabel('Underlying Price at Expiry')
                    plt.ylabel('Payoff')
                    plt.legend()
                    buf2 = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf2, format='png', dpi=150)
                    plt.close()
                    buf2.seek(0)
                    fig_bufs.append(buf2)
                except Exception:
                    pass

                # Build PDF
                pdf_buf = io.BytesIO()
                c = canvas.Canvas(pdf_buf, pagesize=letter)
                width, height = letter

                # Title
                c.setFont('Helvetica-Bold', 16)
                c.drawString(40, height - 50, 'Derivatives Analysis Report')
                c.setFont('Helvetica', 10)
                c.drawString(40, height - 70, f"Generated: {params['timestamp']}")

                # Parameters
                y = height - 100
                c.setFont('Helvetica-Bold', 12)
                c.drawString(40, y, 'Parameters:')
                c.setFont('Helvetica', 10)
                y -= 16
                for k, v in params.items():
                    c.drawString(50, y, f"{k}: {v}")
                    y -= 14
                    if y < 120:
                        c.showPage()
                        y = height - 50

                # Insert figures
                for buf in fig_bufs:
                    try:
                        img = ImageReader(buf)
                        c.drawImage(img, 40, y - 220, width=520, height=200)
                        y -= 230
                        if y < 120:
                            c.showPage()
                            y = height - 50
                    except Exception:
                        # continue if image embedding fails
                        continue

                c.showPage()
                c.save()
                pdf_buf.seek(0)

                b64 = base64.b64encode(pdf_buf.read()).decode()
                href = f'<a href="data:application/pdf;base64,{b64}" download="derivatives_analysis_report.pdf">Download PDF Report</a>'
                st.markdown(href, unsafe_allow_html=True)

            except Exception as e:
                st.error('PDF generation requires additional packages: install reportlab and matplotlib')
                st.info('Run: pip install reportlab matplotlib')
                st.exception(e)
    
    # Information panel
    with st.expander("ℹ️ About This Enhanced Dashboard"):
        st.markdown("""
        ### 🚀 Advanced Features:
        
        **Exotic Options:**
        - **Asian Options**: Geometric (analytical) and Arithmetic (Monte Carlo) averaging
        - **Barrier Options**: Up/Down, In/Out variants with tree and MC pricing
        - **Finite Difference Greeks**: For complex payoffs where analytical formulas don't exist
        
        **Implied Volatility Calibration:**
        - **Market Data Integration**: Real option chain processing
        - **Surface Fitting**: RBF and griddata interpolation methods  
        - **Volatility Smile Analysis**: Term structure and skew visualization
        - **Calibration Quality Metrics**: RMSE and goodness-of-fit measures
        
        **Pricing Methods:**
        - **Black-Scholes**: Analytical solutions for European options
        - **Binomial Trees**: American exercise and exotic payoffs
        - **Monte Carlo**: Complex path-dependent options with confidence intervals
        
        **Advanced Analytics:**
        - **Model Comparison**: Convergence analysis across methods
        - **Sensitivity Surfaces**: Multi-dimensional parameter analysis
        - **Risk Management**: Complete Greeks suite with finite differences
        
        ### 📊 Use Cases:
        - Derivatives trading desk analytics
        - Risk management and hedging
        - Academic research and teaching
        - Model validation and benchmarking
        - Volatility surface construction and analysis
        """)

if __name__ == "__main__":
    main()