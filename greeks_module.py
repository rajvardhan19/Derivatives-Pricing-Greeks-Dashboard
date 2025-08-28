"""
Greeks calculation module for options
Implements analytical and finite difference methods for risk sensitivities
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Callable, Optional
import warnings

class AnalyticalGreeks:
    """
    Analytical Greeks calculation using Black-Scholes formulas
    
    Provides closed-form solutions for European option sensitivities:
    - Delta: Price sensitivity to underlying movement
    - Gamma: Delta sensitivity (convexity measure)
    - Vega: Volatility sensitivity
    - Theta: Time decay
    - Rho: Interest rate sensitivity
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = AnalyticalGreeks._d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Vega - volatility sensitivity (same for calls and puts)"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = AnalyticalGreeks._d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% volatility change
    
    @staticmethod
    def theta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Theta - time decay"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = AnalyticalGreeks._d1(S, K, T, r, sigma)
        d2 = AnalyticalGreeks._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        else:
            theta = ((-S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        
        return theta
    
    @staticmethod
    def rho(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Rho - interest rate sensitivity"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d2 = AnalyticalGreeks._d2(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100


class FiniteDifferenceGreeks:
    """
    Finite difference Greeks calculation for exotic options
    
    Uses numerical differentiation to calculate Greeks when analytical
    formulas are not available or complex to derive.
    """
    
    def __init__(self, epsilon: Dict[str, float] = None):
        """
        Initialize finite difference calculator
        
        Parameters:
        -----------
        epsilon : Dict[str, float], optional
            Step sizes for different parameters
        """
        self.epsilon = epsilon or {
            'spot': 0.01,      # 1% for underlying price
            'vol': 0.01,       # 1% for volatility  
            'rate': 0.0001,    # 1 basis point for interest rate
            'time': 1/365      # 1 day for time
        }
    
    def delta(self, pricing_func: Callable, S: float, K: float, T: float, 
              r: float, sigma: float, option_type: str, **kwargs) -> float:
        """
        Calculate Delta using central difference
        
        Parameters:
        -----------
        pricing_func : Callable
            Option pricing function
        S, K, T, r, sigma, option_type : option parameters
        **kwargs : additional arguments for pricing function
        
        Returns:
        --------
        float
            Delta value
        """
        eps = self.epsilon['spot'] * S
        
        price_up = pricing_func(S + eps, K, T, r, sigma, option_type, **kwargs)
        price_down = pricing_func(S - eps, K, T, r, sigma, option_type, **kwargs)
        
        return (price_up - price_down) / (2 * eps)
    
    def gamma(self, pricing_func: Callable, S: float, K: float, T: float,
              r: float, sigma: float, option_type: str, **kwargs) -> float:
        """Calculate Gamma using central difference"""
        eps = self.epsilon['spot'] * S
        
        price_up = pricing_func(S + eps, K, T, r, sigma, option_type, **kwargs)
        price_center = pricing_func(S, K, T, r, sigma, option_type, **kwargs)
        price_down = pricing_func(S - eps, K, T, r, sigma, option_type, **kwargs)
        
        return (price_up - 2 * price_center + price_down) / (eps**2)
    
    def vega(self, pricing_func: Callable, S: float, K: float, T: float,
             r: float, sigma: float, option_type: str, **kwargs) -> float:
        """Calculate Vega using central difference"""
        eps = self.epsilon['vol'] * sigma
        
        price_up = pricing_func(S, K, T, r, sigma + eps, option_type, **kwargs)
        price_down = pricing_func(S, K, T, r, sigma - eps, option_type, **kwargs)
        
        return (price_up - price_down) / (2 * eps) / 100  # Per 1% volatility change
    
    def theta(self, pricing_func: Callable, S: float, K: float, T: float,
              r: float, sigma: float, option_type: str, **kwargs) -> float:
        """Calculate Theta using forward difference"""
        eps = self.epsilon['time']
        
        if T <= eps:
            return 0.0
        
        price_now = pricing_func(S, K, T, r, sigma, option_type, **kwargs)
        price_later = pricing_func(S, K, T - eps, r, sigma, option_type, **kwargs)
        
        return -(price_later - price_now) / eps
    
    def rho(self, pricing_func: Callable, S: float, K: float, T: float,
            r: float, sigma: float, option_type: str, **kwargs) -> float:
        """Calculate Rho using central difference"""
        eps = self.epsilon['rate']
        
        price_up = pricing_func(S, K, T, r + eps, sigma, option_type, **kwargs)
        price_down = pricing_func(S, K, T, r - eps, sigma, option_type, **kwargs)
        
        return (price_up - price_down) / (2 * eps) / 100  # Per 1% rate change
    
    def calculate_all_greeks(self, pricing_func: Callable, S: float, K: float, T: float,
                           r: float, sigma: float, option_type: str, **kwargs) -> Dict[str, float]:
        """
        Calculate all Greeks using finite differences
        
        Parameters:
        -----------
        pricing_func : Callable
            Option pricing function
        S, K, T, r, sigma, option_type : option parameters
        **kwargs : additional arguments for pricing function
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all Greeks
        """
        try:
            return {
                'delta': self.delta(pricing_func, S, K, T, r, sigma, option_type, **kwargs),
                'gamma': self.gamma(pricing_func, S, K, T, r, sigma, option_type, **kwargs),
                'vega': self.vega(pricing_func, S, K, T, r, sigma, option_type, **kwargs),
                'theta': self.theta(pricing_func, S, K, T, r, sigma, option_type, **kwargs),
                'rho': self.rho(pricing_func, S, K, T, r, sigma, option_type, **kwargs)
            }
        except Exception as e:
            warnings.warn(f"Error in finite difference Greeks calculation: {e}")
            return {greek: 0.0 for greek in ['delta', 'gamma', 'vega', 'theta', 'rho']}


class GreeksPortfolio:
    """
    Portfolio-level Greeks calculation and risk management
    
    Manages Greeks for multiple positions and provides portfolio-level
    risk metrics and hedging calculations.
    """
    
    def __init__(self):
        """Initialize empty portfolio"""
        self.positions = []
    
    def add_position(self, position: Dict) -> None:
        """
        Add position to portfolio
        
        Parameters:
        -----------
        position : Dict
            Position dictionary containing:
            - quantity: number of contracts
            - greeks: dictionary of Greeks values
            - option_type: 'call' or 'put'
            - underlying: underlying asset symbol
            - expiry: expiration date
        """
        required_keys = ['quantity', 'greeks', 'option_type', 'underlying']
        
        if not all(key in position for key in required_keys):
            raise ValueError(f"Position must contain keys: {required_keys}")
        
        self.positions.append(position.copy())
    
    def remove_position(self, index: int) -> None:
        """Remove position by index"""
        if 0 <= index < len(self.positions):
            self.positions.pop(index)
        else:
            raise IndexError("Position index out of range")
    
    def calculate_portfolio_greeks(self, by_underlying: bool = False) -> Dict:
        """
        Calculate portfolio-level Greeks
        
        Parameters:
        -----------
        by_underlying : bool
            If True, return Greeks grouped by underlying asset
        
        Returns:
        --------
        Dict
            Portfolio Greeks (total or by underlying)
        """
        if not self.positions:
            return {}
        
        if by_underlying:
            # Group by underlying asset
            portfolio_greeks = {}
            
            for position in self.positions:
                underlying = position['underlying']
                quantity = position['quantity']
                greeks = position['greeks']
                
                if underlying not in portfolio_greeks:
                    portfolio_greeks[underlying] = {
                        'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 
                        'theta': 0.0, 'rho': 0.0
                    }
                
                for greek, value in greeks.items():
                    if greek in portfolio_greeks[underlying]:
                        portfolio_greeks[underlying][greek] += quantity * value
            
            return portfolio_greeks
        
        else:
            # Total portfolio Greeks
            total_greeks = {
                'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 
                'theta': 0.0, 'rho': 0.0
            }
            
            for position in self.positions:
                quantity = position['quantity']
                greeks = position['greeks']
                
                for greek, value in greeks.items():
                    if greek in total_greeks:
                        total_greeks[greek] += quantity * value
            
            return total_greeks
    
    def calculate_hedge_ratios(self, hedge_instruments: List[Dict]) -> Dict:
        """
        Calculate hedge ratios to neutralize portfolio Greeks
        
        Parameters:
        -----------
        hedge_instruments : List[Dict]
            List of available hedge instruments with their Greeks
            Each dict should contain: 'symbol', 'greeks', 'price'
        
        Returns:
        --------
        Dict
            Hedge ratios for each instrument
        """
        portfolio_greeks = self.calculate_portfolio_greeks()
        
        # Simple delta hedging example
        hedge_ratios = {}
        
        for instrument in hedge_instruments:
            symbol = instrument['symbol']
            inst_greeks = instrument['greeks']
            
            # Calculate hedge ratio to neutralize delta
            if 'delta' in inst_greeks and inst_greeks['delta'] != 0:
                delta_hedge_ratio = -portfolio_greeks.get('delta', 0) / inst_greeks['delta']
                hedge_ratios[symbol] = {
                    'delta_hedge_ratio': delta_hedge_ratio,
                    'contracts_needed': int(round(delta_hedge_ratio))
                }
        
        return hedge_ratios
    
    def risk_report(self) -> Dict:
        """
        Generate comprehensive risk report
        
        Returns:
        --------
        Dict
            Risk metrics and exposure analysis
        """
        if not self.positions:
            return {'message': 'No positions in portfolio'}
        
        portfolio_greeks = self.calculate_portfolio_greeks()
        greeks_by_underlying = self.calculate_portfolio_greeks(by_underlying=True)
        
        # Calculate risk metrics
        total_positions = len(self.positions)
        total_delta = portfolio_greeks.get('delta', 0)
        total_gamma = portfolio_greeks.get('gamma', 0)
        total_vega = portfolio_greeks.get('vega', 0)
        total_theta = portfolio_greeks.get('theta', 0)
        
        # Risk categorization
        delta_risk = 'High' if abs(total_delta) > 100 else 'Medium' if abs(total_delta) > 50 else 'Low'
        gamma_risk = 'High' if abs(total_gamma) > 10 else 'Medium' if abs(total_gamma) > 5 else 'Low'
        vega_risk = 'High' if abs(total_vega) > 50 else 'Medium' if abs(total_vega) > 25 else 'Low'
        
        return {
            'portfolio_summary': {
                'total_positions': total_positions,
                'underlyings': list(greeks_by_underlying.keys()),
                'total_greeks': portfolio_greeks
            },
            'risk_assessment': {
                'delta_risk': delta_risk,
                'gamma_risk': gamma_risk,
                'vega_risk': vega_risk,
                'daily_theta_decay': total_theta
            },
            'exposure_by_underlying': greeks_by_underlying,
            'recommendations': self._generate_risk_recommendations(portfolio_greeks)
        }
    
    def _generate_risk_recommendations(self, greeks: Dict[str, float]) -> List[str]:
        """Generate risk management recommendations"""
        recommendations = []
        
        delta = greeks.get('delta', 0)
        gamma = greeks.get('gamma', 0)
        vega = greeks.get('vega', 0)
        theta = greeks.get('theta', 0)
        
        if abs(delta) > 100:
            recommendations.append(f"High delta exposure ({delta:.2f}). Consider delta hedging with underlying or futures.")
        
        if abs(gamma) > 10:
            recommendations.append(f"High gamma exposure ({gamma:.4f}). Monitor for large price moves.")
        
        if abs(vega) > 50:
            recommendations.append(f"High vega exposure ({vega:.2f}). Hedge volatility risk if needed.")
        
        if theta < -10:
            recommendations.append(f"Significant time decay ({theta:.2f}/day). Monitor theta burn.")
        
        if not recommendations:
            recommendations.append("Portfolio Greeks within acceptable ranges.")
        
        return recommendations


class GreeksVisualization:
    """
    Visualization utilities for Greeks analysis
    
    Provides methods to create visualizations for Greeks behavior
    and sensitivity analysis.
    """
    
    @staticmethod
    def generate_greeks_surface_data(S_range: np.ndarray, T_range: np.ndarray,
                                   K: float, r: float, sigma: float, 
                                   option_type: str, greek: str) -> np.ndarray:
        """
        Generate data for 3D Greeks surface plots
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of underlying prices
        T_range : np.ndarray
            Range of times to maturity
        K : float
            Strike price
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        greek : str
            Greek to calculate ('delta', 'gamma', 'vega', 'theta', 'rho')
        
        Returns:
        --------
        np.ndarray
            2D array of Greek values
        """
        greek_values = np.zeros((len(T_range), len(S_range)))
        
        for i, T in enumerate(T_range):
            for j, S in enumerate(S_range):
                try:
                    if greek == 'delta':
                        value = AnalyticalGreeks.delta(S, K, T, r, sigma, option_type)
                    elif greek == 'gamma':
                        value = AnalyticalGreeks.gamma(S, K, T, r, sigma)
                    elif greek == 'vega':
                        value = AnalyticalGreeks.vega(S, K, T, r, sigma)
                    elif greek == 'theta':
                        value = AnalyticalGreeks.theta(S, K, T, r, sigma, option_type)
                    elif greek == 'rho':
                        value = AnalyticalGreeks.rho(S, K, T, r, sigma, option_type)
                    else:
                        value = 0.0
                    
                    greek_values[i, j] = value
                    
                except Exception:
                    greek_values[i, j] = 0.0
        
        return greek_values
    
    @staticmethod
    def calculate_greeks_profile(S_range: np.ndarray, K: float, T: float, r: float,
                               sigma: float, option_type: str) -> Dict[str, np.ndarray]:
        """
        Calculate Greeks profile across underlying price range
        
        Parameters:
        -----------
        S_range : np.ndarray
            Range of underlying prices
        K, T, r, sigma, option_type : option parameters
        
        Returns:
        --------
        Dict[str, np.ndarray]
            Dictionary of Greeks arrays
        """
        greeks_profile = {
            'delta': np.zeros_like(S_range),
            'gamma': np.zeros_like(S_range),
            'vega': np.zeros_like(S_range),
            'theta': np.zeros_like(S_range),
            'rho': np.zeros_like(S_range)
        }
        
        for i, S in enumerate(S_range):
            try:
                greeks = AnalyticalGreeks.calculate_all_greeks(S, K, T, r, sigma, option_type)
                
                for greek in greeks_profile:
                    greeks_profile[greek][i] = greeks.get(greek, 0.0)
                    
            except Exception:
                for greek in greeks_profile:
                    greeks_profile[greek][i] = 0.0
        
        return greeks_profile
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula"""
        return AnalyticalGreeks._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def calculate_all_greeks(S: float, K: float, T: float, r: float, 
                           sigma: float, option_type: str) -> Dict[str, float]:
        """
        Calculate all Greeks for European options
        
        Parameters:
        -----------
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity (in years)
        r : float
            Risk-free interest rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing all Greeks
        """
        if T <= 0:
            return {
                'delta': 0.0,
                'gamma': 0.0,
                'vega': 0.0,
                'theta': 0.0,
                'rho': 0.0,
                'lambda': 0.0,  # elasticity
                'epsilon': 0.0,  # dividend sensitivity
                'vanna': 0.0,   # cross-derivative of delta and vega
                'charm': 0.0,   # time decay of delta
                'vomma': 0.0,   # volatility convexity
                'ultima': 0.0   # sensitivity of vomma to volatility
            }
        
        if sigma <= 0:
            warnings.warn("Volatility must be positive")
            return {greek: 0.0 for greek in ['delta', 'gamma', 'vega', 'theta', 'rho', 
                                           'lambda', 'epsilon', 'vanna', 'charm', 'vomma', 'ultima']}
        
        try:
            d1 = AnalyticalGreeks._d1(S, K, T, r, sigma)
            d2 = AnalyticalGreeks._d2(S, K, T, r, sigma)
            
            # Standard normal PDF and CDF
            phi_d1 = norm.pdf(d1)
            Phi_d1 = norm.cdf(d1)
            Phi_d2 = norm.cdf(d2)
            
            # First-order Greeks
            if option_type.lower() == 'call':
                delta = Phi_d1
                theta = ((-S * phi_d1 * sigma) / (2 * np.sqrt(T)) - 
                        r * K * np.exp(-r * T) * Phi_d2) / 365
                rho = K * T * np.exp(-r * T) * Phi_d2 / 100
            elif option_type.lower() == 'put':
                delta = Phi_d1 - 1
                theta = ((-S * phi_d1 * sigma) / (2 * np.sqrt(T)) + 
                        r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            # Second-order Greeks (same for calls and puts)
            gamma = phi_d1 / (S * sigma * np.sqrt(T))
            vega = S * phi_d1 * np.sqrt(T) / 100  # Per 1% volatility change
            
            # Additional Greeks
            from .pricing import BlackScholesEngine
            option_price = BlackScholesEngine.price_european_option(S, K, T, r, sigma, option_type)
            
            # Lambda (elasticity)
            lambda_greek = delta * S / option_price if option_price != 0 else 0
            
            # Epsilon (dividend sensitivity) - assuming zero dividends
            epsilon = 0.0
            
            # Second-order cross Greeks
            vanna = -phi_d1 * d2 / sigma / 100  # d(delta)/d(sigma)
            charm = -phi_d1 * (2 * r * T - d2 * sigma * np.sqrt(T)) / (2 * T * sigma * np.sqrt(T)) / 365
            
            # Third-order Greeks
            vomma = vega * d1 * d2 / sigma / 100  # d(vega)/d(sigma)
            ultima = -vega * (d1 * d2 * (1 - d1 * d2) + d1**2 + d2**2) / (sigma**2) / 10000
            
            return {
                'delta': delta,
                'gamma': gamma,
                'vega': vega,
                'theta': theta,
                'rho': rho,
                'lambda': lambda_greek,
                'epsilon': epsilon,
                'vanna': vanna,
                'charm': charm,
                'vomma': vomma,
                'ultima': ultima
            }
            
        except Exception as e:
            warnings.warn(f"Error in Greeks calculation: {e}")
            return {greek: 0.0 for greek in ['delta', 'gamma', 'vega', 'theta', 'rho', 
                                           'lambda', 'epsilon', 'vanna', 'charm', 'vomma', 'ultima']}
    
    @staticmethod
    def delta(S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Delta - price sensitivity to underlying movement"""
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = AnalyticalGreeks._d1(S, K, T, r, sigma)
        
        if option_type.lower() == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate Gamma - Delta sensitivity (same for calls and puts)"""
        if T <= 0 or sigma <= 0: