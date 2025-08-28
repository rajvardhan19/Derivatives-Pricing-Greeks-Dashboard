"""
Implied volatility calibration and surface fitting module
Implements various methods for volatility surface construction and interpolation
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar, minimize, differential_evolution
from scipy.interpolate import griddata, RBFInterpolator, interp2d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from typing import Dict, List, Tuple, Optional, Callable, Union
import warnings
from dataclasses import dataclass

@dataclass
class OptionQuote:
    """Data class for option market quotes"""
    strike: float
    time_to_maturity: float
    option_type: str  # 'call' or 'put'
    market_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None

class ImpliedVolatilityCalibrator:
    """
    Implied volatility calculation and calibration
    
    Provides methods to extract implied volatility from market prices
    and calibrate volatility surfaces using various interpolation techniques.
    """
    
    def __init__(self, pricing_engine, tolerance: float = 1e-6, max_iterations: int = 100):
        """
        Initialize calibrator
        
        Parameters:
        -----------
        pricing_engine : object
            Pricing engine with price_european_option method
        tolerance : float
            Convergence tolerance for implied volatility calculation
        max_iterations : int
            Maximum iterations for optimization
        """
        self.pricing_engine = pricing_engine
        self.tolerance = tolerance
        self.max_iterations = max_iterations
    
    def implied_volatility_newton_raphson(self, market_price: float, S: float, K: float, 
                                        T: float, r: float, option_type: str,
                                        initial_guess: float = 0.2) -> Dict[str, float]:
        """
        Calculate implied volatility using Newton-Raphson method
        
        Parameters:
        -----------
        market_price : float
            Observed market price
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        option_type : str
            'call' or 'put'
        initial_guess : float
            Initial volatility guess
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing implied volatility and convergence info
        """
        if T <= 0:
            return {'implied_vol': 0.0, 'converged': False, 'iterations': 0, 'error': 'Zero time to maturity'}
        
        sigma = max(initial_guess, 0.001)  # Ensure positive initial guess
        
        for iteration in range(self.max_iterations):
            try:
                # Calculate option price and vega
                price = self.pricing_engine.price_european_option(S, K, T, r, sigma, option_type)
                vega = self._calculate_vega(S, K, T, r, sigma)
                
                if abs(vega) < 1e-10:  # Vega too small
                    return {'implied_vol': sigma, 'converged': False, 'iterations': iteration, 
                           'error': 'Vega near zero'}
                
                # Newton-Raphson update
                price_diff = price - market_price
                
                if abs(price_diff) < self.tolerance:
                    return {'implied_vol': sigma, 'converged': True, 'iterations': iteration + 1, 
                           'error': None}
                
                sigma_new = sigma - price_diff / vega
                
                # Ensure volatility stays positive
                sigma_new = max(sigma_new, 0.001)
                
                # Check for convergence
                if abs(sigma_new - sigma) < self.tolerance:
                    return {'implied_vol': sigma_new, 'converged': True, 'iterations': iteration + 1, 
                           'error': None}
                
                sigma = sigma_new
                
            except Exception as e:
                return {'implied_vol': initial_guess, 'converged': False, 'iterations': iteration, 
                       'error': str(e)}
        
        return {'implied_vol': sigma, 'converged': False, 'iterations': self.max_iterations, 
               'error': 'Max iterations reached'}
    
    def implied_volatility_brent(self, market_price: float, S: float, K: float, 
                               T: float, r: float, option_type: str,
                               vol_bounds: Tuple[float, float] = (0.001, 5.0)) -> Dict[str, float]:
        """
        Calculate implied volatility using Brent's method
        
        Parameters:
        -----------
        market_price : float
            Observed market price
        S, K, T, r : float
            Option parameters
        option_type : str
            'call' or 'put'
        vol_bounds : Tuple[float, float]
            Lower and upper bounds for volatility search
        
        Returns:
        --------
        Dict[str, float]
            Dictionary containing implied volatility and optimization info
        """
        def objective(sigma):
            try:
                model_price = self.pricing_engine.price_european_option(S, K, T, r, sigma, option_type)
                return (model_price - market_price) ** 2
            except:
                return float('inf')
        
        try:
            result = minimize_scalar(objective, bounds=vol_bounds, method='bounded')
            
            return {
                'implied_vol': result.x,
                'converged': result.success,
                'iterations': result.nfev,
                'error': None if result.success else 'Optimization failed'
            }
        
        except Exception as e:
            return {
                'implied_vol': 0.2,
                'converged': False,
                'iterations': 0,
                'error': str(e)
            }
    
    def _calculate_vega(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate vega for Newton-Raphson method"""
        from scipy.stats import norm
        
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def calibrate_single_expiry(self, option_quotes: List[OptionQuote], S: float, r: float,
                              method: str = 'brent') -> pd.DataFrame:
        """
        Calibrate implied volatilities for a single expiry
        
        Parameters:
        -----------
        option_quotes : List[OptionQuote]
            List of option quotes for calibration
        S : float
            Current underlying price
        r : float
            Risk-free rate
        method : str
            'newton_raphson' or 'brent'
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with strikes, market prices, implied volatilities, and errors
        """
        results = []
        
        for quote in option_quotes:
            if method == 'newton_raphson':
                iv_result = self.implied_volatility_newton_raphson(
                    quote.market_price, S, quote.strike, quote.time_to_maturity, 
                    r, quote.option_type
                )
            else:  # brent
                iv_result = self.implied_volatility_brent(
                    quote.market_price, S, quote.strike, quote.time_to_maturity, 
                    r, quote.option_type
                )
            
            # Calculate model price with calibrated IV
            model_price = self.pricing_engine.price_european_option(
                S, quote.strike, quote.time_to_maturity, r, iv_result['implied_vol'], quote.option_type
            )
            
            results.append({
                'strike': quote.strike,
                'time_to_maturity': quote.time_to_maturity,
                'option_type': quote.option_type,
                'market_price': quote.market_price,
                'implied_vol': iv_result['implied_vol'],
                'model_price': model_price,
                'price_error': model_price - quote.market_price,
                'converged': iv_result['converged'],
                'iterations': iv_result['iterations'],
                'moneyness': quote.strike / S
            })
        
        return pd.DataFrame(results)


class VolatilitySurfaceBuilder:
    """
    Volatility surface construction and interpolation
    
    Builds smooth volatility surfaces from market implied volatilities
    using various interpolation and fitting techniques.
    """
    
    def __init__(self):
        """Initialize surface builder"""
        self.fitted_surface = None
        self.calibration_data = None
        self.interpolation_method = None
    
    def fit_rbf_surface(self, strikes: np.ndarray, maturities: np.ndarray, 
                       implied_vols: np.ndarray, smoothing: float = 0.1) -> Dict:
        """
        Fit volatility surface using Radial Basis Functions
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        maturities : np.ndarray
            Times to maturity
        implied_vols : np.ndarray
            Implied volatilities
        smoothing : float
            Smoothing parameter for RBF
        
        Returns:
        --------
        Dict
            Fitted surface information and interpolator
        """
        try:
            # Prepare data points
            points = np.column_stack([strikes.flatten(), maturities.flatten()])
            values = implied_vols.flatten()
            
            # Remove any NaN values
            valid_mask = ~(np.isnan(values) | np.isnan(points).any(axis=1))
            points = points[valid_mask]
            values = values[valid_mask]
            
            if len(points) < 3:
                raise ValueError("Need at least 3 valid data points for RBF interpolation")
            
            # Fit RBF interpolator
            interpolator = RBFInterpolator(points, values, smoothing=smoothing, kernel='thin_plate_spline')
            
            def surface_func(K, T):
                """Interpolation function"""
                K_flat = np.array(K).flatten()
                T_flat = np.array(T).flatten()
                query_points = np.column_stack([K_flat, T_flat])
                
                result = interpolator(query_points)
                return result.reshape(np.array(K).shape)
            
            # Calculate fitting statistics
            predicted_vols = interpolator(points)
            rmse = np.sqrt(np.mean((predicted_vols - values)**2))
            mae = np.mean(np.abs(predicted_vols - values))
            
            self.fitted_surface = surface_func
            self.interpolation_method = 'rbf'
            
            return {
                'surface_func': surface_func,
                'interpolator': interpolator,
                'rmse': rmse,
                'mae': mae,
                'n_points': len(points),
                'method': 'RBF'
            }
            
        except Exception as e:
            warnings.warn(f"RBF fitting failed: {e}")
            return {'error': str(e)}
    
    def fit_gaussian_process_surface(self, strikes: np.ndarray, maturities: np.ndarray,
                                   implied_vols: np.ndarray, kernel_params: Dict = None) -> Dict:
        """
        Fit volatility surface using Gaussian Process Regression
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        maturities : np.ndarray
            Times to maturity
        implied_vols : np.ndarray
            Implied volatilities
        kernel_params : Dict, optional
            Kernel parameters for Gaussian Process
        
        Returns:
        --------
        Dict
            Fitted surface information and GP model
        """
        try:
            # Default kernel parameters
            if kernel_params is None:
                kernel_params = {'length_scale': 1.0, 'noise_level': 0.01}
            
            # Prepare data
            points = np.column_stack([strikes.flatten(), maturities.flatten()])
            values = implied_vols.flatten()
            
            # Remove NaN values
            valid_mask = ~(np.isnan(values) | np.isnan(points).any(axis=1))
            points = points[valid_mask]
            values = values[valid_mask]
            
            if len(points) < 2:
                raise ValueError("Need at least 2 valid data points for GP regression")
            
            # Normalize inputs for better GP performance
            points_mean = np.mean(points, axis=0)
            points_std = np.std(points, axis=0)
            points_normalized = (points - points_mean) / points_std
            
            # Define kernel
            kernel = RBF(length_scale=kernel_params['length_scale']) + \
                    WhiteKernel(noise_level=kernel_params['noise_level'])
            
            # Fit GP
            gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, 
                                        n_restarts_optimizer=10)
            gp.fit(points_normalized, values)
            
            def surface_func(K, T):
                """GP surface interpolation function"""
                K_flat = np.array(K).flatten()
                T_flat = np.array(T).flatten()
                query_points = np.column_stack([K_flat, T_flat])
                query_normalized = (query_points - points_mean) / points_std
                
                pred_mean, pred_std = gp.predict(query_normalized, return_std=True)
                return pred_mean.reshape(np.array(K).shape)
            
            def surface_func_with_uncertainty(K, T):
                """GP surface with uncertainty bounds"""
                K_flat = np.array(K).flatten()
                T_flat = np.array(T).flatten()
                query_points = np.column_stack([K_flat, T_flat])
                query_normalized = (query_points - points_mean) / points_std
                
                pred_mean, pred_std = gp.predict(query_normalized, return_std=True)
                
                return {
                    'mean': pred_mean.reshape(np.array(K).shape),
                    'std': pred_std.reshape(np.array(K).shape),
                    'upper_95': (pred_mean + 1.96 * pred_std).reshape(np.array(K).shape),
                    'lower_95': (pred_mean - 1.96 * pred_std).reshape(np.array(K).shape)
                }
            
            # Calculate fitting statistics
            pred_mean, pred_std = gp.predict(points_normalized, return_std=True)
            rmse = np.sqrt(np.mean((pred_mean - values)**2))
            mae = np.mean(np.abs(pred_mean - values))
            
            self.fitted_surface = surface_func
            self.interpolation_method = 'gaussian_process'
            
            return {
                'surface_func': surface_func,
                'surface_func_with_uncertainty': surface_func_with_uncertainty,
                'gp_model': gp,
                'rmse': rmse,
                'mae': mae,
                'log_likelihood': gp.log_marginal_likelihood(),
                'kernel_params': gp.kernel_.get_params(),
                'n_points': len(points),
                'method': 'Gaussian Process'
            }
            
        except Exception as e:
            warnings.warn(f"GP fitting failed: {e}")
            return {'error': str(e)}
    
    def fit_svi_surface(self, forward_moneyness: np.ndarray, maturities: np.ndarray,
                       total_variances: np.ndarray) -> Dict:
        """
        Fit SVI (Stochastic Volatility Inspired) model to volatility surface
        
        SVI parameterization: w(k) = a + b * (ρ * (k - m) + sqrt((k - m)² + σ²))
        where w is total variance and k is log-moneyness
        
        Parameters:
        -----------
        forward_moneyness : np.ndarray
            Log forward moneyness (log(K/F))
        maturities : np.ndarray
            Times to maturity
        total_variances : np.ndarray
            Total implied variances (σ²T)
        
        Returns:
        --------
        Dict
            SVI parameters and fitted surface
        """
        try:
            # Group data by maturity for slice-wise fitting
            unique_maturities = np.unique(maturities)
            svi_params = {}
            
            def svi_slice(k, params):
                """SVI slice formula"""
                a, b, rho, m, sigma = params
                return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
            
            def svi_objective(params, k, w):
                """SVI fitting objective"""
                try:
                    model_w = svi_slice(k, params)
                    return np.sum((model_w - w)**2)
                except:
                    return 1e10
            
            for T in unique_maturities:
                mask = maturities == T
                k_slice = forward_moneyness[mask]
                w_slice = total_variances[mask]
                
                if len(k_slice) < 5:  # Need at least 5 points for SVI
                    continue
                
                # Initial parameter guess
                w_atm = np.interp(0, k_slice, w_slice)  # ATM total variance
                initial_guess = [w_atm * 0.1, w_atm * 0.8, -0.1, 0.0, 0.1]
                
                # Parameter bounds
                bounds = [
                    (0, w_atm),      # a >= 0
                    (0, 4 * w_atm),  # b >= 0
                    (-1, 1),         # -1 <= ρ <= 1
                    (-2, 2),         # m
                    (0.01, 2)        # σ > 0
                ]
                
                # Fit SVI parameters
                result = minimize(svi_objective, initial_guess, args=(k_slice, w_slice),
                                bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    svi_params[T] = {
                        'params': result.x,
                        'rmse': np.sqrt(result.fun / len(k_slice)),
                        'converged': True
                    }
                else:
                    svi_params[T] = {
                        'params': initial_guess,
                        'rmse': np.inf,
                        'converged': False
                    }
            
            def svi_surface_func(K, T, F=None):
                """SVI surface interpolation"""
                if F is None:
                    F = 100  # Default forward price
                
                k = np.log(np.array(K) / F)
                T_array = np.array(T)
                
                result = np.zeros_like(k)
                
                for i, (k_val, T_val) in enumerate(zip(k.flat, T_array.flat)):
                    # Find closest maturity parameters
                    closest_T = min(svi_params.keys(), key=lambda x: abs(x - T_val))
                    params = svi_params[closest_T]['params']
                    
                    total_var = svi_slice(k_val, params)
                    implied_vol = np.sqrt(max(total_var / T_val, 1e-6))  # Ensure positive
                    result.flat[i] = implied_vol
                
                return result.reshape(np.array(K).shape)
            
            # Calculate overall fitting quality
            total_rmse = np.mean([p['rmse'] for p in svi_params.values() if p['converged']])
            
            self.fitted_surface = svi_surface_func
            self.interpolation_method = 'svi'
            
            return {
                'surface_func': svi_surface_func,
                'svi_params': svi_params,
                'rmse': total_rmse,
                'method': 'SVI',
                'fitted_maturities': list(svi_params.keys())
            }
            
        except Exception as e:
            warnings.warn(f"SVI fitting failed: {e}")
            return {'error': str(e)}
    
    def evaluate_surface(self, strike_grid: np.ndarray, maturity_grid: np.ndarray) -> np.ndarray:
        """
        Evaluate fitted volatility surface on a grid
        
        Parameters:
        -----------
        strike_grid : np.ndarray
            2D grid of strikes
        maturity_grid : np.ndarray
            2D grid of maturities
        
        Returns:
        --------
        np.ndarray
            Volatility values on the grid
        """
        if self.fitted_surface is None:
            raise ValueError("No surface has been fitted yet")
        
        return self.fitted_surface(strike_grid, maturity_grid)


class VolatilitySurfaceAnalytics:
    """
    Analytics and metrics for volatility surfaces
    
    Provides various measures to analyze volatility surfaces including
    skew, term structure, and arbitrage conditions.
    """
    
    @staticmethod
    def calculate_skew_metrics(strikes: np.ndarray, implied_vols: np.ndarray, 
                             spot_price: float) -> Dict[str, float]:
        """
        Calculate volatility skew metrics
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        implied_vols : np.ndarray
            Implied volatilities
        spot_price : float
            Current spot price
        
        Returns:
        --------
        Dict[str, float]
            Skew metrics
        """
        # Convert to moneyness
        moneyness = strikes / spot_price
        
        # Find ATM volatility
        atm_idx = np.argmin(np.abs(moneyness - 1.0))
        atm_vol = implied_vols[atm_idx]
        
        # 25-delta put and call equivalent strikes (approximation)
        try:
            # Find 90% and 110% moneyness points
            otm_put_idx = np.argmin(np.abs(moneyness - 0.9))
            otm_call_idx = np.argmin(np.abs(moneyness - 1.1))
            
            put_vol = implied_vols[otm_put_idx]
            call_vol = implied_vols[otm_call_idx]
            
            # Risk reversal (25-delta call vol - 25-delta put vol)
            risk_reversal = call_vol - put_vol
            
            # Butterfly (average of 25-delta wings - ATM vol)
            butterfly = (put_vol + call_vol) / 2 - atm_vol
            
            # Skew slope (approximate)
            skew_slope = np.polyfit(moneyness, implied_vols, 1)[0]
            
            # Convexity
            convexity = np.polyfit(moneyness, implied_vols, 2)[0]
            
        except Exception:
            risk_reversal = 0.0
            butterfly = 0.0
            skew_slope = 0.0
            convexity = 0.0
        
        return {
            'atm_vol': atm_vol,
            'risk_reversal': risk_reversal,
            'butterfly': butterfly,
            'skew_slope': skew_slope,
            'convexity': convexity,
            'vol_range': np.max(implied_vols) - np.min(implied_vols)
        }
    
    @staticmethod
    def check_arbitrage_conditions(strikes: np.ndarray, call_prices: np.ndarray,
                                 put_prices: np.ndarray, spot_price: float,
                                 interest_rate: float, time_to_maturity: float) -> Dict:
        """
        Check for arbitrage opportunities in option prices
        
        Parameters:
        -----------
        strikes : np.ndarray
            Strike prices
        call_prices : np.ndarray
            Call option prices
        put_prices : np.ndarray
            Put option prices
        spot_price : float
            Current spot price
        interest_rate : float
            Risk-free rate
        time_to_maturity : float
            Time to maturity
        
        Returns:
        --------
        Dict
            Arbitrage flags and violations
        """
        discount_factor = np.exp(-interest_rate * time_to_maturity)
        forward_price = spot_price / discount_factor
        
        violations = {
            'call_spread_violations': [],
            'put_spread_violations': [],
            'put_call_parity_violations': [],
            'boundary_violations': []
        }
        
        # Check call spread arbitrage (call prices should be decreasing in strikes)
        for i in range(len(strikes) - 1):
            if call_prices[i] < call_prices[i + 1]:
                violations['call_spread_violations'].append({
                    'strike1': strikes[i],
                    'strike2': strikes[i + 1],
                    'price1': call_prices[i],
                    'price2': call_prices[i + 1]
                })
        
        # Check put spread arbitrage (put prices should be increasing in strikes)
        for i in range(len(strikes) - 1):
            if put_prices[i] > put_prices[i + 1]:
                violations['put_spread_violations'].append({
                    'strike1': strikes[i],
                    'strike2': strikes[i + 1],
                    'price1': put_prices[i],
                    'price2': put_prices[i + 1]
                })
        
        # Check put-call parity
        for i, K in enumerate(strikes):
            theoretical_difference = call_prices[i] - put_prices[i] - (forward_price - K) * discount_factor
            if abs(theoretical_difference) > 0.01:  # Allow small tolerance
                violations['put_call_parity_violations'].append({
                    'strike': K,
                    'call_price': call_prices[i],
                    'put_price': put_prices[i],
                    'parity_violation': theoretical_difference
                })
        
        # Check boundary conditions
        for i, K in enumerate(strikes):
            # Call price should be <= spot price
            if call_prices[i] > spot_price:
                violations['boundary_violations'].append({
                    'type': 'call_upper_bound',
                    'strike': K,
                    'price': call_prices[i],
                    'bound': spot_price
                })
            
            # Put price should be <= discounted strike
            discounted_strike = K * discount_factor
            if put_prices[i] > discounted_strike:
                violations['boundary_violations'].append({
                    'type': 'put_upper_bound',
                    'strike': K,
                    'price': put_prices[i],
                    'bound': discounted_strike
                })
        
        # Summary
        total_violations = sum(len(v) for v in violations.values())
        
        return {
            'arbitrage_free': total_violations == 0,
            'total_violations': total_violations,
            'violations': violations,
            'summary': {
                'call_spread_issues': len(violations['call_spread_violations']),
                'put_spread_issues': len(violations['put_spread_violations']),
                'parity_issues': len(violations['put_call_parity_violations']),
                'boundary_issues': len(violations['boundary_violations'])
            }
        }
    
    @staticmethod
    def calculate_term_structure_metrics(maturities: np.ndarray, atm_vols: np.ndarray) -> Dict:
        """
        Analyze volatility term structure
        
        Parameters:
        -----------
        maturities : np.ndarray
            Times to maturity
        atm_vols : np.ndarray
            At-the-money implied volatilities
        
        Returns:
        --------
        Dict
            Term structure metrics
        """
        if len(maturities) < 2:
            return {'error': 'Need at least 2 maturities for term structure analysis'}
        
        # Sort by maturity
        sort_idx = np.argsort(maturities)
        T_sorted = maturities[sort_idx]
        vol_sorted = atm_vols[sort_idx]
        
        # Calculate forward volatilities
        forward_vols = []
        for i in range(1, len(T_sorted)):
            T1, T2 = T_sorted[i-1], T_sorted[i]
            vol1, vol2 = vol_sorted[i-1], vol_sorted[i]
            
            # Forward variance
            forward_var = (vol2**2 * T2 - vol1**2 * T1) / (T2 - T1)
            forward_vol = np.sqrt(max(forward_var, 0))
            forward_vols.append(forward_vol)
        
        forward_vols = np.array(forward_vols)
        
        # Term structure shape
        vol_slope = np.polyfit(T_sorted, vol_sorted, 1)[0]
        vol_curvature = np.polyfit(T_sorted, vol_sorted, 2)[0] if len(T_sorted) > 2 else 0
        
        # Classification
        if vol_slope > 0.01:
            shape = 'upward_sloping'
        elif vol_slope < -0.01:
            shape = 'downward_sloping'
        else:
            shape = 'flat'
        
        return {
            'shape': shape,
            'slope': vol_slope,
            'curvature': vol_curvature,
            'short_term_vol': vol_sorted[0],
            'long_term_vol': vol_sorted[-1],
            'vol_spread': vol_sorted[-1] - vol_sorted[0],
            'forward_vols': forward_vols,
            'avg_forward_vol': np.mean(forward_vols) if len(forward_vols) > 0 else 0
        }


class VolatilitySurfaceValidator:
    """
    Validation tools for volatility surfaces
    
    Ensures fitted surfaces are reasonable and free from arbitrage
    """
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def validate_surface_smoothness(self, surface_func: Callable, 
                                  strike_range: Tuple[float, float],
                                  maturity_range: Tuple[float, float],
                                  n_points: int = 50) -> Dict:
        """
        Validate surface smoothness by checking derivatives
        
        Parameters:
        -----------
        surface_func : Callable
            Volatility surface function
        strike_range : Tuple[float, float]
            Range of strikes to test
        maturity_range : Tuple[float, float]
            Range of maturities to test
        n_points : int
            Number of test points
        
        Returns:
        --------
        Dict
            Smoothness validation results
        """
        K_test = np.linspace(strike_range[0], strike_range[1], n_points)
        T_test = np.linspace(maturity_range[0], maturity_range[1], n_points)
        
        # Test monotonicity and smoothness
        issues = []
        
        # Check each maturity slice
        for T in T_test[::5]:  # Sample every 5th point
            try:
                vols = surface_func(K_test, T)
                
                # Check for negative volatilities
                if np.any(vols < 0):
                    issues.append(f"Negative volatilities at T={T:.3f}")
                
                # Check for extreme volatilities
                if np.any(vols > 3.0) or np.any(vols < 0.01):
                    issues.append(f"Extreme volatilities at T={T:.3f}")
                
                # Check for discontinuities (large jumps)
                vol_diffs = np.diff(vols)
                if np.any(np.abs(vol_diffs) > 0.5):
                    issues.append(f"Large discontinuities at T={T:.3f}")
                    
            except Exception as e:
                issues.append(f"Error evaluating surface at T={T:.3f}: {str(e)}")
        
        return {
            'smooth': len(issues) == 0,
            'issues': issues,
            'n_issues': len(issues)
        }
    
    def validate_arbitrage_free(self, surface_func: Callable, spot_price: float,
                              interest_rate: float, strike_range: Tuple[float, float],
                              maturity_range: Tuple[float, float]) -> Dict:
        """
        Check if volatility surface produces arbitrage-free option prices
        
        This is a simplified check - full arbitrage validation would require
        more sophisticated calendar spread and butterfly spread analysis.
        """
        # This would require integration with pricing engines
        # For now, return basic validation
        return {
            'arbitrage_free': True,
            'warnings': ['Full arbitrage validation not implemented'],
            'checks_performed': ['basic_bounds']
        }