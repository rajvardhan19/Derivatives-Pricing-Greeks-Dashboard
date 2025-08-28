"""
Pricing engines for vanilla and exotic options
Implements Black-Scholes, Binomial Tree, and Monte Carlo methods
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, Tuple, Optional
import warnings

class BlackScholesEngine:
    """
    Black-Scholes pricing engine for European options
    
    Implements the classic Black-Scholes-Merton formula for vanilla options
    with analytical solutions for calls and puts.
    """
    
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula"""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    @staticmethod
    def _d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula"""
        return BlackScholesEngine._d1(S, K, T, r, sigma) - sigma * np.sqrt(T)
    
    @staticmethod
    def price_european_option(S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str) -> float:
        """
        Price European option using Black-Scholes formula
        
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
        float
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
        
        if sigma <= 0:
            warnings.warn("Volatility must be positive")
            return 0.0
            
        try:
            d1 = BlackScholesEngine._d1(S, K, T, r, sigma)
            d2 = BlackScholesEngine._d2(S, K, T, r, sigma)
            
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            elif option_type.lower() == 'put':
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
                
            return max(price, 0)
            
        except Exception as e:
            warnings.warn(f"Error in Black-Scholes calculation: {e}")
            return 0.0


class BinomialTreeEngine:
    """
    Binomial tree pricing engine for American and exotic options
    
    Implements Cox-Ross-Rubinstein binomial tree model with support for:
    - American exercise
    - Barrier options
    - Asian options (geometric approximation)
    """
    
    @staticmethod
    def price_american_option(S: float, K: float, T: float, r: float, 
                            sigma: float, option_type: str, n_steps: int = 100) -> float:
        """
        Price American option using binomial tree
        
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
        n_steps : int
            Number of time steps in the tree
            
        Returns:
        --------
        float
            Option price
        """
        if T <= 0:
            if option_type.lower() == 'call':
                return max(S - K, 0)
            else:
                return max(K - S, 0)
                
        if n_steps <= 0:
            raise ValueError("Number of steps must be positive")
            
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Check for arbitrage conditions
        if p < 0 or p > 1:
            warnings.warn("Risk-neutral probability outside [0,1] - arbitrage may exist")
        
        # Initialize asset prices at maturity
        prices = np.zeros(n_steps + 1)
        for i in range(n_steps + 1):
            prices[i] = S * (u ** (n_steps - i)) * (d ** i)
        
        # Initialize option values at maturity
        if option_type.lower() == 'call':
            option_values = np.maximum(prices - K, 0)
        elif option_type.lower() == 'put':
            option_values = np.maximum(K - prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Work backwards through the tree
        for j in range(n_steps - 1, -1, -1):
            for i in range(j + 1):
                # Calculate option value if held
                hold_value = np.exp(-r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1])
                
                # Calculate intrinsic value if exercised early
                current_price = S * (u ** (j - i)) * (d ** i)
                if option_type.lower() == 'call':
                    exercise_value = max(current_price - K, 0)
                else:
                    exercise_value = max(K - current_price, 0)
                
                # For American options, take maximum of hold and exercise
                option_values[i] = max(hold_value, exercise_value)
        
        return option_values[0]
    
    @staticmethod
    def price_barrier_option(S: float, K: float, T: float, r: float, sigma: float,
                           barrier: float, option_type: str, barrier_type: str, 
                           n_steps: int = 200) -> float:
        """
        Price barrier options using binomial tree
        
        Parameters:
        -----------
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        barrier : float
            Barrier level
        option_type : str
            'call' or 'put'
        barrier_type : str
            'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        n_steps : int
            Number of time steps
            
        Returns:
        --------
        float
            Barrier option price
        """
        if T <= 0:
            return 0.0
            
        dt = T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp(r * dt) - d) / (u - d)
        
        # Initialize price and barrier tracking arrays
        price_tree = np.zeros((n_steps + 1, n_steps + 1))
        option_tree = np.zeros((n_steps + 1, n_steps + 1))
        barrier_hit = np.zeros((n_steps + 1, n_steps + 1), dtype=bool)
        
        # Set initial price
        price_tree[0, 0] = S
        
        # Check if barrier is already breached
        if barrier_type in ['up-and-out', 'up-and-in']:
            barrier_hit[0, 0] = S >= barrier
        else:  # down barriers
            barrier_hit[0, 0] = S <= barrier
        
        # Build the tree
        for i in range(n_steps):
            for j in range(i + 1):
                if i + 1 <= n_steps:
                    # Up move
                    price_up = price_tree[i, j] * u
                    price_tree[i + 1, j] = price_up
                    
                    # Down move
                    price_down = price_tree[i, j] * d
                    if j + 1 <= n_steps:
                        price_tree[i + 1, j + 1] = price_down
                    
                    # Check barrier conditions
                    if barrier_type in ['up-and-out', 'up-and-in']:
                        barrier_hit[i + 1, j] = barrier_hit[i, j] or (price_up >= barrier)
                        if j + 1 <= n_steps:
                            barrier_hit[i + 1, j + 1] = barrier_hit[i, j] or (price_down >= barrier)
                    else:  # down barriers
                        barrier_hit[i + 1, j] = barrier_hit[i, j] or (price_up <= barrier)
                        if j + 1 <= n_steps:
                            barrier_hit[i + 1, j + 1] = barrier_hit[i, j] or (price_down <= barrier)
        
        # Calculate terminal payoffs
        for j in range(n_steps + 1):
            terminal_price = price_tree[n_steps, j]
            
            # Vanilla payoff
            if option_type.lower() == 'call':
                vanilla_payoff = max(terminal_price - K, 0)
            else:
                vanilla_payoff = max(K - terminal_price, 0)
            
            # Apply barrier condition
            if barrier_type in ['up-and-out', 'down-and-out']:
                # Knock-out: payoff is zero if barrier was hit
                option_tree[n_steps, j] = vanilla_payoff if not barrier_hit[n_steps, j] else 0
            else:  # knock-in
                # Knock-in: payoff only if barrier was hit
                option_tree[n_steps, j] = vanilla_payoff if barrier_hit[n_steps, j] else 0
        
        # Backward induction
        for i in range(n_steps - 1, -1, -1):
            for j in range(i + 1):
                option_tree[i, j] = np.exp(-r * dt) * (
                    p * option_tree[i + 1, j] + (1 - p) * option_tree[i + 1, j + 1]
                )
        
        return option_tree[0, 0]
    
    @staticmethod
    def price_asian_option_geometric(S: float, K: float, T: float, r: float, 
                                   sigma: float, option_type: str) -> float:
        """
        Price geometric Asian option using analytical approximation
        
        For arithmetic Asian options, use Monte Carlo methods instead.
        
        Parameters:
        -----------
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
            
        Returns:
        --------
        float
            Geometric Asian option price
        """
        if T <= 0:
            return 0.0
            
        # Adjusted parameters for geometric Asian option
        sigma_adj = sigma / np.sqrt(3)
        r_adj = (r + sigma**2 / 6) / 2
        
        # Use Black-Scholes formula with adjusted parameters
        d1 = (np.log(S / K) + (r_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
        d2 = d1 - sigma_adj * np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = (S * np.exp((r_adj - r) * T) * norm.cdf(d1) - 
                    K * np.exp(-r * T) * norm.cdf(d2))
        elif option_type.lower() == 'put':
            price = (K * np.exp(-r * T) * norm.cdf(-d2) - 
                    S * np.exp((r_adj - r) * T) * norm.cdf(-d1))
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        return max(price, 0)


class MonteCarloEngine:
    """
    Monte Carlo pricing engine for exotic options
    
    Implements Monte Carlo simulation for path-dependent options including:
    - Arithmetic Asian options
    - Barrier options with continuous monitoring
    - Lookback options
    - Other complex payoffs
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo engine
        
        Parameters:
        -----------
        random_seed : int, optional
            Random seed for reproducible results
        """
        if random_seed is not None:
            np.random.seed(random_seed)
    
    @staticmethod
    def _generate_price_paths(S: float, T: float, r: float, sigma: float,
                            n_simulations: int, n_steps: int) -> np.ndarray:
        """
        Generate geometric Brownian motion price paths
        
        Parameters:
        -----------
        S : float
            Initial price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        n_simulations : int
            Number of simulation paths
        n_steps : int
            Number of time steps per path
            
        Returns:
        --------
        np.ndarray
            Price paths array of shape (n_simulations, n_steps + 1)
        """
        dt = T / n_steps
        
        # Generate random numbers
        Z = np.random.standard_normal((n_simulations, n_steps))
        
        # Initialize price paths
        paths = np.zeros((n_simulations, n_steps + 1))
        paths[:, 0] = S
        
        # Generate paths using geometric Brownian motion
        for t in range(1, n_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
            )
        
        return paths
    
    def price_asian_option_arithmetic(self, S: float, K: float, T: float, r: float,
                                    sigma: float, option_type: str,
                                    n_simulations: int = 10000, 
                                    n_steps: int = 252) -> Dict[str, float]:
        """
        Price arithmetic Asian option using Monte Carlo
        
        Parameters:
        -----------
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        n_simulations : int
            Number of Monte Carlo simulations
        n_steps : int
            Number of time steps per simulation
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing price, standard error, and confidence interval
        """
        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'confidence_interval': (0.0, 0.0)}
        
        # Generate price paths
        paths = self._generate_price_paths(S, T, r, sigma, n_simulations, n_steps)
        
        # Calculate average prices for each path
        avg_prices = np.mean(paths, axis=1)
        
        # Calculate payoffs
        if option_type.lower() == 'call':
            payoffs = np.maximum(avg_prices - K, 0)
        elif option_type.lower() == 'put':
            payoffs = np.maximum(K - avg_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # 95% confidence interval
        confidence_interval = (
            option_price - 1.96 * std_error,
            option_price + 1.96 * std_error
        )
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }
    
    def price_barrier_option(self, S: float, K: float, T: float, r: float, sigma: float,
                           barrier: float, option_type: str, barrier_type: str,
                           n_simulations: int = 10000, 
                           n_steps: int = 252) -> Dict[str, float]:
        """
        Price barrier option using Monte Carlo with continuous monitoring
        
        Parameters:
        -----------
        S : float
            Current underlying price
        K : float
            Strike price
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        barrier : float
            Barrier level
        option_type : str
            'call' or 'put'
        barrier_type : str
            'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
        n_simulations : int
            Number of Monte Carlo simulations
        n_steps : int
            Number of time steps per simulation
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing price, standard error, and confidence interval
        """
        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'confidence_interval': (0.0, 0.0)}
        
        # Generate price paths
        paths = self._generate_price_paths(S, T, r, sigma, n_simulations, n_steps)
        
        payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            path = paths[i, :]
            final_price = path[-1]
            
            # Check barrier condition
            if barrier_type in ['up-and-out', 'up-and-in']:
                barrier_hit = np.any(path >= barrier)
            elif barrier_type in ['down-and-out', 'down-and-in']:
                barrier_hit = np.any(path <= barrier)
            else:
                raise ValueError("Invalid barrier_type")
            
            # Calculate vanilla payoff
            if option_type.lower() == 'call':
                vanilla_payoff = max(final_price - K, 0)
            elif option_type.lower() == 'put':
                vanilla_payoff = max(K - final_price, 0)
            else:
                raise ValueError("option_type must be 'call' or 'put'")
            
            # Apply barrier condition
            if barrier_type in ['up-and-out', 'down-and-out']:
                # Knock-out: payoff is zero if barrier was hit
                payoffs[i] = vanilla_payoff if not barrier_hit else 0
            else:  # knock-in
                # Knock-in: payoff only if barrier was hit
                payoffs[i] = vanilla_payoff if barrier_hit else 0
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # 95% confidence interval
        confidence_interval = (
            option_price - 1.96 * std_error,
            option_price + 1.96 * std_error
        )
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }
    
    def price_lookback_option(self, S: float, K: float, T: float, r: float, sigma: float,
                            option_type: str, lookback_type: str = 'fixed',
                            n_simulations: int = 10000, 
                            n_steps: int = 252) -> Dict[str, float]:
        """
        Price lookback options using Monte Carlo
        
        Parameters:
        -----------
        S : float
            Current underlying price
        K : float
            Strike price (for fixed strike lookback)
        T : float
            Time to maturity
        r : float
            Risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        lookback_type : str
            'fixed' (fixed strike) or 'floating' (floating strike)
        n_simulations : int
            Number of Monte Carlo simulations
        n_steps : int
            Number of time steps per simulation
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing price, standard error, and confidence interval
        """
        if T <= 0:
            return {'price': 0.0, 'std_error': 0.0, 'confidence_interval': (0.0, 0.0)}
        
        # Generate price paths
        paths = self._generate_price_paths(S, T, r, sigma, n_simulations, n_steps)
        
        payoffs = np.zeros(n_simulations)
        
        for i in range(n_simulations):
            path = paths[i, :]
            final_price = path[-1]
            max_price = np.max(path)
            min_price = np.min(path)
            
            if lookback_type == 'fixed':
                if option_type.lower() == 'call':
                    payoffs[i] = max(max_price - K, 0)
                elif option_type.lower() == 'put':
                    payoffs[i] = max(K - min_price, 0)
                else:
                    raise ValueError("option_type must be 'call' or 'put'")
                    
            elif lookback_type == 'floating':
                if option_type.lower() == 'call':
                    payoffs[i] = max(final_price - min_price, 0)
                elif option_type.lower() == 'put':
                    payoffs[i] = max(max_price - final_price, 0)
                else:
                    raise ValueError("option_type must be 'call' or 'put'")
            else:
                raise ValueError("lookback_type must be 'fixed' or 'floating'")
        
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Calculate statistics
        option_price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(n_simulations)
        
        # 95% confidence interval
        confidence_interval = (
            option_price - 1.96 * std_error,
            option_price + 1.96 * std_error
        )
        
        return {
            'price': option_price,
            'std_error': std_error,
            'confidence_interval': confidence_interval
        }