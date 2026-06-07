from .signal_backtest import WalkForwardBacktest, make_signal_fn, treasury_total_return, approximate_10y_duration
from .performance import (
    sharpe_ratio, max_drawdown, annualized_return, annualized_volatility,
    hit_rate, win_loss_ratio, full_metrics, print_tearsheet, regime_performance,
)
from .regime_analysis import (
    label_macro_regimes, label_curve_regimes,
    performance_by_regime, regime_transition_matrix,
    regime_duration_stats, attribution_report,
)

__all__ = [
    "WalkForwardBacktest", "make_signal_fn", "treasury_total_return", "approximate_10y_duration",
    "sharpe_ratio", "max_drawdown", "annualized_return", "annualized_volatility",
    "hit_rate", "win_loss_ratio", "full_metrics", "print_tearsheet", "regime_performance",
    "label_macro_regimes", "label_curve_regimes",
    "performance_by_regime", "regime_transition_matrix",
    "regime_duration_stats", "attribution_report",
]
