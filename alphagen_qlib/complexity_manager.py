"""
Expression complexity management and penalties for AlphaForge.
Implements complexity scoring, penalties, and filtering for alpha expressions.
"""

import torch
from typing import Dict, List, Optional, Union, Tuple
from alphagen.data.expression import Expression, BinaryOperator, UnaryOperator, Feature
import logging

logger = logging.getLogger(__name__)

class ComplexityManager:
    """
    Manages expression complexity scoring and penalties for alpha factor optimization.

    Implements complexity metrics to prevent overfitting and encourage parsimonious
    alpha factors that generalize well to unseen market data.
    """

    def __init__(
        self,
        max_depth: int = 8,
        max_nodes: int = 20,
        penalty_weight: float = 0.1,
        feature_penalty: float = 0.01,
        operator_penalties: Optional[Dict[str, float]] = None
    ):
        """
        Initialize complexity manager with scoring parameters.

        Args:
            max_depth: Maximum expression tree depth
            max_nodes: Maximum number of nodes in expression tree
            penalty_weight: Weight for complexity penalty in final score
            feature_penalty: Base penalty per feature used
            operator_penalties: Custom penalties for specific operators
        """
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.penalty_weight = penalty_weight
        self.feature_penalty = feature_penalty

        # Default operator complexity penalties
        self.operator_penalties = operator_penalties or {
            'Add': 0.1,
            'Sub': 0.1,
            'Mul': 0.2,
            'Div': 0.3,  # Higher penalty for division (numerical instability)
            'Pow': 0.5,  # High penalty for power operations
            'Log': 0.4,
            'Abs': 0.2,
            'Sign': 0.3,
            'Greater': 0.4,
            'Less': 0.4,
            'Ref': 0.3,  # Time reference operations
            'Rank': 0.6,  # Cross-sectional ranking (expensive)
        }

    def calculate_complexity(self, expr: Expression) -> Dict[str, float]:
        """
        Calculate comprehensive complexity metrics for an expression.

        Args:
            expr: Expression to evaluate

        Returns:
            Dict with complexity metrics: depth, nodes, operator_penalty,
            feature_count, total_score
        """
        metrics = {
            'depth': self._calculate_depth(expr),
            'nodes': self._count_nodes(expr),
            'operator_penalty': self._calculate_operator_penalty(expr),
            'feature_count': self._count_features(expr),
            'feature_penalty': 0.0,
            'total_score': 0.0
        }

        # Calculate feature penalty
        metrics['feature_penalty'] = metrics['feature_count'] * self.feature_penalty

        # Calculate total complexity score
        metrics['total_score'] = (
            metrics['depth'] * 0.2 +
            metrics['nodes'] * 0.1 +
            metrics['operator_penalty'] +
            metrics['feature_penalty']
        )

        return metrics

    def apply_complexity_penalty(
        self,
        ic_score: float,
        expr: Expression
    ) -> Tuple[float, Dict[str, float]]:
        """
        Apply complexity penalty to IC score.

        Args:
            ic_score: Original IC score
            expr: Expression that generated the IC score

        Returns:
            Tuple of (penalized_score, complexity_metrics)
        """
        complexity = self.calculate_complexity(expr)
        penalty = complexity['total_score'] * self.penalty_weight
        penalized_score = ic_score - penalty

        logger.debug(f"IC: {ic_score:.6f}, Complexity: {complexity['total_score']:.4f}, "
                    f"Penalty: {penalty:.6f}, Final: {penalized_score:.6f}")

        return penalized_score, complexity

    def is_valid_complexity(self, expr: Expression) -> bool:
        """
        Check if expression meets complexity constraints.

        Args:
            expr: Expression to validate

        Returns:
            True if expression is within complexity limits
        """
        depth = self._calculate_depth(expr)
        nodes = self._count_nodes(expr)

        if depth > self.max_depth:
            logger.warning(f"Expression depth {depth} exceeds limit {self.max_depth}")
            return False

        if nodes > self.max_nodes:
            logger.warning(f"Expression nodes {nodes} exceeds limit {self.max_nodes}")
            return False

        return True

    def _calculate_depth(self, expr: Expression, current_depth: int = 0) -> int:
        """Calculate maximum depth of expression tree."""
        if isinstance(expr, Feature):
            return current_depth + 1
        elif isinstance(expr, (int, float)):
            return current_depth + 1
        elif isinstance(expr, BinaryOperator):
            left_depth = self._calculate_depth(expr.lhs, current_depth + 1)
            right_depth = self._calculate_depth(expr.rhs, current_depth + 1)
            return max(left_depth, right_depth)
        elif isinstance(expr, UnaryOperator):
            return self._calculate_depth(expr.expr, current_depth + 1)
        else:
            # Handle other expression types
            return current_depth + 1

    def _count_nodes(self, expr: Expression) -> int:
        """Count total number of nodes in expression tree."""
        if isinstance(expr, Feature):
            return 1
        elif isinstance(expr, (int, float)):
            return 1
        elif isinstance(expr, BinaryOperator):
            return 1 + self._count_nodes(expr.lhs) + self._count_nodes(expr.rhs)
        elif isinstance(expr, UnaryOperator):
            return 1 + self._count_nodes(expr.expr)
        else:
            return 1

    def _calculate_operator_penalty(self, expr: Expression) -> float:
        """Calculate penalty based on operator types used."""
        if isinstance(expr, Feature) or isinstance(expr, (int, float)):
            return 0.0

        operator_name = expr.__class__.__name__
        penalty = self.operator_penalties.get(operator_name, 0.2)  # Default penalty

        if isinstance(expr, BinaryOperator):
            penalty += self._calculate_operator_penalty(expr.lhs)
            penalty += self._calculate_operator_penalty(expr.rhs)
        elif isinstance(expr, UnaryOperator):
            penalty += self._calculate_operator_penalty(expr.expr)

        return penalty

    def _count_features(self, expr: Expression) -> int:
        """Count unique features used in expression."""
        features = set()
        self._collect_features(expr, features)
        return len(features)

    def _collect_features(self, expr: Expression, features: set) -> None:
        """Recursively collect all features used in expression."""
        if isinstance(expr, Feature):
            features.add(expr.feature_type)
        elif isinstance(expr, BinaryOperator):
            self._collect_features(expr.lhs, features)
            self._collect_features(expr.rhs, features)
        elif isinstance(expr, UnaryOperator):
            self._collect_features(expr.expr, features)

class ComplexityAwareAlphaPool:
    """
    Extended AlphaPool with complexity-aware evaluation and filtering.
    """

    def __init__(
        self,
        base_pool,
        complexity_manager: ComplexityManager = None,
        enable_complexity_filtering: bool = True
    ):
        """
        Initialize complexity-aware alpha pool.

        Args:
            base_pool: Base AlphaPool instance
            complexity_manager: ComplexityManager instance
            enable_complexity_filtering: Whether to filter by complexity
        """
        self.base_pool = base_pool
        self.complexity_manager = complexity_manager or ComplexityManager()
        self.enable_complexity_filtering = enable_complexity_filtering
        self.complexity_stats = []

    def try_new_expr_with_complexity(self, expr: Expression) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate expression with complexity-aware scoring.

        Args:
            expr: Expression to evaluate

        Returns:
            Tuple of (complexity_adjusted_score, complexity_metrics)
        """
        # Check complexity constraints first
        if self.enable_complexity_filtering:
            if not self.complexity_manager.is_valid_complexity(expr):
                return -1.0, {'total_score': float('inf')}

        try:
            # Get base IC score from original pool
            base_ic = self.base_pool.try_new_expr(expr)

            if base_ic == -1.0:
                return -1.0, {'total_score': 0.0}

            # Apply complexity penalty
            adjusted_score, complexity = self.complexity_manager.apply_complexity_penalty(
                base_ic, expr
            )

            # Store complexity statistics
            complexity['base_ic'] = base_ic
            complexity['adjusted_ic'] = adjusted_score
            complexity['expression'] = str(expr)
            self.complexity_stats.append(complexity)

            return adjusted_score, complexity

        except Exception as e:
            logger.error(f"Error evaluating expression with complexity: {e}")
            return -1.0, {'total_score': 0.0}

    def get_complexity_report(self) -> Dict[str, float]:
        """
        Generate complexity statistics report.

        Returns:
            Dict with complexity statistics across all evaluated expressions
        """
        if not self.complexity_stats:
            return {}

        import numpy as np

        depths = [s['depth'] for s in self.complexity_stats]
        nodes = [s['nodes'] for s in self.complexity_stats]
        penalties = [s['total_score'] for s in self.complexity_stats]

        return {
            'total_expressions': len(self.complexity_stats),
            'avg_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'avg_nodes': np.mean(nodes),
            'max_nodes': np.max(nodes),
            'avg_complexity': np.mean(penalties),
            'max_complexity': np.max(penalties),
            'complexity_std': np.std(penalties)
        }

def create_complexity_aware_pool(base_pool, **complexity_kwargs):
    """
    Factory function to create complexity-aware alpha pool.

    Args:
        base_pool: Base AlphaPool instance
        **complexity_kwargs: Arguments for ComplexityManager

    Returns:
        ComplexityAwareAlphaPool instance
    """
    complexity_manager = ComplexityManager(**complexity_kwargs)
    return ComplexityAwareAlphaPool(base_pool, complexity_manager)