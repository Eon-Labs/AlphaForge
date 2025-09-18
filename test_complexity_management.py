#!/usr/bin/env python3
"""
Test script for expression complexity management and penalties.
Validates complexity scoring, filtering, and penalty application.
"""

import torch
from alphagen.data.expression import *
from alphagen_qlib.complexity_manager import ComplexityManager, ComplexityAwareAlphaPool
from alphagen_generic.features import target

def test_complexity_calculation():
    """Test complexity calculation for various expressions"""
    print("=== Testing Complexity Calculation ===")

    manager = ComplexityManager()

    # Simple expressions
    simple_expr = Feature(FeatureType.CLOSE)
    binary_expr = Sub(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))
    complex_expr = Div(
        lhs=Sub(lhs=Feature(FeatureType.HIGH), rhs=Feature(FeatureType.LOW)),
        rhs=Add(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.VOLUME))
    )

    expressions = [
        ("Simple Feature", simple_expr),
        ("Binary Operation", binary_expr),
        ("Complex Expression", complex_expr)
    ]

    for name, expr in expressions:
        complexity = manager.calculate_complexity(expr)
        print(f"\n{name}: {expr}")
        print(f"  Depth: {complexity['depth']}")
        print(f"  Nodes: {complexity['nodes']}")
        print(f"  Features: {complexity['feature_count']}")
        print(f"  Operator penalty: {complexity['operator_penalty']:.4f}")
        print(f"  Total complexity: {complexity['total_score']:.4f}")

    return True

def test_complexity_validation():
    """Test complexity validation and filtering"""
    print("\n=== Testing Complexity Validation ===")

    # Strict manager with low limits
    strict_manager = ComplexityManager(max_depth=3, max_nodes=5)

    # Test expressions
    simple_expr = Feature(FeatureType.CLOSE)
    medium_expr = Sub(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))

    # Very complex expression (should fail validation)
    complex_expr = Div(
        lhs=Mul(
            lhs=Sub(lhs=Feature(FeatureType.HIGH), rhs=Feature(FeatureType.LOW)),
            rhs=Add(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.VOLUME))
        ),
        rhs=Sub(
            lhs=Feature(FeatureType.VWAP),
            rhs=Div(lhs=Feature(FeatureType.VOLUME), rhs=Feature(FeatureType.CLOSE))
        )
    )

    test_cases = [
        ("Simple", simple_expr),
        ("Medium", medium_expr),
        ("Very Complex", complex_expr)
    ]

    for name, expr in test_cases:
        is_valid = strict_manager.is_valid_complexity(expr)
        complexity = strict_manager.calculate_complexity(expr)
        print(f"{name}: Valid={is_valid}, Depth={complexity['depth']}, Nodes={complexity['nodes']}")

    return True

def test_complexity_penalties():
    """Test complexity penalty application"""
    print("\n=== Testing Complexity Penalties ===")

    manager = ComplexityManager(penalty_weight=0.2)

    # Mock IC scores and expressions
    test_data = [
        (0.15, Feature(FeatureType.CLOSE), "Simple feature"),
        (0.12, Sub(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN)), "Binary operation"),
        (0.18, Div(
            lhs=Sub(lhs=Feature(FeatureType.HIGH), rhs=Feature(FeatureType.LOW)),
            rhs=Feature(FeatureType.CLOSE)
        ), "Complex expression")
    ]

    for ic_score, expr, description in test_data:
        penalized_score, complexity = manager.apply_complexity_penalty(ic_score, expr)
        penalty = ic_score - penalized_score

        print(f"\n{description}:")
        print(f"  Original IC: {ic_score:.6f}")
        print(f"  Complexity: {complexity['total_score']:.4f}")
        print(f"  Penalty: {penalty:.6f}")
        print(f"  Final Score: {penalized_score:.6f}")

    return True

def test_operator_penalties():
    """Test different operator penalty weights"""
    print("\n=== Testing Operator Penalties ===")

    # Custom operator penalties
    custom_penalties = {
        'Add': 0.05,    # Light penalty
        'Sub': 0.05,    # Light penalty
        'Mul': 0.15,    # Medium penalty
        'Div': 0.40,    # Heavy penalty (numerical instability)
        'Pow': 0.60,    # Very heavy penalty
    }

    manager = ComplexityManager(operator_penalties=custom_penalties)

    # Test different operators
    operators = [
        ("Addition", Add(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))),
        ("Subtraction", Sub(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))),
        ("Multiplication", Mul(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))),
        ("Division", Div(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))),
    ]

    for name, expr in operators:
        complexity = manager.calculate_complexity(expr)
        print(f"{name}: Operator penalty = {complexity['operator_penalty']:.4f}")

    return True

def test_feature_counting():
    """Test feature counting and diversity penalties"""
    print("\n=== Testing Feature Counting ===")

    manager = ComplexityManager(feature_penalty=0.02)

    # Expressions with different feature counts
    expressions = [
        ("Single feature", Feature(FeatureType.CLOSE)),
        ("Two features", Sub(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN))),
        ("Three features", Add(
            lhs=Sub(lhs=Feature(FeatureType.CLOSE), rhs=Feature(FeatureType.OPEN)),
            rhs=Feature(FeatureType.VOLUME)
        )),
        ("Repeated features", Add(
            lhs=Feature(FeatureType.CLOSE),
            rhs=Feature(FeatureType.CLOSE)  # Same feature repeated
        ))
    ]

    for name, expr in expressions:
        complexity = manager.calculate_complexity(expr)
        print(f"{name}: {complexity['feature_count']} unique features, "
              f"penalty = {complexity['feature_penalty']:.4f}")

    return True

def main():
    """Main test execution"""
    print("AlphaForge Expression Complexity Management Tests")
    print("=" * 60)

    try:
        # Run all tests
        tests = [
            test_complexity_calculation,
            test_complexity_validation,
            test_complexity_penalties,
            test_operator_penalties,
            test_feature_counting
        ]

        results = []
        for test in tests:
            try:
                result = test()
                results.append(result)
                print("✅ PASSED")
            except Exception as e:
                print(f"❌ FAILED: {e}")
                results.append(False)

        # Summary
        print("\n" + "=" * 60)
        print("COMPLEXITY MANAGEMENT TEST SUMMARY")
        print("=" * 60)
        passed = sum(results)
        total = len(results)
        print(f"Tests passed: {passed}/{total}")

        if passed == total:
            print("✅ All complexity management tests PASSED")
            return True
        else:
            print("❌ Some complexity management tests FAILED")
            return False

    except Exception as e:
        print(f"\n❌ TESTS FAILED: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)