"""
Meta-Learning Optimizer for Strategy Generation
Learns how to generate better strategies faster
"""

import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class StrategyFeatures:
    """Features extracted from a strategy for ML"""
    strategy_type: str
    num_indicators: int
    leverage: float
    position_size: float
    stop_loss: float
    complexity_score: float
    risk_level: str
    asset_classes: int
    creation_time_features: Dict[str, float]
    parent_fitness: float = 0.0


@dataclass
class GenerationOutcome:
    """Outcome of generating and testing a strategy"""
    features: StrategyFeatures
    success: bool
    fitness_score: float
    time_to_evaluate: float
    failure_reason: Optional[str] = None
    generation_method: str = "random"


class FeatureExtractor:
    """Extract ML features from strategies"""
    
    def __init__(self):
        self.strategy_type_map = {
            'momentum': 1, 'mean_reversion': 2, 'trend_following': 3,
            'volatility': 4, 'arbitrage': 5, 'hybrid': 6, 'other': 0
        }
        
    def extract_features(self, strategy: Dict, parent_fitness: float = 0.0) -> StrategyFeatures:
        """Extract features from strategy definition"""
        
        # Basic features
        strategy_type = strategy.get('type', 'other')
        indicators = strategy.get('indicators', [])
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        stop_loss = strategy.get('stop_loss', 0.1)
        
        # Complexity score
        complexity_score = (
            len(indicators) * 0.3 +
            (leverage - 1) * 0.2 +
            len(strategy.keys()) * 0.1
        )
        
        # Risk level
        if leverage > 2.5 or position_size > 0.3:
            risk_level = 'high'
        elif leverage < 1.2 and position_size < 0.15:
            risk_level = 'low'
        else:
            risk_level = 'medium'
        
        # Asset classes
        asset_classes = len(strategy.get('asset_classes', {}))
        
        # Time features
        now = datetime.now()
        time_features = {
            'hour': now.hour / 24.0,
            'day_of_week': now.weekday() / 6.0,
            'month': now.month / 12.0
        }
        
        return StrategyFeatures(
            strategy_type=strategy_type,
            num_indicators=len(indicators),
            leverage=leverage,
            position_size=position_size,
            stop_loss=stop_loss,
            complexity_score=complexity_score,
            risk_level=risk_level,
            asset_classes=asset_classes,
            creation_time_features=time_features,
            parent_fitness=parent_fitness
        )
    
    def features_to_vector(self, features: StrategyFeatures) -> np.ndarray:
        """Convert features to numerical vector for ML"""
        vector = [
            self.strategy_type_map.get(features.strategy_type, 0),
            features.num_indicators,
            features.leverage,
            features.position_size,
            features.stop_loss,
            features.complexity_score,
            1 if features.risk_level == 'low' else 0,
            1 if features.risk_level == 'medium' else 0,
            1 if features.risk_level == 'high' else 0,
            features.asset_classes,
            features.creation_time_features['hour'],
            features.creation_time_features['day_of_week'],
            features.creation_time_features['month'],
            features.parent_fitness
        ]
        return np.array(vector)


class MetaLearningOptimizer:
    """
    Meta-learning system that learns to generate better strategies
    """
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.generation_history = deque(maxlen=1000)
        
        # ML models
        self.success_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.fitness_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.time_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
        
        # Feature scaling
        self.scaler = StandardScaler()
        self.models_trained = False
        
        # Strategy generation biases learned from data
        self.learned_biases = {
            'strategy_type_preferences': defaultdict(float),
            'parameter_ranges': defaultdict(tuple),
            'indicator_combinations': defaultdict(float),
            'success_patterns': []
        }
        
        # Performance tracking
        self.generation_speed_history = deque(maxlen=100)
        self.success_rate_history = deque(maxlen=100)
        
    def record_generation_outcome(self, strategy: Dict, success: bool, 
                                fitness_score: float, time_to_evaluate: float,
                                failure_reason: str = None, generation_method: str = "random",
                                parent_fitness: float = 0.0):
        """Record the outcome of generating and testing a strategy"""
        
        features = self.feature_extractor.extract_features(strategy, parent_fitness)
        
        outcome = GenerationOutcome(
            features=features,
            success=success,
            fitness_score=fitness_score,
            time_to_evaluate=time_to_evaluate,
            failure_reason=failure_reason,
            generation_method=generation_method
        )
        
        self.generation_history.append(outcome)
        
        # Update learned biases
        self._update_learned_biases(outcome)
        
        # Retrain models periodically
        if len(self.generation_history) >= 50 and len(self.generation_history) % 25 == 0:
            self._retrain_models()
    
    def _update_learned_biases(self, outcome: GenerationOutcome):
        """Update learned biases based on new outcome"""
        features = outcome.features
        
        # Update strategy type preferences
        if outcome.success:
            self.learned_biases['strategy_type_preferences'][features.strategy_type] += 0.1
        else:
            self.learned_biases['strategy_type_preferences'][features.strategy_type] -= 0.02
        
        # Update parameter ranges for successful strategies
        if outcome.success:
            strategy_type = features.strategy_type
            
            # Learn good leverage ranges
            current_range = self.learned_biases['parameter_ranges'].get(f'{strategy_type}_leverage', (0.5, 3.0))
            new_min = min(current_range[0], features.leverage * 0.9)
            new_max = max(current_range[1], features.leverage * 1.1)
            self.learned_biases['parameter_ranges'][f'{strategy_type}_leverage'] = (new_min, new_max)
            
            # Learn good position size ranges
            current_range = self.learned_biases['parameter_ranges'].get(f'{strategy_type}_position_size', (0.05, 0.3))
            new_min = min(current_range[0], features.position_size * 0.9)
            new_max = max(current_range[1], features.position_size * 1.1)
            self.learned_biases['parameter_ranges'][f'{strategy_type}_position_size'] = (new_min, new_max)
        
        # Track success patterns
        if outcome.success and len(self.learned_biases['success_patterns']) < 50:
            pattern = {
                'type': features.strategy_type,
                'complexity': features.complexity_score,
                'risk_level': features.risk_level,
                'num_indicators': features.num_indicators,
                'fitness': outcome.fitness_score
            }
            self.learned_biases['success_patterns'].append(pattern)
    
    def _retrain_models(self):
        """Retrain ML models with new data"""
        if len(self.generation_history) < 20:
            return
        
        logger.info(f"Retraining models with {len(self.generation_history)} samples")
        
        # Prepare data
        X = []
        y_success = []
        y_fitness = []
        y_time = []
        
        for outcome in self.generation_history:
            features_vector = self.feature_extractor.features_to_vector(outcome.features)
            X.append(features_vector)
            y_success.append(1.0 if outcome.success else 0.0)
            y_fitness.append(outcome.fitness_score)
            y_time.append(outcome.time_to_evaluate)
        
        X = np.array(X)
        y_success = np.array(y_success)
        y_fitness = np.array(y_fitness)
        y_time = np.array(y_time)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train models
        try:
            self.success_predictor.fit(X_scaled, y_success)
            self.fitness_predictor.fit(X_scaled, y_fitness)
            self.time_predictor.fit(X_scaled, y_time)
            self.models_trained = True
            
            # Evaluate model performance
            success_score = self.success_predictor.score(X_scaled, y_success)
            fitness_score = self.fitness_predictor.score(X_scaled, y_fitness)
            
            logger.info(f"Model performance - Success R²: {success_score:.3f}, Fitness R²: {fitness_score:.3f}")
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
    
    def predict_strategy_potential(self, strategy: Dict, parent_fitness: float = 0.0) -> Dict[str, float]:
        """Predict the potential of a strategy before testing"""
        if not self.models_trained:
            return {'success_probability': 0.5, 'expected_fitness': 0.5, 'expected_time': 5.0}
        
        features = self.feature_extractor.extract_features(strategy, parent_fitness)
        features_vector = self.feature_extractor.features_to_vector(features)
        features_scaled = self.scaler.transform([features_vector])
        
        try:
            success_prob = self.success_predictor.predict(features_scaled)[0]
            expected_fitness = self.fitness_predictor.predict(features_scaled)[0]
            expected_time = self.time_predictor.predict(features_scaled)[0]
            
            return {
                'success_probability': np.clip(success_prob, 0, 1),
                'expected_fitness': np.clip(expected_fitness, 0, 1),
                'expected_time': max(0.1, expected_time)
            }
        except Exception as e:
            logger.error(f"Error predicting strategy potential: {e}")
            return {'success_probability': 0.5, 'expected_fitness': 0.5, 'expected_time': 5.0}
    
    def generate_optimized_strategy(self, base_strategy: Optional[Dict] = None, 
                                  target_regime: str = 'neutral') -> Dict:
        """Generate strategy using learned patterns"""
        
        # Start with base or create new
        if base_strategy:
            strategy = base_strategy.copy()
        else:
            strategy = self._create_base_strategy(target_regime)
        
        # Apply learned biases
        strategy = self._apply_learned_biases(strategy)
        
        # Optimize parameters using ML guidance
        strategy = self._optimize_parameters_ml(strategy)
        
        return strategy
    
    def _create_base_strategy(self, target_regime: str) -> Dict:
        """Create base strategy biased toward successful patterns"""
        
        # Choose strategy type based on learned preferences
        type_prefs = self.learned_biases['strategy_type_preferences']
        if type_prefs:
            # Weighted random selection
            types = list(type_prefs.keys())
            weights = [max(0.1, type_prefs[t]) for t in types]
            weights = np.array(weights) / sum(weights)
            strategy_type = np.random.choice(types, p=weights)
        else:
            strategy_type = np.random.choice(['momentum', 'mean_reversion', 'trend_following'])
        
        # Base strategy
        strategy = {
            'name': f'MetaLearned_{strategy_type}_{datetime.now().strftime("%H%M%S")}',
            'type': strategy_type,
            'generation_method': 'meta_learned'
        }
        
        return strategy
    
    def _apply_learned_biases(self, strategy: Dict) -> Dict:
        """Apply learned biases to strategy parameters"""
        strategy_type = strategy.get('type', 'momentum')
        
        # Apply learned parameter ranges
        leverage_key = f'{strategy_type}_leverage'
        if leverage_key in self.learned_biases['parameter_ranges']:
            min_lev, max_lev = self.learned_biases['parameter_ranges'][leverage_key]
            strategy['leverage'] = np.random.uniform(min_lev, max_lev)
        else:
            strategy['leverage'] = np.random.uniform(1.0, 2.5)
        
        position_key = f'{strategy_type}_position_size'
        if position_key in self.learned_biases['parameter_ranges']:
            min_pos, max_pos = self.learned_biases['parameter_ranges'][position_key]
            strategy['position_size'] = np.random.uniform(min_pos, max_pos)
        else:
            strategy['position_size'] = np.random.uniform(0.1, 0.25)
        
        # Choose number of indicators based on successful patterns
        successful_patterns = self.learned_biases['success_patterns']
        if successful_patterns:
            similar_patterns = [p for p in successful_patterns if p['type'] == strategy_type]
            if similar_patterns:
                avg_indicators = np.mean([p['num_indicators'] for p in similar_patterns])
                strategy['num_indicators'] = int(np.clip(avg_indicators + np.random.normal(0, 1), 2, 6))
            else:
                strategy['num_indicators'] = np.random.randint(2, 5)
        else:
            strategy['num_indicators'] = np.random.randint(2, 5)
        
        # Add other parameters
        strategy['stop_loss'] = np.random.uniform(0.05, 0.15)
        strategy['indicators'] = self._choose_indicators(strategy_type, strategy['num_indicators'])
        
        return strategy
    
    def _choose_indicators(self, strategy_type: str, num_indicators: int) -> List[str]:
        """Choose indicators based on strategy type and learned patterns"""
        indicator_pools = {
            'momentum': ['RSI', 'MACD', 'ADX', 'ROC', 'MOM', 'STOCH'],
            'mean_reversion': ['RSI', 'BB', 'STOCH', 'CCI', 'WILLIAMS'],
            'trend_following': ['EMA', 'SMA', 'MACD', 'ADX', 'PSAR'],
            'volatility': ['ATR', 'BB', 'KELT', 'VIX', 'GARCH']
        }
        
        pool = indicator_pools.get(strategy_type, ['RSI', 'MACD', 'EMA', 'ATR'])
        return list(np.random.choice(pool, size=min(num_indicators, len(pool)), replace=False))
    
    def _optimize_parameters_ml(self, strategy: Dict) -> Dict:
        """Use ML to optimize strategy parameters"""
        if not self.models_trained:
            return strategy
        
        # Try small variations and keep the best predicted
        best_strategy = strategy.copy()
        best_prediction = self.predict_strategy_potential(strategy)
        best_score = best_prediction['success_probability'] * best_prediction['expected_fitness']
        
        # Try parameter variations
        for _ in range(10):
            variant = strategy.copy()
            
            # Small random variations
            if 'leverage' in variant:
                variant['leverage'] *= np.random.uniform(0.9, 1.1)
                variant['leverage'] = np.clip(variant['leverage'], 0.5, 4.0)
            
            if 'position_size' in variant:
                variant['position_size'] *= np.random.uniform(0.9, 1.1)
                variant['position_size'] = np.clip(variant['position_size'], 0.01, 0.5)
            
            if 'stop_loss' in variant:
                variant['stop_loss'] *= np.random.uniform(0.9, 1.1)
                variant['stop_loss'] = np.clip(variant['stop_loss'], 0.02, 0.3)
            
            # Predict variant performance
            prediction = self.predict_strategy_potential(variant)
            score = prediction['success_probability'] * prediction['expected_fitness']
            
            if score > best_score:
                best_strategy = variant
                best_score = score
        
        return best_strategy
    
    def should_skip_strategy(self, strategy: Dict) -> Tuple[bool, str]:
        """Decide if a strategy should be skipped based on predictions"""
        if not self.models_trained:
            return False, ""
        
        prediction = self.predict_strategy_potential(strategy)
        
        # Skip if very low success probability
        if prediction['success_probability'] < 0.1:
            return True, f"Low success probability: {prediction['success_probability']:.2f}"
        
        # Skip if expected to take too long with low fitness
        if prediction['expected_time'] > 10 and prediction['expected_fitness'] < 0.3:
            return True, f"High time ({prediction['expected_time']:.1f}s) with low fitness"
        
        return False, ""
    
    def get_generation_recommendations(self) -> Dict:
        """Get recommendations for improving strategy generation"""
        recommendations = {
            'focus_areas': [],
            'avoid_patterns': [],
            'parameter_adjustments': {},
            'success_rate_trend': 'unknown'
        }
        
        if len(self.generation_history) < 20:
            recommendations['focus_areas'].append("Collect more data (need 20+ samples)")
            return recommendations
        
        recent = list(self.generation_history)[-20:]
        
        # Analyze success rate trend
        early_success = np.mean([1 if g.success else 0 for g in recent[:10]])
        late_success = np.mean([1 if g.success else 0 for g in recent[10:]])
        
        if late_success > early_success + 0.1:
            recommendations['success_rate_trend'] = 'improving'
        elif late_success < early_success - 0.1:
            recommendations['success_rate_trend'] = 'declining'
        else:
            recommendations['success_rate_trend'] = 'stable'
        
        # Find successful patterns
        successful = [g for g in recent if g.success]
        failed = [g for g in recent if not g.success]
        
        if successful:
            # Most successful strategy types
            success_types = [g.features.strategy_type for g in successful]
            most_common_success = max(set(success_types), key=success_types.count)
            recommendations['focus_areas'].append(f"Focus on {most_common_success} strategies")
            
            # Good parameter ranges
            avg_leverage = np.mean([g.features.leverage for g in successful])
            avg_position = np.mean([g.features.position_size for g in successful])
            recommendations['parameter_adjustments'] = {
                'leverage_target': avg_leverage,
                'position_size_target': avg_position
            }
        
        if failed:
            # Common failure patterns
            failure_reasons = [g.failure_reason for g in failed if g.failure_reason]
            if failure_reasons:
                most_common_failure = max(set(failure_reasons), key=failure_reasons.count)
                recommendations['avoid_patterns'].append(f"Avoid: {most_common_failure}")
        
        return recommendations
    
    def save_learned_knowledge(self, filepath: str):
        """Save learned knowledge to file"""
        knowledge = {
            'learned_biases': dict(self.learned_biases),
            'generation_history': [asdict(g) for g in list(self.generation_history)[-100:]],
            'models_trained': self.models_trained,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(knowledge, f, indent=2, default=str)
        
        # Save models separately
        if self.models_trained:
            model_data = {
                'success_predictor': self.success_predictor,
                'fitness_predictor': self.fitness_predictor,
                'time_predictor': self.time_predictor,
                'scaler': self.scaler
            }
            with open(filepath.replace('.json', '_models.pkl'), 'wb') as f:
                pickle.dump(model_data, f)
        
        logger.info(f"Saved learned knowledge to {filepath}")
    
    def load_learned_knowledge(self, filepath: str):
        """Load learned knowledge from file"""
        try:
            with open(filepath, 'r') as f:
                knowledge = json.load(f)
            
            # Restore learned biases
            self.learned_biases = defaultdict(float)
            for key, value in knowledge['learned_biases'].items():
                self.learned_biases[key] = value
            
            # Try to load models
            model_filepath = filepath.replace('.json', '_models.pkl')
            try:
                with open(model_filepath, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.success_predictor = model_data['success_predictor']
                self.fitness_predictor = model_data['fitness_predictor']
                self.time_predictor = model_data['time_predictor']
                self.scaler = model_data['scaler']
                self.models_trained = True
                
                logger.info("Loaded trained models successfully")
            except FileNotFoundError:
                logger.info("No trained models found, will train from scratch")
            
            logger.info(f"Loaded learned knowledge from {filepath}")
            
        except FileNotFoundError:
            logger.info("No previous knowledge found, starting fresh")
    
    def generate_learning_report(self) -> Dict:
        """Generate comprehensive learning progress report"""
        report = {
            'learning_status': {
                'total_samples': len(self.generation_history),
                'models_trained': self.models_trained,
                'success_patterns_learned': len(self.learned_biases['success_patterns'])
            },
            'performance_trends': {},
            'learned_preferences': dict(self.learned_biases['strategy_type_preferences']),
            'recommendations': self.get_generation_recommendations()
        }
        
        if len(self.generation_history) >= 20:
            recent = list(self.generation_history)[-20:]
            
            report['performance_trends'] = {
                'recent_success_rate': np.mean([1 if g.success else 0 for g in recent]),
                'avg_fitness': np.mean([g.fitness_score for g in recent]),
                'avg_generation_time': np.mean([g.time_to_evaluate for g in recent])
            }
        
        return report


def demonstrate_meta_learning():
    """Demonstrate meta-learning optimizer"""
    logger.info("=== Meta-Learning Optimizer Demo ===")
    
    optimizer = MetaLearningOptimizer()
    
    # Simulate strategy generation and testing
    for i in range(50):
        # Generate strategy
        strategy = {
            'name': f'TestStrategy_{i}',
            'type': np.random.choice(['momentum', 'mean_reversion', 'trend_following']),
            'leverage': np.random.uniform(1.0, 3.0),
            'position_size': np.random.uniform(0.05, 0.3),
            'stop_loss': np.random.uniform(0.05, 0.2),
            'indicators': np.random.choice(['RSI', 'MACD', 'BB', 'EMA'], size=3, replace=False).tolist()
        }
        
        # Simulate testing outcome
        success = np.random.random() < 0.3  # 30% success rate
        fitness = np.random.random() if success else np.random.random() * 0.5
        test_time = np.random.uniform(1, 10)
        
        # Record outcome
        optimizer.record_generation_outcome(
            strategy=strategy,
            success=success,
            fitness_score=fitness,
            time_to_evaluate=test_time,
            failure_reason="Low returns" if not success else None
        )
        
        # Every 10 strategies, show progress
        if (i + 1) % 10 == 0:
            logger.info(f"Processed {i + 1} strategies")
            
            # Try generating optimized strategy
            if optimizer.models_trained:
                optimized = optimizer.generate_optimized_strategy()
                prediction = optimizer.predict_strategy_potential(optimized)
                logger.info(f"Generated optimized strategy with predicted success: {prediction['success_probability']:.2f}")
    
    # Generate final report
    report = optimizer.generate_learning_report()
    logger.info("\n=== Learning Report ===")
    logger.info(f"Total samples: {report['learning_status']['total_samples']}")
    logger.info(f"Models trained: {report['learning_status']['models_trained']}")
    
    if 'performance_trends' in report:
        trends = report['performance_trends']
        logger.info(f"Recent success rate: {trends['recent_success_rate']:.1%}")
        logger.info(f"Average fitness: {trends['avg_fitness']:.3f}")
    
    logger.info("\nLearned preferences:")
    for strategy_type, preference in report['learned_preferences'].items():
        logger.info(f"  {strategy_type}: {preference:.2f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    demonstrate_meta_learning()