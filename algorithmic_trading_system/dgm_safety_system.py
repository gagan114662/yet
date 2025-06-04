"""
DGM Safety System - Claude Code Safety Patterns for DGM
Layered safety with permission scopes, ANR detection, and sandboxing
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum
import json
import traceback
from contextlib import asynccontextmanager
import resource
import psutil
import signal

logger = logging.getLogger(__name__)


class PermissionScope(Enum):
    """Permission scopes for different strategy operations"""
    EXPERIMENTAL = "experimental"
    PRODUCTION = "production"
    RESEARCH = "research"
    SANDBOX = "sandbox"
    RESTRICTED = "restricted"


class SafetyLevel(Enum):
    """Safety levels for operations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class OperationPermissions:
    """Permissions for different operations"""
    scope: PermissionScope
    allowed_operations: Set[str]
    risk_limits: Dict[str, float]
    timeout_seconds: int
    resource_limits: Dict[str, Any]


@dataclass
class SafetyViolation:
    """Record of safety violation"""
    timestamp: float
    violation_type: str
    strategy_id: str
    details: Dict[str, Any]
    severity: SafetyLevel
    action_taken: str


class DGMSafetySystem:
    """
    Claude Code safety patterns applied to DGM evolution
    Provides sandboxing, permission management, and ANR detection
    """
    
    def __init__(self):
        self.permission_scopes = self._initialize_permission_scopes()
        self.safety_violations = []
        self.active_executions = {}
        self.resource_monitors = {}
        
        # Safety thresholds
        self.max_execution_time = 300  # 5 minutes max
        self.max_memory_mb = 2048
        self.max_cpu_percent = 80
        
        logger.info("🛡️ DGM Safety System initialized")
    
    def _initialize_permission_scopes(self) -> Dict[PermissionScope, OperationPermissions]:
        """Initialize permission scopes with safety boundaries"""
        return {
            PermissionScope.EXPERIMENTAL: OperationPermissions(
                scope=PermissionScope.EXPERIMENTAL,
                allowed_operations={
                    'low_risk_mutations', 'indicator_changes', 'parameter_tuning',
                    'basic_backtesting', 'performance_analysis'
                },
                risk_limits={
                    'max_leverage': 2.0,
                    'max_position_size': 0.2,
                    'max_drawdown_estimate': 0.25
                },
                timeout_seconds=60,
                resource_limits={
                    'max_memory_mb': 512,
                    'max_cpu_percent': 50
                }
            ),
            
            PermissionScope.PRODUCTION: OperationPermissions(
                scope=PermissionScope.PRODUCTION,
                allowed_operations={
                    'proven_patterns', 'minor_optimizations', 'validated_strategies',
                    'production_backtesting', 'risk_assessment'
                },
                risk_limits={
                    'max_leverage': 1.5,
                    'max_position_size': 0.15,
                    'max_drawdown_estimate': 0.20
                },
                timeout_seconds=30,
                resource_limits={
                    'max_memory_mb': 256,
                    'max_cpu_percent': 30
                }
            ),
            
            PermissionScope.RESEARCH: OperationPermissions(
                scope=PermissionScope.RESEARCH,
                allowed_operations={
                    'aggressive_mutations', 'new_strategy_types', 'experimental_features',
                    'advanced_backtesting', 'deep_analysis', 'novel_explorations'
                },
                risk_limits={
                    'max_leverage': 5.0,
                    'max_position_size': 0.5,
                    'max_drawdown_estimate': 0.5
                },
                timeout_seconds=180,
                resource_limits={
                    'max_memory_mb': 1024,
                    'max_cpu_percent': 70
                }
            ),
            
            PermissionScope.SANDBOX: OperationPermissions(
                scope=PermissionScope.SANDBOX,
                allowed_operations={
                    'safe_testing', 'mock_operations', 'validation_only'
                },
                risk_limits={
                    'max_leverage': 1.0,
                    'max_position_size': 0.1,
                    'max_drawdown_estimate': 0.1
                },
                timeout_seconds=15,
                resource_limits={
                    'max_memory_mb': 128,
                    'max_cpu_percent': 20
                }
            ),
            
            PermissionScope.RESTRICTED: OperationPermissions(
                scope=PermissionScope.RESTRICTED,
                allowed_operations={'read_only', 'analysis_only'},
                risk_limits={
                    'max_leverage': 0.0,
                    'max_position_size': 0.0,
                    'max_drawdown_estimate': 0.0
                },
                timeout_seconds=10,
                resource_limits={
                    'max_memory_mb': 64,
                    'max_cpu_percent': 10
                }
            )
        }
    
    def check_permissions(self, strategy: Dict, operation: str, scope: PermissionScope) -> Tuple[bool, str]:
        """Check if operation is permitted for strategy in given scope"""
        
        if scope not in self.permission_scopes:
            return False, f"Unknown permission scope: {scope}"
        
        permissions = self.permission_scopes[scope]
        
        # Check operation permission
        if operation not in permissions.allowed_operations:
            return False, f"Operation '{operation}' not allowed in {scope.value} scope"
        
        # Check risk limits
        risk_check, risk_reason = self._check_risk_limits(strategy, permissions.risk_limits)
        if not risk_check:
            return False, f"Risk limit violation: {risk_reason}"
        
        return True, "Permission granted"
    
    def _check_risk_limits(self, strategy: Dict, risk_limits: Dict[str, float]) -> Tuple[bool, str]:
        """Check if strategy parameters are within risk limits"""
        
        # Check leverage
        leverage = strategy.get('leverage', 1.0)
        if leverage > risk_limits.get('max_leverage', float('inf')):
            return False, f"Leverage {leverage} exceeds limit {risk_limits['max_leverage']}"
        
        # Check position size
        position_size = strategy.get('position_size', 0.1)
        if position_size > risk_limits.get('max_position_size', float('inf')):
            return False, f"Position size {position_size} exceeds limit {risk_limits['max_position_size']}"
        
        # Check estimated drawdown risk
        estimated_dd = self._estimate_drawdown_risk(strategy)
        if estimated_dd > risk_limits.get('max_drawdown_estimate', float('inf')):
            return False, f"Estimated drawdown {estimated_dd:.2%} exceeds limit {risk_limits['max_drawdown_estimate']:.2%}"
        
        return True, "Risk limits satisfied"
    
    def _estimate_drawdown_risk(self, strategy: Dict) -> float:
        """Estimate potential drawdown risk from strategy parameters"""
        # Simple heuristic - in real implementation, use sophisticated risk models
        leverage = strategy.get('leverage', 1.0)
        position_size = strategy.get('position_size', 0.1)
        stop_loss = strategy.get('stop_loss', 0.1)
        
        # Risk increases with leverage and position size, decreases with tighter stops
        base_risk = 0.1
        leverage_risk = (leverage - 1) * 0.05
        position_risk = position_size * 0.2
        stop_protection = (0.1 - stop_loss) * 0.5
        
        estimated_risk = base_risk + leverage_risk + position_risk - stop_protection
        return max(0.01, min(1.0, estimated_risk))
    
    async def safe_strategy_execution(self, strategy: Dict, operation: str, 
                                     scope: PermissionScope = PermissionScope.EXPERIMENTAL,
                                     executor_func = None) -> Dict[str, Any]:
        """
        Execute strategy operation with full safety wrapper
        Includes permission checks, ANR detection, resource monitoring
        """
        execution_id = f"exec_{int(time.time())}_{hash(str(strategy))}"
        start_time = time.time()
        
        # Permission check
        permitted, reason = self.check_permissions(strategy, operation, scope)
        if not permitted:
            violation = SafetyViolation(
                timestamp=time.time(),
                violation_type="permission_denied",
                strategy_id=strategy.get('id', 'unknown'),
                details={'operation': operation, 'scope': scope.value, 'reason': reason},
                severity=SafetyLevel.MEDIUM,
                action_taken="execution_blocked"
            )
            self.safety_violations.append(violation)
            return {'error': 'Permission denied', 'reason': reason, 'scope': scope.value}
        
        # Get timeout for scope
        timeout = self.permission_scopes[scope].timeout_seconds
        
        try:
            # Start resource monitoring
            monitor_task = asyncio.create_task(
                self._monitor_execution_resources(execution_id, scope)
            )
            
            # Execute with timeout and resource limits
            if executor_func:
                result = await asyncio.wait_for(
                    self._execute_with_resource_limits(executor_func, strategy, scope),
                    timeout=timeout
                )
            else:
                # Mock execution for demonstration
                result = await asyncio.wait_for(
                    self._mock_strategy_execution(strategy),
                    timeout=timeout
                )
            
            # Stop monitoring
            monitor_task.cancel()
            
            execution_time = time.time() - start_time
            
            # Validate results
            if self._validate_strategy_results(result):
                logger.info(f"✅ Safe execution completed: {execution_id} in {execution_time:.2f}s")
                return result
            else:
                raise ValueError("Strategy results failed validation")
                
        except asyncio.TimeoutError:
            violation = SafetyViolation(
                timestamp=time.time(),
                violation_type="execution_timeout",
                strategy_id=strategy.get('id', 'unknown'),
                details={'timeout': timeout, 'operation': operation},
                severity=SafetyLevel.HIGH,
                action_taken="execution_terminated"
            )
            self.safety_violations.append(violation)
            logger.error(f"⏰ Execution timeout: {execution_id}")
            return {'error': 'Strategy execution timeout', 'timeout': timeout}
            
        except MemoryError:
            violation = SafetyViolation(
                timestamp=time.time(),
                violation_type="memory_limit",
                strategy_id=strategy.get('id', 'unknown'),
                details={'operation': operation},
                severity=SafetyLevel.CRITICAL,
                action_taken="execution_terminated"
            )
            self.safety_violations.append(violation)
            logger.error(f"💾 Memory limit exceeded: {execution_id}")
            return {'error': 'Memory limit exceeded', 'strategy_id': strategy.get('id')}
            
        except Exception as e:
            violation = SafetyViolation(
                timestamp=time.time(),
                violation_type="execution_error",
                strategy_id=strategy.get('id', 'unknown'),
                details={'error': str(e), 'traceback': traceback.format_exc()},
                severity=SafetyLevel.HIGH,
                action_taken="execution_failed"
            )
            self.safety_violations.append(violation)
            logger.error(f"❌ Execution error: {execution_id}: {e}")
            return {'error': 'Strategy execution failed', 'details': str(e)}
        
        finally:
            # Cleanup
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
    
    async def _execute_with_resource_limits(self, executor_func, strategy: Dict, scope: PermissionScope):
        """Execute function with resource limits"""
        resource_limits = self.permission_scopes[scope].resource_limits
        
        # Set memory limits (in practice, use more sophisticated resource management)
        max_memory_bytes = resource_limits['max_memory_mb'] * 1024 * 1024
        
        # Execute function
        result = await executor_func(strategy)
        
        # Check memory usage
        current_memory = psutil.Process().memory_info().rss
        if current_memory > max_memory_bytes:
            raise MemoryError(f"Memory usage {current_memory} exceeds limit {max_memory_bytes}")
        
        return result
    
    async def _monitor_execution_resources(self, execution_id: str, scope: PermissionScope):
        """Monitor resource usage during execution"""
        resource_limits = self.permission_scopes[scope].resource_limits
        max_cpu = resource_limits['max_cpu_percent']
        max_memory_mb = resource_limits['max_memory_mb']
        
        self.active_executions[execution_id] = {
            'start_time': time.time(),
            'scope': scope,
            'max_cpu': max_cpu,
            'max_memory_mb': max_memory_mb
        }
        
        try:
            while True:
                await asyncio.sleep(1)  # Check every second
                
                # Get current process info
                process = psutil.Process()
                cpu_percent = process.cpu_percent()
                memory_mb = process.memory_info().rss / (1024 * 1024)
                
                # Check limits
                if cpu_percent > max_cpu:
                    violation = SafetyViolation(
                        timestamp=time.time(),
                        violation_type="cpu_limit_exceeded",
                        strategy_id=execution_id,
                        details={'cpu_percent': cpu_percent, 'limit': max_cpu},
                        severity=SafetyLevel.HIGH,
                        action_taken="resource_warning"
                    )
                    self.safety_violations.append(violation)
                    logger.warning(f"⚠️ CPU usage {cpu_percent}% exceeds limit {max_cpu}%")
                
                if memory_mb > max_memory_mb:
                    violation = SafetyViolation(
                        timestamp=time.time(),
                        violation_type="memory_limit_exceeded",
                        strategy_id=execution_id,
                        details={'memory_mb': memory_mb, 'limit': max_memory_mb},
                        severity=SafetyLevel.CRITICAL,
                        action_taken="execution_terminated"
                    )
                    self.safety_violations.append(violation)
                    logger.error(f"💾 Memory usage {memory_mb}MB exceeds limit {max_memory_mb}MB")
                    raise MemoryError(f"Memory limit exceeded: {memory_mb}MB > {max_memory_mb}MB")
                
        except asyncio.CancelledError:
            # Monitoring cancelled (normal completion)
            pass
    
    async def _mock_strategy_execution(self, strategy: Dict) -> Dict[str, Any]:
        """Mock strategy execution for demonstration"""
        # Simulate execution time
        execution_time = strategy.get('leverage', 1.0) * 0.5  # Higher leverage = longer execution
        await asyncio.sleep(min(execution_time, 5.0))
        
        # Generate mock results
        import numpy as np
        return {
            'cagr': np.random.uniform(0.05, 0.30),
            'sharpe_ratio': np.random.uniform(0.5, 1.5),
            'max_drawdown': np.random.uniform(0.05, 0.25),
            'execution_time': execution_time,
            'strategy_id': strategy.get('id', 'unknown'),
            'safety_scope': 'mock_execution'
        }
    
    def _validate_strategy_results(self, result: Dict) -> bool:
        """Validate strategy results for sanity"""
        if 'error' in result:
            return False
        
        # Check for reasonable values
        cagr = result.get('cagr', 0)
        if cagr < -1.0 or cagr > 10.0:  # Unreasonable returns
            return False
        
        sharpe = result.get('sharpe_ratio', 0)
        if sharpe < -5.0 or sharpe > 10.0:  # Unreasonable Sharpe
            return False
        
        drawdown = result.get('max_drawdown', 0)
        if drawdown < 0 or drawdown > 1.0:  # Invalid drawdown
            return False
        
        return True
    
    def get_safety_report(self) -> Dict[str, Any]:
        """Generate comprehensive safety report"""
        
        # Violation analysis
        violation_counts = {}
        severity_counts = {}
        
        for violation in self.safety_violations:
            violation_counts[violation.violation_type] = violation_counts.get(violation.violation_type, 0) + 1
            severity_counts[violation.severity.value] = severity_counts.get(violation.severity.value, 0) + 1
        
        # Recent violations (last 24 hours)
        recent_threshold = time.time() - 24 * 3600
        recent_violations = [v for v in self.safety_violations if v.timestamp > recent_threshold]
        
        # Active executions
        active_count = len(self.active_executions)
        
        return {
            'safety_overview': {
                'total_violations': len(self.safety_violations),
                'recent_violations_24h': len(recent_violations),
                'active_executions': active_count,
                'permission_scopes_configured': len(self.permission_scopes)
            },
            'violation_breakdown': {
                'by_type': violation_counts,
                'by_severity': severity_counts
            },
            'recent_violations': [
                {
                    'type': v.violation_type,
                    'severity': v.severity.value,
                    'timestamp': datetime.fromtimestamp(v.timestamp).isoformat(),
                    'action_taken': v.action_taken
                }
                for v in recent_violations[-10:]  # Last 10 recent violations
            ],
            'permission_scope_summary': {
                scope.value: {
                    'timeout_seconds': perms.timeout_seconds,
                    'operation_count': len(perms.allowed_operations),
                    'max_leverage': perms.risk_limits.get('max_leverage', 'unlimited'),
                    'max_memory_mb': perms.resource_limits.get('max_memory_mb', 'unlimited')
                }
                for scope, perms in self.permission_scopes.items()
            },
            'recommendations': self._generate_safety_recommendations()
        }
    
    def _generate_safety_recommendations(self) -> List[str]:
        """Generate safety recommendations based on violation history"""
        recommendations = []
        
        # Analyze violation patterns
        if len(self.safety_violations) > 0:
            violation_types = [v.violation_type for v in self.safety_violations]
            
            if violation_types.count('execution_timeout') > 3:
                recommendations.append("Consider increasing timeout limits or optimizing strategy execution")
            
            if violation_types.count('memory_limit_exceeded') > 2:
                recommendations.append("Implement more aggressive memory management or increase limits")
            
            if violation_types.count('permission_denied') > 5:
                recommendations.append("Review permission scopes - may be too restrictive")
            
            # Recent violations
            recent_violations = [v for v in self.safety_violations if time.time() - v.timestamp < 3600]  # Last hour
            if len(recent_violations) > 5:
                recommendations.append("High violation rate detected - review current operations")
        
        if not recommendations:
            recommendations.append("Safety system operating normally")
        
        return recommendations
    
    @asynccontextmanager
    async def safe_execution_context(self, scope: PermissionScope = PermissionScope.EXPERIMENTAL):
        """Context manager for safe execution with automatic cleanup"""
        execution_id = f"ctx_{int(time.time())}"
        
        try:
            logger.info(f"🔒 Entering safe execution context: {scope.value}")
            yield scope
        except Exception as e:
            logger.error(f"❌ Exception in safe execution context: {e}")
            # Record violation
            violation = SafetyViolation(
                timestamp=time.time(),
                violation_type="context_exception",
                strategy_id=execution_id,
                details={'error': str(e)},
                severity=SafetyLevel.HIGH,
                action_taken="context_cleanup"
            )
            self.safety_violations.append(violation)
            raise
        finally:
            logger.info(f"🔓 Exiting safe execution context: {scope.value}")
            # Cleanup any resources
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]


# Example usage and integration functions

async def safe_dgm_evolution_cycle(safety_system: DGMSafetySystem, strategies: List[Dict]) -> List[Dict]:
    """Example of safe DGM evolution cycle"""
    results = []
    
    for strategy in strategies:
        # Determine appropriate scope based on strategy risk
        if strategy.get('type') == 'experimental':
            scope = PermissionScope.RESEARCH
        elif strategy.get('leverage', 1.0) > 2.0:
            scope = PermissionScope.EXPERIMENTAL
        else:
            scope = PermissionScope.PRODUCTION
        
        # Execute safely
        result = await safety_system.safe_strategy_execution(
            strategy=strategy,
            operation='backtest_strategy',
            scope=scope
        )
        
        results.append(result)
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    async def demo_safety_system():
        """Demonstrate DGM safety system"""
        safety = DGMSafetySystem()
        
        print("🛡️ DGM Safety System Demonstration")
        print("=" * 50)
        
        # Test strategies with different risk levels
        test_strategies = [
            {
                'id': 'safe_strategy',
                'type': 'momentum',
                'leverage': 1.5,
                'position_size': 0.15,
                'stop_loss': 0.1
            },
            {
                'id': 'risky_strategy',
                'type': 'experimental',
                'leverage': 4.0,  # High leverage
                'position_size': 0.4,  # Large position
                'stop_loss': 0.05
            },
            {
                'id': 'moderate_strategy',
                'type': 'mean_reversion',
                'leverage': 2.2,
                'position_size': 0.2,
                'stop_loss': 0.08
            }
        ]
        
        print("\n🧪 Testing strategies with safety system...")
        
        # Test each strategy
        for strategy in test_strategies:
            print(f"\n📊 Testing {strategy['id']}:")
            
            # Test permission check
            scope = PermissionScope.EXPERIMENTAL
            permitted, reason = safety.check_permissions(strategy, 'backtest_strategy', scope)
            print(f"   Permission check: {'✅ GRANTED' if permitted else '❌ DENIED'} - {reason}")
            
            if permitted:
                # Execute safely
                result = await safety.safe_strategy_execution(
                    strategy=strategy,
                    operation='backtest_strategy',
                    scope=scope
                )
                
                if 'error' in result:
                    print(f"   Execution: ❌ {result['error']}")
                else:
                    print(f"   Execution: ✅ Success - CAGR: {result.get('cagr', 0):.1%}")
        
        # Generate safety report
        report = safety.get_safety_report()
        print(f"\n📋 Safety Report:")
        print(f"   Total violations: {report['safety_overview']['total_violations']}")
        print(f"   Active executions: {report['safety_overview']['active_executions']}")
        
        if report['recent_violations']:
            print(f"   Recent violations:")
            for violation in report['recent_violations']:
                print(f"     - {violation['type']} ({violation['severity']})")
        
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"   - {rec}")
    
    # Run demo
    asyncio.run(demo_safety_system())