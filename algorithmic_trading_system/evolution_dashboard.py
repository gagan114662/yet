#!/usr/bin/env python3
"""
Real-Time Evolution Dashboard
Displays streaming evolution events, strategy family trees, and performance metrics
"""

import asyncio
import time
import json
import threading
from datetime import datetime
from typing import Dict, List, Any
import sys
import os

class EvolutionDashboard:
    """Real-time dashboard for monitoring strategy evolution"""
    
    def __init__(self):
        self.events: List[Dict] = []
        self.active_strategies: Dict[str, Dict] = {}
        self.generation_stats: List[Dict] = []
        self.family_tree: Dict[str, List[str]] = {}
        self.running = False
        
    def log_event(self, event_type: str, data: Dict[str, Any]):
        """Log evolution event"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        }
        self.events.append(event)
        self._display_event(event)
    
    def _display_event(self, event: Dict):
        """Display event in real-time"""
        timestamp = event['timestamp'].split('T')[1][:8]
        event_type = event['type']
        data = event['data']
        
        # Color coding
        colors = {
            'STRATEGY_CREATED': '\033[92m',  # Green
            'EVALUATION_START': '\033[94m',  # Blue
            'EVALUATION_COMPLETE': '\033[93m',  # Yellow
            'MUTATION': '\033[95m',  # Magenta
            'BREEDING': '\033[96m',  # Cyan
            'CHAMPION_FOUND': '\033[91m',  # Red (bright)
            'GENERATION_COMPLETE': '\033[97m',  # White
        }
        
        color = colors.get(event_type, '\033[0m')
        reset = '\033[0m'
        
        print(f"{color}[{timestamp}] {event_type}{reset}")
        
        if event_type == 'STRATEGY_CREATED':
            print(f"  🧬 Created: {data['name']}")
            print(f"     Generation: {data['generation']}")
            print(f"     Parent(s): {', '.join(data.get('parents', ['SEED']))}")
            
        elif event_type == 'EVALUATION_START':
            print(f"  🔬 Evaluating: {data['name']}")
            print(f"     Deploying to QuantConnect...")
            
        elif event_type == 'EVALUATION_COMPLETE':
            print(f"  📊 Results: {data['name']}")
            print(f"     CAGR: {data['performance']:.2f}%")
            print(f"     Sharpe: {data.get('sharpe', 0):.2f}")
            print(f"     QC Project: {data.get('project_id', 'N/A')}")
            
        elif event_type == 'MUTATION':
            print(f"  🧬 Mutation: {data['parent']} → {data['child']}")
            print(f"     Type: {data['mutation_type']}")
            
        elif event_type == 'BREEDING':
            print(f"  👶 Breeding: {data['parent1']} × {data['parent2']}")
            print(f"     Offspring: {data['child']}")
            
        elif event_type == 'CHAMPION_FOUND':
            print(f"  🏆 CHAMPION: {data['name']}")
            print(f"     Performance: {data['performance']:.2f}% CAGR")
            print(f"     🎯 TARGET ACHIEVED!")
            
        elif event_type == 'GENERATION_COMPLETE':
            print(f"  📈 Generation {data['generation']} Summary:")
            print(f"     Population: {data['population_size']}")
            print(f"     Best: {data['best_performance']:.2f}% CAGR")
            print(f"     Average: {data['avg_performance']:.2f}% CAGR")
            print(f"     Champions: {data['champion_count']}")
        
        print()  # Add spacing
    
    def display_family_tree(self):
        """Display strategy family relationships"""
        print("\n" + "="*60)
        print("🌳 STRATEGY FAMILY TREE")
        print("="*60)
        
        # Group by generation
        generations = {}
        for strategy_name, parents in self.family_tree.items():
            # Extract generation from name
            if '_Gen' in strategy_name:
                gen = int(strategy_name.split('_Gen')[1].split('_')[0])
            else:
                gen = 0
            
            if gen not in generations:
                generations[gen] = []
            generations[gen].append((strategy_name, parents))
        
        for gen in sorted(generations.keys()):
            print(f"\n📊 Generation {gen}:")
            for strategy_name, parents in generations[gen]:
                if parents:
                    parent_str = " × ".join(parents)
                    print(f"  {strategy_name} ← {parent_str}")
                else:
                    print(f"  {strategy_name} (SEED)")
    
    def display_performance_summary(self):
        """Display current performance summary"""
        print("\n" + "="*60)
        print("📊 PERFORMANCE SUMMARY")
        print("="*60)
        
        if self.active_strategies:
            strategies = list(self.active_strategies.values())
            strategies.sort(key=lambda x: x.get('performance', 0), reverse=True)
            
            print("\n🏆 Top Performers:")
            for i, strategy in enumerate(strategies[:5], 1):
                perf = strategy.get('performance', 0)
                name = strategy.get('name', 'Unknown')
                gen = strategy.get('generation', 0)
                print(f"  {i}. {name}")
                print(f"     Performance: {perf:.2f}% CAGR")
                print(f"     Generation: {gen}")
                print(f"     QC Project: {strategy.get('qc_project_id', 'N/A')}")
                print()
    
    async def start_monitoring(self):
        """Start real-time monitoring"""
        self.running = True
        print("🚀 EVOLUTION DASHBOARD STARTED")
        print("="*60)
        print("Monitoring live strategy evolution...")
        print("Use Ctrl+C to stop monitoring")
        print("="*60)
        
        while self.running:
            await asyncio.sleep(1)
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.running = False
        print("\n🛑 Dashboard monitoring stopped")

# Integration with evolution system
def create_dashboard_logger(dashboard: EvolutionDashboard):
    """Create logging functions that integrate with dashboard"""
    
    def log_strategy_created(name: str, generation: int, parents: List[str] = None):
        dashboard.log_event('STRATEGY_CREATED', {
            'name': name,
            'generation': generation,
            'parents': parents or []
        })
        dashboard.family_tree[name] = parents or []
    
    def log_evaluation_start(name: str):
        dashboard.log_event('EVALUATION_START', {'name': name})
    
    def log_evaluation_complete(name: str, performance: float, sharpe: float = None, project_id: str = None):
        dashboard.log_event('EVALUATION_COMPLETE', {
            'name': name,
            'performance': performance,
            'sharpe': sharpe,
            'project_id': project_id
        })
        dashboard.active_strategies[name] = {
            'name': name,
            'performance': performance,
            'sharpe': sharpe,
            'qc_project_id': project_id
        }
    
    def log_mutation(parent: str, child: str, mutation_type: str):
        dashboard.log_event('MUTATION', {
            'parent': parent,
            'child': child,
            'mutation_type': mutation_type
        })
    
    def log_breeding(parent1: str, parent2: str, child: str):
        dashboard.log_event('BREEDING', {
            'parent1': parent1,
            'parent2': parent2,
            'child': child
        })
    
    def log_champion(name: str, performance: float):
        dashboard.log_event('CHAMPION_FOUND', {
            'name': name,
            'performance': performance
        })
    
    def log_generation_complete(generation: int, stats: Dict):
        dashboard.log_event('GENERATION_COMPLETE', {
            'generation': generation,
            **stats
        })
        dashboard.generation_stats.append(stats)
    
    return {
        'strategy_created': log_strategy_created,
        'evaluation_start': log_evaluation_start,
        'evaluation_complete': log_evaluation_complete,
        'mutation': log_mutation,
        'breeding': log_breeding,
        'champion': log_champion,
        'generation_complete': log_generation_complete
    }

if __name__ == "__main__":
    # Demo dashboard
    async def demo():
        dashboard = EvolutionDashboard()
        logger = create_dashboard_logger(dashboard)
        
        # Simulate some events
        logger['strategy_created']("MomentumBase_Gen0", 0)
        await asyncio.sleep(1)
        
        logger['evaluation_start']("MomentumBase_Gen0")
        await asyncio.sleep(2)
        
        logger['evaluation_complete']("MomentumBase_Gen0", 18.5, 1.2, "12345")
        await asyncio.sleep(1)
        
        logger['mutation']("MomentumBase_Gen0", "MomentumBase_MLEV_1", "LEVERAGE_BOOST")
        await asyncio.sleep(1)
        
        logger['champion']("MomentumBase_MLEV_1", 26.3)
        
        dashboard.display_family_tree()
        dashboard.display_performance_summary()
    
    asyncio.run(demo())