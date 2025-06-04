#!/usr/bin/env python3
"""
Demo Darwin Gödel Trading Machine evolution with simulated backtests
Shows how strategies evolve to reach 25% CAGR target
"""

import os
import json
import random
import shutil
from datetime import datetime

class DemoAgent:
    """Simplified agent for demonstration"""
    
    def __init__(self, agent_id, parent_id=None):
        self.agent_id = agent_id
        self.parent_id = parent_id
        self.mutations = []
        self.code_features = {
            "leverage": 2.0,
            "indicators": ["sma"],
            "frequency": "weekly",
            "assets": ["SPY"],
            "risk_management": False,
            "stop_loss": None,
            "position_sizing": "fixed"
        }
        self.performance = {"cagr": 0.0, "sharpe": 0.0, "drawdown": 0.0}
        
    def evaluate(self):
        """Simulate backtest results based on features"""
        base_cagr = 0.06  # 6% base
        
        # Leverage impact
        leverage_boost = (self.code_features["leverage"] - 1) * 0.03
        
        # Indicator impact
        indicator_boost = len(self.code_features["indicators"]) * 0.015
        
        # Frequency impact
        freq_boost = 0.02 if self.code_features["frequency"] == "daily" else 0
        
        # Asset diversification
        asset_boost = (len(self.code_features["assets"]) - 1) * 0.01
        
        # Risk management
        risk_boost = 0.02 if self.code_features["risk_management"] else 0
        
        # Calculate CAGR with some randomness
        cagr = base_cagr + leverage_boost + indicator_boost + freq_boost + asset_boost + risk_boost
        cagr *= random.uniform(0.8, 1.2)  # Add variance
        
        # Calculate other metrics
        sharpe = cagr / (0.15 + self.code_features["leverage"] * 0.05)
        drawdown = 0.1 + self.code_features["leverage"] * 0.05
        
        self.performance = {
            "cagr": min(cagr, 0.40),  # Cap at 40%
            "sharpe": sharpe,
            "drawdown": drawdown
        }
        
        return self.performance
    
    def propose_modification(self):
        """Propose next improvement based on performance"""
        cagr = self.performance["cagr"]
        
        if cagr < 0.10:
            proposals = [
                ("leverage", lambda: min(5.0, self.code_features["leverage"] + 1)),
                ("indicators", lambda: self.code_features["indicators"] + ["rsi"]),
                ("frequency", lambda: "daily")
            ]
        elif cagr < 0.20:
            proposals = [
                ("indicators", lambda: self.code_features["indicators"] + ["macd"]),
                ("assets", lambda: self.code_features["assets"] + ["QQQ"]),
                ("risk_management", lambda: True)
            ]
        else:
            proposals = [
                ("position_sizing", lambda: "volatility_scaled"),
                ("stop_loss", lambda: 0.05),
                ("assets", lambda: self.code_features["assets"] + ["TQQQ"])
            ]
            
        # Select random proposal
        feature, modifier = random.choice(proposals)
        return feature, modifier
    
    def self_modify(self, feature, modifier):
        """Create modified child"""
        child = DemoAgent(f"{self.agent_id}_gen{len(self.mutations)+1}", self.agent_id)
        
        # Copy features
        child.code_features = self.code_features.copy()
        child.code_features["indicators"] = self.code_features["indicators"].copy()
        child.code_features["assets"] = self.code_features["assets"].copy()
        
        # Apply modification
        child.code_features[feature] = modifier()
        child.mutations = self.mutations + [f"Modified {feature}"]
        
        return child


class DemoDGM:
    """Simplified DGM for demonstration"""
    
    def __init__(self):
        self.archive = []
        self.generation = 0
        self.best_agent = None
        self.best_cagr = 0.0
        
    def run_evolution(self, generations=10):
        """Run evolution demo"""
        print("🧬 Darwin Gödel Trading Machine - Demo Evolution")
        print("=" * 60)
        print("Target: 25% CAGR")
        print("=" * 60)
        
        # Initialize with base agent
        base_agent = DemoAgent("base")
        base_agent.evaluate()
        self.archive.append(base_agent)
        
        print(f"\nInitial Agent:")
        print(f"  CAGR: {base_agent.performance['cagr']*100:.1f}%")
        print(f"  Features: leverage={base_agent.code_features['leverage']}, "
              f"indicators={len(base_agent.code_features['indicators'])}")
        
        for gen in range(1, generations + 1):
            print(f"\n{'='*60}")
            print(f"Generation {gen}")
            
            # Select parents (top performers)
            parents = sorted(self.archive, key=lambda a: a.performance['cagr'], reverse=True)[:3]
            
            new_agents = []
            for parent in parents:
                # Propose and apply modification
                feature, modifier = parent.propose_modification()
                child = parent.self_modify(feature, modifier)
                
                # Evaluate child
                child.evaluate()
                new_agents.append(child)
                
                print(f"\n  Parent {parent.agent_id} → Child {child.agent_id}")
                print(f"  Modification: {child.mutations[-1]}")
                print(f"  CAGR: {parent.performance['cagr']*100:.1f}% → {child.performance['cagr']*100:.1f}%")
                
                # Track best
                if child.performance['cagr'] > self.best_cagr:
                    self.best_cagr = child.performance['cagr']
                    self.best_agent = child
                    print(f"  🎯 New best! CAGR: {self.best_cagr*100:.1f}%")
            
            # Add to archive
            self.archive.extend(new_agents)
            
            # Show progress
            print(f"\nGeneration {gen} Summary:")
            print(f"  Archive size: {len(self.archive)}")
            print(f"  Best CAGR so far: {self.best_cagr*100:.1f}%")
            
            # Check if target reached
            if self.best_cagr >= 0.25:
                print(f"\n🏆 TARGET ACHIEVED! 25%+ CAGR!")
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print("EVOLUTION COMPLETE")
        print(f"{'='*60}")
        
        if self.best_agent:
            print(f"\nBest Agent: {self.best_agent.agent_id}")
            print(f"CAGR: {self.best_agent.performance['cagr']*100:.1f}%")
            print(f"Sharpe: {self.best_agent.performance['sharpe']:.2f}")
            print(f"Max Drawdown: {self.best_agent.performance['drawdown']*100:.1f}%")
            print(f"\nEvolved Features:")
            for key, value in self.best_agent.code_features.items():
                print(f"  {key}: {value}")
            print(f"\nMutation History:")
            for i, mutation in enumerate(self.best_agent.mutations, 1):
                print(f"  {i}. {mutation}")
        
        # Show evolution path
        print(f"\n📊 Performance Evolution:")
        
        # Track lineage of best agent
        lineage = []
        current = self.best_agent
        while current:
            lineage.append(current)
            current = next((a for a in self.archive if a.agent_id == current.parent_id), None)
        
        lineage.reverse()
        for agent in lineage:
            print(f"  {agent.agent_id}: {agent.performance['cagr']*100:.1f}% CAGR")


if __name__ == "__main__":
    dgm = DemoDGM()
    dgm.run_evolution(generations=15)