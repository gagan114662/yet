"""
Darwin Gödel Trading Machine - Core Implementation
Self-improving trading system based on DGM architecture
"""

import os
import json
import shutil
import subprocess
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

class TradingAgent:
    """Individual trading agent with self-modification capabilities"""
    
    def __init__(self, agent_id: str, code_path: str, parent_id: Optional[str] = None):
        self.agent_id = agent_id
        self.code_path = code_path
        self.parent_id = parent_id
        self.performance_metrics = {}
        self.generation = 0
        self.mutations = []
        
    def evaluate(self, lean_workspace: str) -> Dict:
        """Run backtest and return performance metrics"""
        # Copy agent code to lean workspace
        project_name = f"dgm_agent_{self.agent_id}"
        project_path = os.path.join(lean_workspace, project_name)
        
        # Try to remove if exists, ignore errors
        if os.path.exists(project_path):
            try:
                shutil.rmtree(project_path)
            except:
                # Try alternative approach
                os.system(f"rm -rf '{project_path}'")
                
        shutil.copytree(self.code_path, project_path)
        
        # Run backtest using Lean CLI
        try:
            result = subprocess.run(
                ["/home/vandan/.local/bin/lean", "backtest", project_name],
                cwd=lean_workspace,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse results
            if result.returncode == 0:
                return self._parse_backtest_results(result.stdout)
            else:
                return {"cagr": -1.0, "sharpe": -1.0, "max_drawdown": 1.0}
                
        except Exception as e:
            logging.error(f"Evaluation failed for agent {self.agent_id}: {e}")
            return {"cagr": -1.0, "sharpe": -1.0, "max_drawdown": 1.0}
    
    def _parse_backtest_results(self, output: str) -> Dict:
        """Extract metrics from backtest output"""
        metrics = {
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 1.0,
            "total_trades": 0,
            "win_rate": 0.0
        }
        
        # Parse CAGR
        if "Compounding Annual Return" in output:
            try:
                cagr_line = [l for l in output.split('\n') if "Compounding Annual Return" in l][0]
                cagr_value = float(cagr_line.split()[-1].replace('%', '')) / 100
                metrics["cagr"] = cagr_value
            except:
                pass
                
        # Parse Sharpe
        if "Sharpe Ratio" in output and "Probabilistic" not in output:
            try:
                sharpe_line = [l for l in output.split('\n') if "Sharpe Ratio" in l and "Probabilistic" not in l][0]
                metrics["sharpe"] = float(sharpe_line.split()[-1])
            except:
                pass
                
        # Parse Drawdown
        if "Drawdown" in output:
            try:
                dd_line = [l for l in output.split('\n') if "Drawdown" in l][0]
                dd_value = float(dd_line.split()[-1].replace('%', '')) / 100
                metrics["max_drawdown"] = dd_value
            except:
                pass
                
        return metrics
    
    def propose_modification(self) -> str:
        """Analyze performance and propose next improvement"""
        # Analyze current performance
        cagr = self.performance_metrics.get("cagr", 0)
        sharpe = self.performance_metrics.get("sharpe", 0)
        drawdown = self.performance_metrics.get("max_drawdown", 1)
        
        proposals = []
        
        # Performance-based proposals
        if cagr < 0.25:  # Below 25% target
            if cagr < 0.10:
                proposals.append("Increase leverage to 4x-5x for higher returns")
                proposals.append("Add momentum indicators (RSI, MACD) for better timing")
                proposals.append("Trade more frequently (daily instead of weekly)")
            else:
                proposals.append("Optimize position sizing based on volatility")
                proposals.append("Add trend strength filters")
                proposals.append("Implement profit targets and stop losses")
                
        if sharpe < 1.0:
            proposals.append("Add risk management with volatility scaling")
            proposals.append("Implement regime detection")
            proposals.append("Diversify with multiple assets")
            
        if drawdown > 0.20:
            proposals.append("Add drawdown protection mechanism")
            proposals.append("Reduce position size during high volatility")
            proposals.append("Implement trailing stops")
            
        # Random exploration
        exploration_proposals = [
            "Add machine learning predictions",
            "Implement pairs trading",
            "Add options strategies",
            "Use intraday data",
            "Add sentiment analysis",
            "Implement mean reversion signals",
            "Add volume indicators",
            "Use multiple timeframes"
        ]
        
        # Combine targeted and exploratory proposals
        all_proposals = proposals + np.random.choice(exploration_proposals, 2).tolist()
        
        return np.random.choice(all_proposals)
    
    def self_modify(self, modification: str) -> 'TradingAgent':
        """Create modified version of self"""
        # Create new agent
        new_id = f"{self.agent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}"
        
        # Create new path in agents directory
        agents_dir = os.path.join(os.path.dirname(self.code_path), "agents")
        os.makedirs(agents_dir, exist_ok=True)
        new_path = os.path.join(agents_dir, new_id)
        
        # Copy code
        shutil.copytree(self.code_path, new_path)
        
        # Apply modification (simplified - in practice would use LLM)
        self._apply_modification(new_path, modification)
        
        # Create new agent
        new_agent = TradingAgent(new_id, new_path, self.agent_id)
        new_agent.generation = self.generation + 1
        new_agent.mutations = self.mutations + [modification]
        
        return new_agent
    
    def _apply_modification(self, code_path: str, modification: str):
        """Apply code modification (simplified implementation)"""
        main_file = os.path.join(code_path, "main.py")
        
        # Read current code
        with open(main_file, 'r') as f:
            code = f.read()
            
        # Apply modifications based on proposal
        if "leverage" in modification.lower():
            code = code.replace("set_leverage(2.0)", "set_leverage(4.0)")
            code = code.replace("set_leverage(3.0)", "set_leverage(5.0)")
            
        elif "momentum" in modification.lower():
            # Add RSI indicator
            if "self.rsi" not in code:
                init_section = code.find("def initialize(self):")
                insert_point = code.find("\n\n", init_section)
                new_code = "\n        self.rsi = self.rsi(self.symbol, 14)"
                code = code[:insert_point] + new_code + code[insert_point:]
                
        elif "frequently" in modification.lower():
            code = code.replace("days < 5", "days < 1")
            code = code.replace("days < 7", "days < 2")
            
        # Write modified code
        with open(main_file, 'w') as f:
            f.write(code)


class DarwinGodelTradingMachine:
    """Main DGM implementation for trading"""
    
    def __init__(self, initial_agent_path: str, lean_workspace: str, target_cagr: float = 0.25):
        self.lean_workspace = lean_workspace
        self.target_cagr = target_cagr
        self.archive = []
        self.generation = 0
        self.best_agent = None
        self.best_performance = {"cagr": -1.0}
        
        # Initialize with first agent
        initial_agent = TradingAgent("agent_0", initial_agent_path)
        self.archive.append(initial_agent)
        
        # Setup logging
        logging.basicConfig(
            filename='dgm_evolution.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
    def select_parents(self, n_parents: int = 3) -> List[TradingAgent]:
        """Select parents for next generation"""
        if len(self.archive) <= n_parents:
            return self.archive
            
        # Calculate selection probabilities
        performances = []
        for agent in self.archive:
            perf = agent.performance_metrics.get("cagr", -1.0)
            # Add exploration bonus for diversity
            diversity_bonus = 0.1 * len(agent.mutations)
            performances.append(max(0, perf + diversity_bonus))
            
        # Normalize probabilities
        total = sum(performances)
        if total > 0:
            probs = [p/total for p in performances]
        else:
            probs = [1/len(self.archive)] * len(self.archive)
            
        # Select parents
        indices = np.random.choice(len(self.archive), n_parents, p=probs, replace=False)
        return [self.archive[i] for i in indices]
    
    def evolve_generation(self):
        """Run one generation of evolution"""
        self.generation += 1
        logging.info(f"Starting generation {self.generation}")
        
        # Select parents
        parents = self.select_parents()
        new_agents = []
        
        for parent in parents:
            # Propose modification
            modification = parent.propose_modification()
            logging.info(f"Parent {parent.agent_id} proposing: {modification}")
            
            # Create modified agent
            try:
                child = parent.self_modify(modification)
                
                # Evaluate child
                metrics = child.evaluate(self.lean_workspace)
                child.performance_metrics = metrics
                
                logging.info(f"Child {child.agent_id} achieved CAGR: {metrics['cagr']*100:.1f}%")
                
                # Add to archive if valid
                if metrics["cagr"] > -0.5:  # Basic validity check
                    new_agents.append(child)
                    
                    # Check if best
                    if metrics["cagr"] > self.best_performance["cagr"]:
                        self.best_agent = child
                        self.best_performance = metrics
                        logging.info(f"New best agent! CAGR: {metrics['cagr']*100:.1f}%")
                        
            except Exception as e:
                logging.error(f"Failed to create child from {parent.agent_id}: {e}")
                
        # Add new agents to archive
        self.archive.extend(new_agents)
        
        # Log generation summary
        logging.info(f"Generation {self.generation} complete. Archive size: {len(self.archive)}")
        logging.info(f"Best CAGR so far: {self.best_performance['cagr']*100:.1f}%")
        
    def run(self, max_generations: int = 50):
        """Run evolution until target achieved or max generations"""
        # Evaluate initial agent
        initial_agent = self.archive[0]
        initial_metrics = initial_agent.evaluate(self.lean_workspace)
        initial_agent.performance_metrics = initial_metrics
        
        logging.info(f"Initial agent CAGR: {initial_metrics['cagr']*100:.1f}%")
        
        while self.generation < max_generations:
            # Check if target achieved
            if self.best_performance["cagr"] >= self.target_cagr:
                logging.info(f"Target achieved! Best CAGR: {self.best_performance['cagr']*100:.1f}%")
                break
                
            # Evolve next generation
            self.evolve_generation()
            
            # Save checkpoint
            if self.generation % 5 == 0:
                self.save_checkpoint()
                
        return self.best_agent
    
    def save_checkpoint(self):
        """Save current state"""
        checkpoint = {
            "generation": self.generation,
            "archive_size": len(self.archive),
            "best_performance": self.best_performance,
            "best_agent_id": self.best_agent.agent_id if self.best_agent else None
        }
        
        with open(f"dgm_checkpoint_gen{self.generation}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)