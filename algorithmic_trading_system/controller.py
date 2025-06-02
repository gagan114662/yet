import time
import json # For potential future use (e.g., saving strategies)
# import smtplib # For future email notifications, not used yet

# Project-specific imports
import config
from strategy_utils import generate_next_strategy, Strategy # Import Strategy for type hinting
from backtester import Backtester

class TargetSeekingController:
    def __init__(self):
        """
        Initializes the TargetSeekingController.
        """
        self.targets = config.TARGET_METRICS
        self.required_successful_strategies = config.REQUIRED_SUCCESSFUL_STRATEGIES

        self.successful_strategies = []  # List to store (strategy_object, results_dict) tuples
        self.iteration_count = 0
        self.start_time = time.time()
        self.last_progress_update_time = time.time() # For periodic reporting

        self.failed_attempts_since_pivot = 0 # For adaptive research
        self.current_research_focus = "initial_random_search" # Initial research state
        self.ADAPT_THRESHOLD = 25 # Added instance attribute

        self.backtester = Backtester()

        # The strategy generator can be a function or an object with a method.
        # For now, we'll assign the function from strategy_utils.
        # This can be made more sophisticated later (e.g., a class for genetic algo).
        self.strategy_generator_func = generate_next_strategy

        self.strategies_archive_path = "strategies_archive" # From plan step 1

        print("TargetSeekingController initialized.")
        print(f"Targets: {self.targets}")
        print(f"Required successful strategies: {self.required_successful_strategies}")
        print(f"Backtester instance created: {type(self.backtester)}")
        print(f"Strategy generator function set: {self.strategy_generator_func.__name__}")

    def meets_all_targets(self, results: dict) -> bool:
        """
        Checks if the given strategy results meet all predefined targets.

        Args:
            results: A dictionary of performance metrics for a strategy.
                     Expected keys match those in self.targets.

        Returns:
            True if all targets are met or exceeded, False otherwise.
        """
        try:
            if not results: # Handle cases where results might be empty or None
                print("Warning: meets_all_targets received empty results.")
                return False

            # Check each target metric
            # Annual Return: should be >= target
            if results.get('annual_return', float('-inf')) < self.targets['annual_return']:
                # print(f"Debug: Annual return {results.get('annual_return')} < {self.targets['annual_return']}")
                return False

            # Max Drawdown: should be >= target (since target is negative, e.g., result -0.10 >= target -0.12)
            if results.get('max_drawdown', float('-inf')) < self.targets['max_drawdown']:
                # print(f"Debug: Max drawdown {results.get('max_drawdown')} < {self.targets['max_drawdown']}")
                return False

            # Sharpe Ratio: should be >= target
            if results.get('sharpe_ratio', float('-inf')) < self.targets['sharpe_ratio']:
                # print(f"Debug: Sharpe ratio {results.get('sharpe_ratio')} < {self.targets['sharpe_ratio']}")
                return False

            # Win Rate: should be >= target
            if results.get('win_rate', float('-inf')) < self.targets['win_rate']:
                # print(f"Debug: Win rate {results.get('win_rate')} < {self.targets['win_rate']}")
                return False

            # If all checks passed
            return True

        except KeyError as e:
            print(f"Error: Missing key {e} in results when checking targets. Results: {results}")
            return False
        except TypeError as e:
            print(f"Error: Type error {e} when checking targets. Results: {results}")
            return False

    # --- Methods for Step 8 (Placeholders for now) ---
    def save_strategy(self, strategy, results):
        """Placeholder for saving a successful strategy."""
        print(f"INFO: [Placeholder] Strategy {strategy.id} would be saved here. Results: {results}")
        # Actual implementation might involve writing to a file in self.strategies_archive_path
        # e.g., with json.dump or a custom format

    def notify_user_breakthrough(self, strategy, results):
        """Placeholder for notifying user about a breakthrough."""
        print(f"SUCCESS: Strategy {strategy.id} met all targets! Results: {results}")
        # Actual implementation will involve more detailed formatted messages or alerts

    def analyze_failure(self, strategy, results):
        """Placeholder for analyzing a failed strategy."""
        # print(f"INFO: [Placeholder] Analyzing failure for strategy {strategy.id}. Results: {results}")
        # Actual implementation will involve deeper analysis based on which targets were missed
        pass # Keep it quiet for now during loop testing

    def maybe_send_progress_update(self):
        """Placeholder for potentially sending a progress update."""
        current_time = time.time()
        if (current_time - self.last_progress_update_time) > config.PROGRESS_UPDATE_INTERVAL_SECONDS:
            self.send_progress_update()
            self.last_progress_update_time = current_time

    def send_progress_update(self):
        """Placeholder for sending a progress update."""
        print(f"PROGRESS: Iteration {self.iteration_count}. Successful: {len(self.successful_strategies)}/{self.required_successful_strategies}. Focus: {self.current_research_focus}. Time: {time.ctime()}")
        # Actual implementation will be a formatted report

    def maybe_adapt_research_direction(self, results):
        """Placeholder for potentially adapting research direction."""
        # ADAPT_THRESHOLD = 25 # Changed to use self.ADAPT_THRESHOLD
        if self.failed_attempts_since_pivot >= self.ADAPT_THRESHOLD:
            self.adapt_research_direction(results)
            self.failed_attempts_since_pivot = 0 # Reset counter after adapting

    def adapt_research_direction(self, latest_results):
        """Placeholder for adapting research direction."""
        old_focus = self.current_research_focus
        if "initial_random_search" in self.current_research_focus:
            self.current_research_focus = "momentum_focused_search"
        elif "momentum_focused_search" in self.current_research_focus:
            self.current_research_focus = "mean_reversion_search"
        elif "mean_reversion_search" in self.current_research_focus:
            self.current_research_focus = "volatility_targeting_search"
        else:
            self.current_research_focus = "initial_random_search" # Cycle back
        print(f"ADAPT: Research focus changed from '{old_focus}' to '{self.current_research_focus}' after {self.ADAPT_THRESHOLD} failures. Iteration: {self.iteration_count}")


    def send_final_success_report(self):
        """Placeholder for sending the final success report."""
        print("="*50)
        print("FINAL SUCCESS: All target criteria met!")
        print(f"Found {len(self.successful_strategies)} successful strategies.")
        for i, (strategy, results) in enumerate(self.successful_strategies):
            print(f"\nStrategy {i+1}:")
            print(f"  ID: {strategy.id}")
            print(f"  Type: {strategy.type}")
            print(f"  Parameters: {strategy.parameters}")
            print(f"  Results: {results}")
        print("="*50)

    # --- Main Loop Structure (Step 7) ---
    def run_until_success(self):
        """
        Runs the strategy generation and backtesting loop until the
        required number of successful strategies are found.
        """
        print(f"\nStarting TargetSeekingController main loop at {time.ctime(self.start_time)}")
        print(f"Seeking {self.required_successful_strategies} strategies meeting targets: {self.targets}")

        try:
            while len(self.successful_strategies) < self.required_successful_strategies:
                self.iteration_count += 1

                strategy = self.strategy_generator_func()
                results = self.backtester.backtest_strategy(strategy)

                # Check for errors from the backtester (e.g., Lean CLI failure)
                if results.get('error'):
                    print(f"--- Iteration {self.iteration_count}: BACKTEST ERROR! Strategy {strategy.id}. Error: {results.get('details', results.get('error'))} ---")
                    self.failed_attempts_since_pivot += 1
                    self.analyze_failure(strategy, results) # Pass error results for analysis
                    is_successful = False # Explicitly mark as not successful
                else:
                    # No backtest error, proceed to check if targets are met
                    is_successful = self.meets_all_targets(results)
                    if is_successful:
                        print(f"--- Iteration {self.iteration_count}: SUCCESS! Strategy {strategy.id} met targets. --- ({len(self.successful_strategies)+1}/{self.required_successful_strategies})")
                        self.successful_strategies.append((strategy, results))
                        self.save_strategy(strategy, results)
                        self.notify_user_breakthrough(strategy, results)
                        self.failed_attempts_since_pivot = 0 # Reset on true success
                    else:
                        # print(f"--- Iteration {self.iteration_count}: Did not meet targets. Strategy {strategy.id} ---") # Can be noisy
                        self.failed_attempts_since_pivot += 1
                        self.analyze_failure(strategy, results)

                self.maybe_send_progress_update()
                self.maybe_adapt_research_direction(results)

                # Iteration based safety break for development
                if self.iteration_count % 500 == 0 and self.iteration_count > 0 : # Check every 500 iterations
                    print(f"INFO: Iteration {self.iteration_count} reached. Successful strategies: {len(self.successful_strategies)}.")
                    if self.iteration_count >= 2000: # Increased limit for testing adaptations
                         print("Warning: Exceeded 2000 iterations. Stopping to prevent excessively long run during development.")
                         break

                time.sleep(0.001) # Reduced sleep, backtesting print will slow it down

        except KeyboardInterrupt:
            print("\nLoop interrupted by user (KeyboardInterrupt).")
        finally:
            print(f"\nExiting main loop. Total iterations: {self.iteration_count}")
            if len(self.successful_strategies) >= self.required_successful_strategies:
                self.send_final_success_report()
            else:
                print(f"Stopped before finding all {self.required_successful_strategies} strategies. Found {len(self.successful_strategies)}.")

            elapsed_time = time.time() - self.start_time
            print(f"Total runtime: {elapsed_time:.2f} seconds.")

# Example usage (optional, for testing this file directly)
if __name__ == '__main__':
    # Need to import config for the placeholder methods if they use it directly
    import config
    import time # For time.ctime in send_progress_update

    controller = TargetSeekingController()

    # --- Test Section: Modify backtester on the fly for testing the loop ---
    # This is a hack for testing. In a real scenario, strategy generation
    # and backtesting results would vary naturally.

    original_backtest_method = controller.backtester.backtest_strategy

    # Wrap generated_successful_count in a list to make it mutable from inner function
    generated_successful_count_wrapper = [0]

    def mock_backtest_strategy(strategy_obj):
        # Call the original backtester's print statements
        print(f"Backtesting strategy ID: {strategy_obj.id}, Type: {strategy_obj.type}, Params: {strategy_obj.parameters}")

        # Use controller.required_successful_strategies directly
        if generated_successful_count_wrapper[0] < controller.required_successful_strategies and controller.iteration_count % 150 == 0: # Make it find a good one every 150 iterations
            generated_successful_count_wrapper[0] += 1
            print(f"MOCK: Generating SUCCESSFUL results for strategy {strategy_obj.id}")
            results = {
                'annual_return': 0.25, 'max_drawdown': -0.10,
                'sharpe_ratio': 2.2, 'win_rate': 0.65, 'trades_executed': 120
            }
        else:
            # print(f"MOCK: Generating REGULAR (failing) results for strategy {strategy_obj.id}")
            results = { # Default failing results
                'annual_return': 0.05, 'max_drawdown': -0.20,
                'sharpe_ratio': 0.5, 'win_rate': 0.45, 'trades_executed': 90
            }
        print(f"Backtest results for {strategy_obj.id}: {results}")
        return results

    # Temporarily override the backtest_strategy method for this test run
    controller.backtester.backtest_strategy = mock_backtest_strategy
    # --- End of Test Section ---

    print("\nStarting run_until_success test with MOCKED backtester...")
    controller.run_until_success()
    print("\nrun_until_success test finished.")

    # Restore original method if controller instance were to be used further (not strictly necessary here)
    controller.backtester.backtest_strategy = original_backtest_method
