import sys
import os
import time # For ctime

# Add the project root to the Python path to allow direct execution of main.py
# and ensure that modules like controller, config, etc., can be found.
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now that the path is set up, we can import the controller
try:
    from controller import TargetSeekingController
    import config # To ensure it's accessible and potentially for other setups
except ImportError as e:
    print(f"Error importing necessary modules in main.py: {e}")
    print("Please ensure that main.py is in the 'algorithmic_trading_system' directory,")
    print("and all other .py files (controller.py, config.py, etc.) are also present.")
    print(f"Current sys.path: {sys.path}")
    print(f"Project root attempt: {project_root}")
    sys.exit(1)

def main():
    """
    Main function to initialize and run the TargetSeekingController.
    """
    print("=====================================================")
    print("=== Algorithmic Trading Strategy Research System ===")
    print("=====================================================")
    print(f"System started at: {time.ctime()}")
    print(f"Configuration: Requiring {config.REQUIRED_SUCCESSFUL_STRATEGIES} strategies.")
    print(f"Configuration: Targets are {config.TARGET_METRICS}")
    print("---")

    try:
        controller_instance = TargetSeekingController()
    except Exception as e:
        print(f"FATAL: Could not initialize TargetSeekingController: {e}")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)

    try:
        controller_instance.run_until_success()
    except Exception as e:
        print(f"FATAL: An unexpected error occurred during controller execution: {e}")
        # import traceback
        # traceback.print_exc()
    finally:
        print("\n---")
        print("System shutdown initiated.")
        print(f"System finished at: {time.ctime()}")
        print("=====================================================")

if __name__ == '__main__':
    # This main.py is intended to be a clean entry point.
    # It will use the default Backtester from backtester.py, which currently
    # provides static, non-successful results.
    # Therefore, the TargetSeekingController's run_until_success loop
    # is expected to run until its iteration safety break.
    main()
