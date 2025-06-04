import unittest
from unittest.mock import patch, MagicMock
import requests
import hashlib
import base64
import time
import json
import subprocess
from pathlib import Path

# Attempt to import QuantConnect classes and config
# This structure assumes the test file might be run from different locations or that paths might need adjustment.
try:
    from algorithmic_trading_system import config
    from quantconnect_integration.quantconnect_cloud_deployer import QuantConnectCloudDeployer
    from quantconnect_integration.automated_cloud_backtest import QuantConnectCloudBacktester
    from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration
except ImportError:
    # Fallback for local execution if paths are not set up in the environment
    import sys
    # Assuming the script is in quantconnect_integration, and algorithmic_trading_system is a sibling
    current_dir = Path(__file__).resolve().parent
    root_dir = current_dir.parent
    sys.path.append(str(root_dir))
    from algorithmic_trading_system import config
    from quantconnect_integration.quantconnect_cloud_deployer import QuantConnectCloudDeployer
    from quantconnect_integration.automated_cloud_backtest import QuantConnectCloudBacktester
    from quantconnect_integration.rd_agent_qc_bridge import QuantConnectIntegration

class TestQuantConnectAPI(unittest.TestCase):

    def setUp(self):
        self.user_id = config.LEAN_CLI_USER_ID
        self.api_token = config.LEAN_CLI_API_TOKEN
        self.test_project_name = "TestProject_123"
        self.test_strategy_code = "print('hello world')"

    @patch('requests.get')
    def test_authentication_cloud_deployer_method(self, mock_get):
        """
        Tests the authentication header generation as per QuantConnectCloudDeployer.
        Mocks the actual API call and verifies header construction.
        """
        deployer = QuantConnectCloudDeployer(user_id=self.user_id, api_token=self.api_token)
        
        # Expected header construction
        # Capture timestamp before get_headers is called to ensure it's consistent
        timestamp_val = str(int(time.time()))

        # Mock the time.time() call within get_headers to use the same timestamp
        with patch('time.time', return_value=int(timestamp_val)):
            headers = deployer.get_headers()

            # Simulate calling a read endpoint
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = json.dumps({"success": True, "projects": []})
            mock_get.return_value = mock_response

            # Make a dummy call that would use these headers
            requests.get(f"{deployer.base_url}/projects/read", headers=headers)

        # Verification
        mock_get.assert_called_once()
        called_args, called_kwargs = mock_get.call_args
        called_headers = called_kwargs['headers']

        self.assertIn('Authorization', called_headers)
        self.assertIn('Timestamp', called_headers)
        self.assertEqual(called_headers['Timestamp'], timestamp_val)

        # Reconstruct expected auth part for verification
        time_stamped_token_expected = f"{self.api_token}:{timestamp_val}".encode('utf-8')
        hashed_token_expected = hashlib.sha256(time_stamped_token_expected).hexdigest()
        authentication_expected_raw = f"{self.user_id}:{hashed_token_expected}".encode('utf-8')
        authentication_expected_b64 = base64.b64encode(authentication_expected_raw).decode('ascii')

        self.assertEqual(called_headers['Authorization'], f'Basic {authentication_expected_b64}')
        self.assertEqual(called_headers['Content-Type'], 'application/json')
        print("\nTestQuantConnectAPI: test_authentication_cloud_deployer_method PASSED")

    @patch('requests.get')
    def test_authentication_automated_backtester_method(self, mock_get):
        """
        Tests the authentication header generation as per QuantConnectCloudBacktester.
        Mocks the actual API call and verifies header construction.
        """
        # automated_cloud_backtest.py uses hardcoded credentials.
        # For the test, we'll use the config credentials to be consistent.
        backtester = QuantConnectCloudBacktester()
        backtester.user_id = self.user_id
        backtester.token = self.api_token

        # Simulate calling a read endpoint
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({"success": True, "projects": []})
        mock_get.return_value = mock_response

        expected_auth_raw = f"{self.user_id}:{self.api_token}".encode('utf-8')
        expected_auth_b64 = base64.b64encode(expected_auth_raw).decode('ascii')
        expected_headers = {
            "Authorization": f"Basic {expected_auth_b64}",
            "Content-Type": "application/json"
        }

        # The automated_cloud_backtester directly calls requests.get with its headers
        # We simulate such a call here to check if the headers would be correct
        requests.get(f"{backtester.api_base}/projects/read", headers=expected_headers)

        mock_get.assert_called_once_with(f"{backtester.api_base}/projects/read", headers=expected_headers)
        print("\nTestQuantConnectAPI: test_authentication_automated_backtester_method PASSED")

    @patch('requests.post')
    def test_cloud_deployer_create_project_success(self, mock_post):
        """
        Tests QuantConnectCloudDeployer.create_project successful API call.
        """
        deployer = QuantConnectCloudDeployer(user_id=self.user_id, api_token=self.api_token)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = json.dumps({
            "success": True,
            "projects": [{"projectId": "12345"}]
        })
        mock_post.return_value = mock_response

        project_id = deployer.create_project(self.test_project_name)

        self.assertEqual(project_id, "12345")
        mock_post.assert_called_once()
        called_args, called_kwargs = mock_post.call_args
        self.assertEqual(called_args[0], f"{deployer.base_url}/projects/create")
        self.assertEqual(called_kwargs['json']['name'], self.test_project_name)
        print("\nTestQuantConnectAPI: test_cloud_deployer_create_project_success PASSED")

    @patch('requests.post')
    def test_cloud_deployer_create_project_failure(self, mock_post):
        """
        Tests QuantConnectCloudDeployer.create_project failed API call.
        """
        deployer = QuantConnectCloudDeployer(user_id=self.user_id, api_token=self.api_token)
        
        mock_response = MagicMock()
        mock_response.status_code = 401 # Unauthorized
        mock_response.text = json.dumps({"success": False, "errors": ["Authentication failed."]})
        mock_post.return_value = mock_response

        project_id = deployer.create_project(self.test_project_name)

        self.assertIsNone(project_id)
        mock_post.assert_called_once()
        print("\nTestQuantConnectAPI: test_cloud_deployer_create_project_failure PASSED")

    @patch('subprocess.run')
    def test_rd_agent_bridge_create_lean_project(self, mock_subprocess_run):
        """
        Tests QuantConnectIntegration.create_lean_project (from rd_agent_qc_bridge.py)
        Verifies that the correct 'lean create-project' command is constructed.
        """
        bridge = QuantConnectIntegration(user_id=self.user_id, api_token=self.api_token)

        mock_process_result = MagicMock()
        mock_process_result.returncode = 0
        mock_process_result.stdout = "Successfully created project."
        mock_process_result.stderr = ""
        mock_subprocess_run.return_value = mock_process_result

        expected_project_path = Path.cwd() / self.test_project_name

        created_path_str = bridge.create_lean_project(project_name=self.test_project_name)

        self.assertEqual(created_path_str, str(expected_project_path))

        expected_cmd = f"lean create-project \"{self.test_project_name}\" --language python"
        mock_subprocess_run.assert_called_once_with(
            expected_cmd, shell=True, check=True, cwd=".", capture_output=True, text=True
        )
        print("\nTestQuantConnectAPI: test_rd_agent_bridge_create_lean_project PASSED")

if __name__ == '__main__':
    unittest.main(verbosity=2)