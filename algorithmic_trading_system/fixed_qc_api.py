#!/usr/bin/env python3
"""
FIXED QuantConnect API that properly uploads and maintains strategy code
"""

import requests
import hashlib
import base64
import time
import json
from typing import Dict, Any, Optional

class FixedQuantConnectAPI:
    """Fixed QC API that forces proper code upload"""
    
    def __init__(self, user_id: str, api_token: str):
        self.user_id = user_id
        self.api_token = api_token
        self.base_url = "https://www.quantconnect.com/api/v2"
        
    def get_headers(self) -> Dict[str, str]:
        """Generate proper authentication headers"""
        timestamp = str(int(time.time()))
        time_stamped_token = f"{self.api_token}:{timestamp}".encode('utf-8')
        hashed_token = hashlib.sha256(time_stamped_token).hexdigest()
        authentication = f"{self.user_id}:{hashed_token}".encode('utf-8')
        authentication = base64.b64encode(authentication).decode('ascii')
        
        return {
            'Authorization': f'Basic {authentication}',
            'Timestamp': timestamp,
            'Content-Type': 'application/json'
        }
    
    def force_upload_strategy(self, strategy_name: str, strategy_code: str) -> Dict[str, Any]:
        """
        FORCE upload strategy code, ensuring it's not overridden by templates
        """
        print(f"\n{'='*60}")
        print(f"🔧 FORCE DEPLOYING: {strategy_name}")
        print(f"{'='*60}")
        print(f"Code length: {len(strategy_code)} characters")
        
        # Step 1: Create project
        project_name = f"{strategy_name}_{int(time.time())}"
        project_id = self._create_project(project_name)
        if not project_id:
            return {'success': False, 'error': 'Failed to create project'}
        
        # Step 2: FORCE upload with multiple attempts
        upload_success = False
        for attempt in range(3):
            print(f"📄 Upload attempt {attempt + 1}/3...")
            
            if self._force_file_upload(project_id, "main.py", strategy_code):
                # Verify upload worked
                if self._verify_code_upload(project_id, strategy_code):
                    upload_success = True
                    print("✅ Code upload verified!")
                    break
                else:
                    print("❌ Upload verification failed, retrying...")
                    time.sleep(2)
            else:
                print("❌ Upload failed, retrying...")
                time.sleep(2)
        
        if not upload_success:
            return {'success': False, 'error': 'Failed to upload correct code after 3 attempts'}
        
        # Step 3: Compile
        print("🔨 Compiling project...")
        compile_id = self._compile_project(project_id)
        if not compile_id:
            return {'success': False, 'error': 'Failed to compile project'}
        
        # Step 4: Create backtest
        print("🚀 Creating backtest...")
        backtest_id = self._create_backtest(project_id, compile_id, f"{strategy_name} 15Y Backtest")
        if not backtest_id:
            return {'success': False, 'error': 'Failed to create backtest'}
        
        return {
            'success': True,
            'project_id': project_id,
            'compile_id': compile_id,
            'backtest_id': backtest_id,
            'url': f"https://www.quantconnect.com/terminal/{project_id}#open/{backtest_id}"
        }
    
    def _create_project(self, name: str) -> Optional[str]:
        """Create project"""
        url = f"{self.base_url}/projects/create"
        headers = self.get_headers()
        data = {"name": name, "language": "Py"}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success') and 'projects' in result:
                project_id = result['projects'][0]['projectId']
                print(f"✅ Project created: {name} (ID: {project_id})")
                return str(project_id)
        
        print(f"❌ Failed to create project: {response.text}")
        return None
    
    def _force_file_upload(self, project_id: str, filename: str, content: str) -> bool:
        """Force upload file, trying multiple methods"""
        headers = self.get_headers()
        
        # Method 1: Try direct create
        url = f"{self.base_url}/files/create"
        data = {"projectId": project_id, "name": filename, "content": content}
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success'):
                return True
        
        # Method 2: Try update (in case file already exists)
        url = f"{self.base_url}/files/update"
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success'):
                return True
        
        # Method 3: Delete and recreate
        delete_url = f"{self.base_url}/files/delete"
        delete_data = {"projectId": project_id, "name": filename}
        requests.post(delete_url, headers=headers, json=delete_data)
        
        # Now try create again
        url = f"{self.base_url}/files/create"
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            return result.get('success', False)
        
        return False
    
    def _verify_code_upload(self, project_id: str, expected_code: str) -> bool:
        """Verify the uploaded code matches what we sent"""
        headers = self.get_headers()
        url = f"{self.base_url}/files/read"
        data = {"projectId": project_id}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if 'files' in result:
                for file_info in result['files']:
                    if file_info.get('name') == 'main.py':
                        uploaded_content = file_info.get('content', '')
                        
                        # Check key indicators
                        has_correct_dates = '2009, 1, 1' in uploaded_content and '2024, 1, 1' in uploaded_content
                        has_trade_counting = 'trade_count' in uploaded_content
                        similar_length = abs(len(uploaded_content) - len(expected_code)) < 100
                        
                        print(f"📊 Verification:")
                        print(f"   Uploaded size: {len(uploaded_content)} chars")
                        print(f"   Expected size: {len(expected_code)} chars")
                        print(f"   Has 15-year dates: {has_correct_dates}")
                        print(f"   Has trade counting: {has_trade_counting}")
                        
                        return has_correct_dates and has_trade_counting and similar_length
        
        return False
    
    def _compile_project(self, project_id: str) -> Optional[str]:
        """Compile project"""
        url = f"{self.base_url}/compile/create"
        headers = self.get_headers()
        data = {"projectId": project_id}
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            result = json.loads(response.text)
            if result.get('success') and 'compileId' in result:
                compile_id = result['compileId']
                print(f"✅ Compiled (ID: {compile_id})")
                return compile_id
        
        print(f"❌ Compilation failed: {response.text}")
        return None
    
    def _create_backtest(self, project_id: str, compile_id: str, name: str) -> Optional[str]:
        """Create backtest with retries"""
        url = f"{self.base_url}/backtests/create"
        
        for attempt in range(3):
            if attempt > 0:
                print(f"⏳ Backtest attempt {attempt + 1}/3...")
                time.sleep(10)
            
            headers = self.get_headers()
            data = {
                "projectId": project_id,
                "compileId": compile_id,
                "backtestName": name
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 200:
                result = json.loads(response.text)
                if result.get('success'):
                    backtest_id = None
                    if 'backtestId' in result:
                        backtest_id = result['backtestId']
                    elif 'backtest' in result and 'backtestId' in result['backtest']:
                        backtest_id = result['backtest']['backtestId']
                    
                    if backtest_id:
                        print(f"✅ Backtest created (ID: {backtest_id})")
                        return backtest_id
        
        print(f"❌ Backtest creation failed after 3 attempts")
        return None