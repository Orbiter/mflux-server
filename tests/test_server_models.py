import sys
import os
import json
import unittest
import tempfile
import shutil
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import server modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Setup Mocks BEFORE importing server modules
mock_mflux_module = MagicMock()
sys.modules['mflux'] = mock_mflux_module
sys.modules['mflux.models'] = mock_mflux_module
sys.modules['mflux.models.z_image'] = mock_mflux_module
sys.modules['mflux.models.z_image.variants'] = mock_mflux_module
sys.modules['mflux.models.z_image.variants.turbo'] = mock_mflux_module
sys.modules['mflux.models.z_image.variants.turbo.z_image_turbo'] = mock_mflux_module
sys.modules['mflux.models.flux'] = mock_mflux_module
sys.modules['mflux.models.flux.variants'] = mock_mflux_module
sys.modules['mflux.models.flux.variants.txt2img'] = mock_mflux_module
sys.modules['mflux.models.flux.variants.txt2img.flux'] = mock_mflux_module
sys.modules['mflux.models.qwen'] = mock_mflux_module
sys.modules['mflux.models.qwen.variants'] = mock_mflux_module
sys.modules['mflux.models.qwen.variants.txt2img'] = mock_mflux_module
sys.modules['mflux.models.qwen.variants.txt2img.qwen_image'] = mock_mflux_module
sys.modules['mflux.models.fibo'] = mock_mflux_module
sys.modules['mflux.models.fibo.variants'] = mock_mflux_module
sys.modules['mflux.models.fibo.variants.txt2img'] = mock_mflux_module
sys.modules['mflux.models.fibo.variants.txt2img.fibo'] = mock_mflux_module

# Mock common modules used by adapters
sys.modules['mflux.models.common'] = mock_mflux_module
sys.modules['mflux.models.common.vae'] = mock_mflux_module
sys.modules['mflux.models.common.vae.tiling_config'] = mock_mflux_module

# 2. Import server modules
with patch.dict('sys.modules', sys.modules):
    import server

class TestServerModels(unittest.TestCase):
    def setUp(self):
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()
        self.test_dir = tempfile.mkdtemp()

        # Create some mock model directories
        # Expected structure: {model_path}/{org}/{repo}
        os.makedirs(os.path.join(self.test_dir, "org1", "model1"))
        os.makedirs(os.path.join(self.test_dir, "org2", "model2"))

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_list_models_empty_path(self):
        """Test /v1/models with no model path configured"""
        server.configured_model_path = None
        server.model_adapter = MagicMock()
        server.model_adapter.model_name = "test-model"

        response = self.client.get('/v1/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertEqual(data['object'], 'list')
        self.assertTrue(any(m['id'] == 'test-model' for m in data['data']))

    def test_list_models_with_path(self):
        """Test /v1/models with a configured model path"""
        server.configured_model_path = self.test_dir
        server.model_adapter = MagicMock()
        server.model_adapter.model_name = "active-model"

        response = self.client.get('/v1/models')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        model_ids = [m['id'] for m in data['data']]
        self.assertIn('active-model', model_ids)
        self.assertIn('org1/model1', model_ids)
        self.assertIn('org2/model2', model_ids)

    def test_index_route(self):
        """Test the / route serves index.html"""
        # Patch the app instance's send_static_file method
        with patch.object(server.app, 'send_static_file') as mock_send:
            mock_send.return_value = "mock index"
            response = self.client.get('/')
            print(f"Index response status: {response.status_code}")
            mock_send.assert_called_with('index.html')
            self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()
