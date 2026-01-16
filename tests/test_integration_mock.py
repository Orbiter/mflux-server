import sys
import os

# Add parent directory to path so we can import server modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import unittest
import base64
import io
from unittest.mock import MagicMock, patch
from PIL import Image

# 1. Setup Mocks BEFORE importing server modules to avoid MLX/mflux dependencies
mock_mflux = MagicMock()
sys.modules['mflux'] = mock_mflux
sys.modules['mflux.models'] = mock_mflux
sys.modules['mflux.models.z_image'] = mock_mflux
sys.modules['mflux.models.z_image.variants'] = mock_mflux
sys.modules['mflux.models.z_image.variants.turbo'] = mock_mflux
sys.modules['mflux.models.z_image.variants.turbo.z_image_turbo'] = mock_mflux
sys.modules['mflux.models.flux'] = mock_mflux
sys.modules['mflux.models.flux.variants'] = mock_mflux
sys.modules['mflux.models.flux.variants.txt2img'] = mock_mflux
sys.modules['mflux.models.flux.variants.txt2img.flux'] = mock_mflux
sys.modules['mflux.models.qwen'] = mock_mflux
sys.modules['mflux.models.qwen.variants'] = mock_mflux
sys.modules['mflux.models.qwen.variants.txt2img'] = mock_mflux
sys.modules['mflux.models.qwen.variants.txt2img.qwen_image'] = mock_mflux
sys.modules['mflux.models.fibo'] = mock_mflux
sys.modules['mflux.models.fibo.variants'] = mock_mflux
sys.modules['mflux.models.fibo.variants.txt2img'] = mock_mflux
sys.modules['mflux.models.fibo.variants.txt2img.fibo'] = mock_mflux
sys.modules['mlx'] = MagicMock()
sys.modules['mlx.core'] = MagicMock()

# 2. Import server and server_adapters
import server
import server_adapters

class TestIntegrationMock(unittest.TestCase):
    def setUp(self):
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()

        # Mock the adapter
        self.mock_adapter = MagicMock(spec=server_adapters.ModelAdapter)
        self.mock_adapter.model_name = "mock-model"

        # Mock the generate method to return a PIL Image
        mock_image = Image.new('RGB', (64, 64), color='red')
        self.mock_adapter.generate.return_value = mock_image

        # Inject the mock adapter into the server
        server.model_adapter = self.mock_adapter

        # Start worker thread (server.py uses a global worker loop)
        import threading
        server.shutdown_event.clear()
        self.worker_thread = threading.Thread(target=server.worker_loop, daemon=True)
        self.worker_thread.start()

    def tearDown(self):
        server.shutdown_event.set()
        # Small sleep to allow worker to exit
        import time
        time.sleep(0.1)

    def test_generations_standard_payload(self):
        """Test POST /v1/images/generations with standard payload"""
        payload = {
            "prompt": "a beautiful sunset",
            "size": "512x512",
            "steps": 10,
            "seed": 42
        }

        response = self.client.post(
            '/v1/images/generations',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

        # Verify adapter was called with correct parameters
        # Note: server.py maps 'steps' to 'num_inference_steps'
        self.mock_adapter.generate.assert_called_once()
        args, kwargs = self.mock_adapter.generate.call_args
        self.assertEqual(kwargs['prompt'], "a beautiful sunset")
        self.assertEqual(kwargs['width'], 512)
        self.assertEqual(kwargs['height'], 512)
        self.assertEqual(kwargs['num_inference_steps'], 10)
        self.assertEqual(kwargs['seed'], 42)

    def test_generations_init_image_payload(self):
        """Test POST /v1/images/generations with init_image (base64)"""
        # Create a small dummy image base64
        img = Image.new('RGB', (10, 10), color='blue')
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        payload = {
            "prompt": "a cat in a hat",
            "init_image": img_base64,
            "image_strength": 0.5
        }

        response = self.client.post(
            '/v1/images/generations',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

        # Check what the adapter received
        self.mock_adapter.generate.assert_called_once()
        args, kwargs = self.mock_adapter.generate.call_args

        # Findings: Check if init_image_path was passed instead of init_image
        print(f"\nAdapter received keys: {list(kwargs.keys())}")
        if 'init_image_path' in kwargs:
            print(f"init_image_path: {kwargs['init_image_path']}")
            # We can't easily check os.path.exists(kwargs['init_image_path']) here
            # because the worker thread might have already deleted it in the finally block.
            # The log output confirms it was created and deleted.

        # The raw init_image should NOT be in kwargs anymore
        self.assertNotIn('init_image', kwargs)
        self.assertIn('init_image_path', kwargs)
        self.assertEqual(kwargs['image_strength'], 0.5)

if __name__ == '__main__':
    unittest.main()
