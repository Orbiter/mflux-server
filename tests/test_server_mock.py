import sys
import os

# Add parent directory to path so we can import server modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import threading
import json
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

# 1. Setup Mocks BEFORE importing server modules
# We need to mock the ZImageTurbo class to avoid loading the actual model weights
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

# Create the mock classes
class MockZImageTurbo:
    def __init__(self, **kwargs):
        print("MockZImageTurbo initialized")

    def generate_image(self, seed, prompt, num_inference_steps, height, width, **kwargs):
        print(f"Mock generate_image called with: prompt='{prompt}', steps={num_inference_steps}, scheduler={kwargs.get('scheduler')}")
        # Return a generated image object (wrapper for adapter handling)
        result = MagicMock()
        result.image = Image.new('RGB', (width, height), color='black')
        return result

class MockFlux1:
    def __init__(self, **kwargs):
        print("MockFlux1 initialized")

    @classmethod
    def from_name(cls, model_name, quantize=None):
        print(f"MockFlux1.from_name: name={model_name}")
        return cls()

    def generate_image(self, seed, prompt, num_inference_steps, height, width, guidance, **kwargs):
        scheduler = kwargs.get('scheduler')
        negative_prompt = kwargs.get('negative_prompt')
        print(f"MockFlux1 generate_image called with: prompt='{prompt}', steps={num_inference_steps}, guidance={guidance}, scheduler={scheduler}, neg='{negative_prompt}'")
        result = MagicMock()
        result.image = Image.new('RGB', (width, height), color='blue')
        return result

mock_mflux_module.ZImageTurbo = MockZImageTurbo
mock_mflux_module.Flux1 = MockFlux1

# Create Qwen and FIBO Mocks
class MockQwenImage:
    def __init__(self, **kwargs):
        print("MockQwenImage initialized")

    def generate_image(self, seed, prompt, num_inference_steps, height, width, guidance, **kwargs):
        print(f"MockQwenImage generate_image called with: prompt='{prompt}', steps={num_inference_steps}, guidance={guidance}")
        result = MagicMock()
        result.image = Image.new('RGB', (width, height), color='green')
        return result

class MockFIBO:
    def __init__(self, **kwargs):
        print("MockFIBO initialized")

    def generate_image(self, seed, prompt, num_inference_steps, height, width, guidance, **kwargs):
        print(f"MockFIBO generate_image called with: prompt='{prompt}', steps={num_inference_steps}, guidance={guidance}")
        result = MagicMock()
        result.image = Image.new('RGB', (width, height), color='yellow')
        return result

mock_mflux_module.QwenImage = MockQwenImage
mock_mflux_module.FIBO = MockFIBO

# 2. Import server modules
# We must ensure server_adapters sees our mock
with patch.dict('sys.modules', sys.modules):
    import server_adapters
    import server

class TestServerMock(unittest.TestCase):
    def setUp(self):
        # Configure the server
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()

        # Initialize the model adapter with our mock
        # We manually set the global model_adapter in server.py
        server.model_adapter = server_adapters.ZImageTurboAdapter()
        server.model_adapter.load("z-image-turbo") # This uses MockZImageTurbo

        # Start the worker thread
        self.worker_thread = threading.Thread(target=server.worker_loop, daemon=True)
        server.shutdown_event.clear()
        self.worker_thread.start()

    def tearDown(self):
        # Signal worker to stop
        server.shutdown_event.set()
        # Wait a bit for it to loop
        time.sleep(0.2)

    def test_generate_image_endpoint(self):
        """Test the OpenAI-compatible generation endpoint"""
        payload = {
            "prompt": "A futuristic city",
            "size": "512x512",
            "steps": 2,
            "scheduler": "euler", # Custom param for ZImage
            "response_format": "b64_json"
        }

        response = self.client.post(
            '/v1/images/generations',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn('created', data)
        self.assertIn('data', data)
        self.assertEqual(len(data['data']), 1)
        self.assertIn('b64_json', data['data'][0])
        print("\nTest passed: Response contains valid structure and b64_json")

    def test_flux_endpoint(self):
        """Test the generation endpoint with Flux model"""
        # Switch model adapter to Flux
        server.model_adapter = server_adapters.FluxAdapter()
        server.model_adapter.load("schnell")

        payload = {
            "prompt": "A blue sky",
            "size": "512x512",
            "steps": 4,
            "guidance": 3.5,
            "model": "schnell",
            "response_format": "b64_json"
        }

        response = self.client.post(
            '/v1/images/generations',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('b64_json', data['data'][0])
        print("\nTest passed: Flux endpoint works")

    def test_qwen_endpoint(self):
        """Test the generation endpoint with Qwen model"""
        server.model_adapter = server_adapters.QwenAdapter()
        server.model_adapter.load("qwen")

        payload = {
            "prompt": "A text prompt",
            "model": "qwen",
            "response_format": "b64_json"
        }

        response = self.client.post(
            '/v1/images/generations',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('b64_json', data['data'][0])
        print("\nTest passed: Qwen endpoint works")

    def test_fibo_endpoint(self):
        """Test the generation endpoint with FIBO model"""
        server.model_adapter = server_adapters.FIBOAdapter()
        server.model_adapter.load("fibo")

        payload = {
            "prompt": "A photo",
            "model": "fibo",
            "response_format": "b64_json"
        }

        response = self.client.post(
            '/v1/images/generations',
            data=json.dumps(payload),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('b64_json', data['data'][0])
        print("\nTest passed: FIBO endpoint works")

if __name__ == '__main__':
    unittest.main()
