import sys
import os
import io

# Add parent directory to path so we can import server modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import threading
import json
import unittest
from unittest.mock import MagicMock, patch
from PIL import Image

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

# Create the mock classes
class MockZImageTurbo:
    def __init__(self, **kwargs):
        print("MockZImageTurbo initialized")

    def generate_image(self, seed, prompt, num_inference_steps, height, width, **kwargs):
        init_image = kwargs.get('image_path')
        mask_image = kwargs.get('mask_image_path') # This might be passed differently depending on implementation

        print(f"Mock generate_image called with: prompt='{prompt}', steps={num_inference_steps}, init_image={init_image}, mask={mask_image}")

        # Verify file existence if paths are provided
        if init_image and not os.path.exists(init_image):
            print(f"WARNING: init_image path does not exist: {init_image}")

        result = MagicMock()
        result.image = Image.new('RGB', (width, height), color='red')
        return result

mock_mflux_module.ZImageTurbo = MockZImageTurbo

# 2. Import server modules
with patch.dict('sys.modules', sys.modules):
    import server_adapters
    import server

class TestServerEditsMock(unittest.TestCase):
    def setUp(self):
        # Configure the server
        server.app.config['TESTING'] = True
        self.client = server.app.test_client()

        # Initialize the model adapter with our mock
        server.model_adapter = server_adapters.ZImageTurboAdapter()
        server.model_adapter.load("z-image-turbo")

        # Start the worker thread
        self.worker_thread = threading.Thread(target=server.worker_loop, daemon=True)
        server.shutdown_event.clear()
        self.worker_thread.start()

    def tearDown(self):
        # Signal worker to stop
        server.shutdown_event.set()
        time.sleep(0.2)

        # Clean up any leftover tasks in queue
        with server.queue_lock:
            server.task_queue = []

    def test_edits_endpoint(self):
        """Test the /v1/images/edits endpoint with multipart/form-data"""

        # Create a dummy image
        img_byte_arr = io.BytesIO()
        Image.new('RGB', (100, 100), color='white').save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        data = {
            'image': (img_byte_arr, 'test_image.png'),
            'prompt': 'A futuristic edit',
            'n': 1,
            'size': '512x512',
            'response_format': 'b64_json',
            'steps': 2
        }

        response = self.client.post(
            '/v1/images/edits',
            data=data,
            content_type='multipart/form-data'
        )

        self.assertEqual(response.status_code, 200)
        resp_data = json.loads(response.data)

        self.assertIn('created', resp_data)
        self.assertIn('data', resp_data)
        self.assertEqual(len(resp_data['data']), 1)
        self.assertIn('b64_json', resp_data['data'][0])
        print("\nTest passed: Edits endpoint handled image upload and generation")

    def test_edits_endpoint_with_mask(self):
        """Test the /v1/images/edits endpoint with both image and mask"""

        # Create dummy image
        img_byte_arr = io.BytesIO()
        Image.new('RGB', (100, 100), color='white').save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)

        # Create dummy mask
        mask_byte_arr = io.BytesIO()
        Image.new('L', (100, 100), color='black').save(mask_byte_arr, format='PNG')
        mask_byte_arr.seek(0)

        data = {
            'image': (img_byte_arr, 'test_image.png'),
            'mask': (mask_byte_arr, 'test_mask.png'),
            'prompt': 'Inpainting test',
            'size': '512x512'
        }

        response = self.client.post(
            '/v1/images/edits',
            data=data,
            content_type='multipart/form-data'
        )

        self.assertEqual(response.status_code, 200)
        resp_data = json.loads(response.data)
        self.assertIn('b64_json', resp_data['data'][0])
        print("\nTest passed: Edits endpoint handled image and mask upload")

if __name__ == '__main__':
    unittest.main()
