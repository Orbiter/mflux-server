import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Add parent directory to path so we can import server modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mock mflux dependencies
mock_mflux_module = MagicMock()
sys.modules['mflux'] = mock_mflux_module
sys.modules['mflux.models'] = mock_mflux_module
sys.modules['mflux.models.z_image'] = mock_mflux_module
sys.modules['mflux.models.z_image.variants'] = mock_mflux_module
sys.modules['mflux.models.z_image.variants.turbo'] = mock_mflux_module
sys.modules['mflux.models.z_image.variants.turbo.z_image_turbo'] = mock_mflux_module

class MockZImageTurbo:
    def __init__(self, quantize=None, model_path=None):
        self.quantize = quantize
        self.model_path = model_path
        print(f"MockZImageTurbo init: path={model_path}")

mock_mflux_module.ZImageTurbo = MockZImageTurbo

# Import adapter
import server_adapters

class TestPathResolution(unittest.TestCase):
    def test_custom_path_resolution(self):
        adapter = server_adapters.ZImageTurboAdapter()

        # We need to simulate the filesystem existence for os.path.exists
        # We want to verify that if we provide a base path "/Volumes/LLMS/image",
        # it correctly looks for and finds "/Volumes/LLMS/image/filipstrand/Z-Image-Turbo-mflux-4bit"

        base_path = "/Volumes/LLMS/image"
        expected_resolved_path = "/Volumes/LLMS/image/filipstrand/Z-Image-Turbo-mflux-4bit"

        # Patch os.path.exists to return True only for our expected path
        with patch('os.path.exists') as mock_exists:
            def side_effect(path):
                return path == expected_resolved_path
            mock_exists.side_effect = side_effect

            adapter.load("z-image-turbo", model_path=base_path)

            # Verify the model was initialized with the resolved path
            self.assertEqual(adapter.model.model_path, expected_resolved_path)
            print("\nPath resolution verified successfully")

if __name__ == '__main__':
    unittest.main()
