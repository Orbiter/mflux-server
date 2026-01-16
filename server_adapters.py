from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, List
import pathlib
import os
from PIL import Image
import mlx.core as mx

# Import the model classes
from mflux.models.z_image.variants.turbo.z_image_turbo import ZImageTurbo
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.fibo.variants.txt2img.fibo import FIBO

class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.
    Each adapter handles initialization and generation for a specific model type,
    mapping generic or API-specific parameters to the model's native API.
    """

    @abstractmethod
    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        """
        Load the model into memory.
        """
        pass

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Image.Image:
        """
        Generate an image from a prompt.
        kwargs will contain model-specific parameters.
        """
        pass

class ZImageTurboAdapter(ModelAdapter):
    """
    Adapter for ZImageTurbo model.
    """
    def __init__(self):
        self.model: Optional[ZImageTurbo] = None
        self.model_name: str = ""

    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        self.model_name = model_name
        # ZImageTurbo native init signature:
        # (quantize: int | None = None, model_path: str | None = None, lora_paths: list[str] | None = None, lora_scales: list[float] | None = None, ...)

        # Resolve model path for custom base path
        # If model_path is provided, we should try to append the known model ID structure
        # In mflux-server context, model_path is often a base directory
        final_model_path = model_path
        if model_path:
             # Known mappings for ZImageTurbo
             # If the user passed a base path, we might need to append the model ID
             # But here, we'll assume if it's passed, it's either the full path or a base path.
             # Let's try to be smart: if the path ends with the model name, use it.
             # If not, and it's a directory, look for the model.
             # For MVP, we'll follow the pattern from server_generic.py logic slightly:
             # If "z-image-turbo" is the requested model, and model_path is given,
             # we check if {model_path}/filipstrand/Z-Image-Turbo-mflux-4bit exists.

             # Check common Z-Image paths
             candidates = [
                 os.path.join(model_path, "filipstrand/Z-Image-Turbo-mflux-4bit"),
                 os.path.join(model_path, "filipstrand/Z-Image-Turbo-mflux-8bit"), # Fallback if quantized differs?
                 model_path # Assume it's the direct path
             ]

             for candidate in candidates:
                 if os.path.exists(candidate):
                     final_model_path = candidate
                     print(f"Resolved model path to: {final_model_path}")
                     break

        self.model = ZImageTurbo(quantize=quantize, model_path=final_model_path)
        print(f"Loaded ZImageTurbo model: {model_name}")

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Default parameters for ZImageTurbo
        # signature: (self, seed: int, prompt: str, num_inference_steps: int = 4, height: int = 1024, width: int = 1024, image_path: pathlib.Path | str | None = None, image_strength: float | None = None, scheduler: str = 'linear') -> PIL.Image.Image

        # Extract parameters with defaults matching the model or API requirements
        # Note: OpenAI API uses 'n' for number of images, 'size' for dimensions (e.g. "1024x1024")
        # But here we expect parsed arguments from the controller.

        seed = kwargs.get('seed', 0) # API should handle seed generation if not provided
        steps = kwargs.get('num_inference_steps', 4) # Default for Turbo is usually low
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        scheduler = kwargs.get('scheduler', 'linear')

        # img2img parameters (optional)
        init_image_path = kwargs.get('init_image_path')
        image_strength = kwargs.get('image_strength')

        # Set seed for MLX
        mx.random.seed(seed)

        print(f"Generating with ZImageTurbo: prompt='{prompt}', steps={steps}, size={width}x{height}, seed={seed}, scheduler={scheduler}")

        result = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            image_path=init_image_path,
            image_strength=image_strength,
            scheduler=scheduler
        )

        # The model returns a GeneratedImage object which wraps the PIL image in .image attribute
        # We want to return the PIL Image directly to match the ModelAdapter interface
        if hasattr(result, "image"):
            return result.image
        return result

class QwenAdapter(ModelAdapter):
    """
    Adapter for QwenImage model.
    """
    def __init__(self):
        self.model: Optional[QwenImage] = None
        self.model_name: str = ""

    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        self.model_name = model_name
        hf_model_name = "Qwen/Qwen-Image" # Default if not overridden

        # Resolve model path
        final_model_path = model_path
        if model_path:
            # Check for common Qwen structures if base path provided
            candidates = [
                os.path.join(model_path, "Qwen/Qwen-Image"),
                os.path.join(model_path, "filipstrand/Qwen-Image-mflux-6bit"),
                model_path
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    final_model_path = candidate
                    print(f"Resolved Qwen model path to: {final_model_path}")
                    break

        # QwenImage doesn't have from_name, requires direct instantiation
        # Will default to huggingface if path is None
        self.model = QwenImage(quantize=quantize, model_path=final_model_path)
        print(f"Loaded Qwen model: {model_name} (Path: {final_model_path or hf_model_name})")

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        seed = kwargs.get('seed', 0)
        steps = kwargs.get('num_inference_steps', 4)
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance = kwargs.get('guidance', 4.0) # Default guidance for Qwen
        scheduler = kwargs.get('scheduler', 'linear')
        negative_prompt = kwargs.get('negative_prompt', None)

        init_image_path = kwargs.get('init_image_path')
        image_strength = kwargs.get('image_strength')

        mx.random.seed(seed)

        print(f"Generating with Qwen: prompt='{prompt}', steps={steps}, size={width}x{height}, seed={seed}")

        result = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=init_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            negative_prompt=negative_prompt
        )

        if hasattr(result, "image"):
            return result.image
        return result

class FIBOAdapter(ModelAdapter):
    """
    Adapter for FIBO model.
    """
    def __init__(self):
        self.model: Optional[FIBO] = None
        self.model_name: str = ""

    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        self.model_name = model_name
        hf_model_name = "briaai/FIBO" # Default

        # Resolve model path
        final_model_path = model_path
        if model_path:
            candidates = [
                os.path.join(model_path, "briaai/FIBO"),
                os.path.join(model_path, "briaai/Fibo-mlx-4bit"),
                os.path.join(model_path, "briaai/Fibo-mlx-8bit"),
                model_path
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    final_model_path = candidate
                    print(f"Resolved FIBO model path to: {final_model_path}")
                    break

        self.model = FIBO(quantize=quantize, model_path=final_model_path)
        print(f"Loaded FIBO model: {model_name} (Path: {final_model_path or hf_model_name})")

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        seed = kwargs.get('seed', 0)
        steps = kwargs.get('num_inference_steps', 4)
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance = kwargs.get('guidance', 4.0)
        scheduler = kwargs.get('scheduler', 'linear')
        negative_prompt = kwargs.get('negative_prompt', None)

        init_image_path = kwargs.get('init_image_path')
        image_strength = kwargs.get('image_strength')

        mx.random.seed(seed)

        print(f"Generating with FIBO: prompt='{prompt}', steps={steps}, size={width}x{height}, seed={seed}")

        result = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=init_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            negative_prompt=negative_prompt
        )

        if hasattr(result, "image"):
            return result.image
        return result

class FluxAdapter(ModelAdapter):
    """
    Adapter for Flux1 model (schnell, dev).
    """
    def __init__(self):
        self.model: Optional[Flux1] = None
        self.model_name: str = ""

    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        self.model_name = model_name

        # Map alias to HF model ID if needed
        hf_model_name = model_name
        if model_name == "schnell":
            hf_model_name = "black-forest-labs/FLUX.1-schnell"
        elif model_name == "dev":
            hf_model_name = "black-forest-labs/FLUX.1-dev"

        # Resolve model path
        final_model_path = model_path
        if model_path:
            # Check for common Flux structures if base path provided
            # Standard mflux structure often mirrors HF org/repo
            candidates = [
                os.path.join(model_path, hf_model_name),
                model_path
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    final_model_path = candidate
                    print(f"Resolved Flux model path to: {final_model_path}")
                    break

        # Flux1 signature: (quantize: int | None = None, model_path: str | None = None, lora_paths: list[str] | None = None, lora_scales: list[float] | None = None, model_config: ModelConfig = ...)
        # Note: We use the constructor directly to support model_path, which from_name() doesn't support well

        # If no model_path is provided, mflux usually needs model_name to download/cache from HF.
        # However, Flux1 constructor takes 'model_path' as the location of weights.
        # If we want to load from HF cache by name, we might need to use from_name IF model_path is None.

        if final_model_path:
             self.model = Flux1(quantize=quantize, model_path=final_model_path)
        else:
             # Fallback to from_name for HF cache loading
             self.model = Flux1.from_name(model_name=hf_model_name, quantize=quantize)

        print(f"Loaded Flux model: {model_name} (ID: {hf_model_name})")

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        seed = kwargs.get('seed', 0)
        # Default steps: 4 for schnell, 25 for dev. API controller might pass a generic default (4)
        # We should respect what's passed, but if the user passed 'flux' generically, we might want smart defaults.
        # For now, we trust the controller passed params.
        steps = kwargs.get('num_inference_steps', 4)
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance = kwargs.get('guidance', 3.5) # Flux supports guidance
        scheduler = kwargs.get('scheduler', 'linear')
        negative_prompt = kwargs.get('negative_prompt', None)

        # img2img parameters
        init_image_path = kwargs.get('init_image_path')
        image_strength = kwargs.get('image_strength')

        mx.random.seed(seed)

        print(f"Generating with Flux: prompt='{prompt}', steps={steps}, size={width}x{height}, seed={seed}, guidance={guidance}")

        result = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=init_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            negative_prompt=negative_prompt
        )

        if hasattr(result, "image"):
            return result.image
        return result

class QwenAdapter(ModelAdapter):
    """
    Adapter for QwenImage model.
    """
    def __init__(self):
        self.model: Optional[QwenImage] = None
        self.model_name: str = ""

    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        self.model_name = model_name
        hf_model_name = "Qwen/Qwen-Image" # Default if not overridden

        # Resolve model path
        final_model_path = model_path
        if model_path:
            # Check for common Qwen structures if base path provided
            candidates = [
                os.path.join(model_path, "Qwen/Qwen-Image"),
                os.path.join(model_path, "filipstrand/Qwen-Image-mflux-6bit"),
                model_path
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    final_model_path = candidate
                    print(f"Resolved Qwen model path to: {final_model_path}")
                    break

        # QwenImage doesn't have from_name, requires direct instantiation
        # Will default to huggingface if path is None
        self.model = QwenImage(quantize=quantize, model_path=final_model_path)
        print(f"Loaded Qwen model: {model_name} (Path: {final_model_path or hf_model_name})")

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        seed = kwargs.get('seed', 0)
        steps = kwargs.get('num_inference_steps', 4)
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance = kwargs.get('guidance', 4.0) # Default guidance for Qwen
        scheduler = kwargs.get('scheduler', 'linear')
        negative_prompt = kwargs.get('negative_prompt', None)

        init_image_path = kwargs.get('init_image_path')
        image_strength = kwargs.get('image_strength')

        mx.random.seed(seed)

        print(f"Generating with Qwen: prompt='{prompt}', steps={steps}, size={width}x{height}, seed={seed}")

        result = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=init_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            negative_prompt=negative_prompt
        )

        if hasattr(result, "image"):
            return result.image
        return result

class FIBOAdapter(ModelAdapter):
    """
    Adapter for FIBO model.
    """
    def __init__(self):
        self.model: Optional[FIBO] = None
        self.model_name: str = ""

    def load(self, model_name: str, quantize: Optional[int] = None, model_path: Optional[str] = None):
        self.model_name = model_name
        hf_model_name = "briaai/FIBO" # Default

        # Resolve model path
        final_model_path = model_path
        if model_path:
            candidates = [
                os.path.join(model_path, "briaai/FIBO"),
                os.path.join(model_path, "briaai/Fibo-mlx-4bit"),
                os.path.join(model_path, "briaai/Fibo-mlx-8bit"),
                model_path
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    final_model_path = candidate
                    print(f"Resolved FIBO model path to: {final_model_path}")
                    break

        self.model = FIBO(quantize=quantize, model_path=final_model_path)
        print(f"Loaded FIBO model: {model_name} (Path: {final_model_path or hf_model_name})")

    def generate(self, prompt: str, **kwargs) -> Image.Image:
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        seed = kwargs.get('seed', 0)
        steps = kwargs.get('num_inference_steps', 4)
        height = kwargs.get('height', 1024)
        width = kwargs.get('width', 1024)
        guidance = kwargs.get('guidance', 4.0)
        scheduler = kwargs.get('scheduler', 'linear')
        negative_prompt = kwargs.get('negative_prompt', None)

        init_image_path = kwargs.get('init_image_path')
        image_strength = kwargs.get('image_strength')

        mx.random.seed(seed)

        print(f"Generating with FIBO: prompt='{prompt}', steps={steps}, size={width}x{height}, seed={seed}")

        result = self.model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=init_image_path,
            image_strength=image_strength,
            scheduler=scheduler,
            negative_prompt=negative_prompt
        )

        if hasattr(result, "image"):
            return result.image
        return result
