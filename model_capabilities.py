"""Versioned model capabilities shared by loading, requests and clients."""
import math
from pathlib import Path

from mflux.models.common.config.model_config import AVAILABLE_MODELS


ADDITIONAL_MODELS = (
    ('anima-turbo', 'anima', 10, 1.0, 'circlestone-labs/Anima'),
    ('anima-aesthetic', 'anima', 30, 4.0, None),
    ('anima-base', 'anima', 30, 4.0, None),
    ('iris-3b', 'iris', 4, 1.0, 'speridlabs/iris-3b'),
    ('iris-3b-base', 'iris', 100, 3.0, None),
    ('ming-image-design', 'ming', 12, 1.0, 'inclusionAI/Ming-Image-0.1-Design'),
    ('qwen-image-2.1-turbo', 'qwen21', 8, 1.0, 'Qwen/Qwen-Image-2.1-Turbo'),
    ('qwen-image-2.1-controlnet-union', 'qwen21-controlnet', 40, 1.0,
     'alibaba-pai/Qwen-Image-2.1-Fun-Controlnet-Union'),
    ('qwen-image-2.1-turbo-controlnet-union', 'qwen21-controlnet', 8, 1.0, None),
    ('krea-2-controlnet-depth', 'krea2-controlnet', 8, 1.0, 'Patil/Krea-2-depth-controlnet'),
    ('qwen-image-edit-2511', 'qwen-edit', 40, 4.0, 'Qwen/Qwen-Image-Edit-2511'),
)
LOAD_FIELDS = {'lora_paths', 'lora_scales', 'bake_lora', 'compute_precision',
               'small_text_encoder', 'small_decoder', 'text_encoder_quantize',
               'vae_full_precision', 'qwen21_backend', 'controlnet_path', 'controlnet_strength'}
GENERATION_FIELDS = {'negative_prompt', 'scheduler', 'step_cache_ratio', 'cfg_end',
                     'use_kv_cache', 'use_step_cache', 'step_cache_threshold',
                     'strength', 'auto_mask', 'enhance_prompt', 'verify', 'verify_retries',
                     'render_text_images', 'uncond_skip_layers', 'controlnet_strength',
                     'output_resolution', 'preset', 'strict_caption_validation'}
IMAGE_FIELDS = ('mask_image', 'control_image', 'depth_image')


def extend_registry(registry):
    for key, loader, steps, guidance, repo_alias in ADDITIONAL_MODELS:
        for name in [*AVAILABLE_MODELS[key].aliases, *([repo_alias] if repo_alias else [])]:
            registry[name] = dict(loader=loader, steps=steps, guidance=guidance, config_name=key)


def capabilities(name, info):
    loader = info['loader']
    key = info.get('config_name', name)
    turbo = loader in ('qwen21', 'qwen21-controlnet') and 'turbo' in key.lower()
    iris_turbo = loader == 'iris' and 'base' not in key.lower()
    limits = {'ideogram4': 0, 'moss-image': 0, 'anima': 0, 'iris': 0, 'ming': 0,
              'flux2': 4, 'flux3': 10, 'qwen21': 10, 'fibo2': 5,
              'qwen-edit': 3, 'qwen21-controlnet': 1, 'krea2-controlnet': 1}
    limit = limits.get(loader, 1)
    load = set()
    generation = set()
    if loader not in ('iris', 'ming'):
        load |= {'lora_paths', 'lora_scales', 'bake_lora'}
    if loader in ('flux2', 'flux3', 'qwen21', 'qwen21-controlnet', 'z-image', 'anima'):
        load.add('compute_precision')
    if loader == 'flux2':
        load.add('small_decoder')
        if '4b' in key.lower():
            load.add('small_text_encoder')
    if loader == 'qwen21':
        load |= {'small_text_encoder', 'qwen21_backend'}
    if loader == 'ming':
        load.add('text_encoder_quantize')
    if loader in ('qwen21-controlnet', 'krea2-controlnet'):
        load.add('controlnet_path')
    if loader == 'krea2-controlnet':
        load.add('controlnet_strength')
    if loader in ('flux2', 'flux3', 'fibo2', 'ernie', 'z-image', 'ideogram4'):
        load.add('vae_full_precision')
    if loader in ('qwen21', 'qwen21-controlnet', 'qwen-edit', 'krea2', 'krea2-controlnet',
                  'fibo2', 'moss-image', 'anima') or loader == 'iris' and not iris_turbo:
        generation.add('negative_prompt')
    schedulers = {'qwen21': ['linear', 'viggle_turbo'] if not turbo else ['linear'],
                  'qwen-edit': ['linear'], 'anima': ['er_sde', 'euler'],
                  'krea2': ['er_sde', 'euler'], 'krea2-controlnet': ['er_sde', 'euler'],
                  'flux2': ['flow_match_euler_discrete']}.get(loader, [])
    if schedulers:
        generation.add('scheduler')
    if loader in ('fibo2', 'moss-image'):
        generation.add('step_cache_ratio')
    if loader in ('fibo2', 'ideogram4'):
        generation.add('cfg_end')
    if loader == 'fibo2':
        generation.add('render_text_images')
    if loader == 'moss-image':
        generation.add('uncond_skip_layers')
    if loader == 'ideogram4':
        generation |= {'preset', 'strict_caption_validation'}
    if loader == 'qwen21':
        generation |= {'use_kv_cache', 'use_step_cache', 'step_cache_threshold', 'strength',
                       'auto_mask', 'enhance_prompt', 'verify', 'verify_retries', 'output_resolution'}
    if loader == 'qwen21-controlnet':
        generation.add('controlnet_strength')
    if loader == 'qwen21-controlnet':
        generation.add('output_resolution')
    mask = loader in ('qwen21', 'qwen21-controlnet', 'fibo2')
    alignment = 32 if loader in ('qwen21', 'qwen21-controlnet') else 16
    fixed = 4 if iris_turbo else None
    return dict(
        edit=limit > 0, multi_image_edit=limit > 1, max_init_images=limit,
        min_init_images=1 if loader == 'qwen-edit' else 0,
        mask=mask, control_image=loader == 'qwen21-controlnet',
        depth_image=loader == 'krea2-controlnet',
        rgba=loader in ('qwen21', 'qwen21-controlnet', 'ming'),
        default_format='PNG' if loader in ('qwen21', 'qwen21-controlnet', 'ming') else 'JPEG',
        dimension_multiple=alignment, min_dimension=256 if loader == 'ideogram4' else alignment,
        max_dimension=2048 if loader == 'ideogram4' else 8192 if loader == 'fibo2' else None,
        min_steps=2 if loader in ('qwen21', 'qwen21-controlnet', 'qwen-edit') else 1,
        max_steps=218 if loader == 'moss-image' else 1000 if loader == 'anima' else None,
        fixed_steps=fixed, quantization=[4, 8] if loader == 'iris' else [4, 5, 6, 8],
        guidance_fixed=1.0 if turbo or iris_turbo else None,
        schedulers=schedulers, load_options=sorted(load), generation_options=sorted(generation),
        default_steps=info.get('steps', 4), default_guidance=info.get('guidance', 3.5),
    )


def resolve_entry(registry, name, base_model=None):
    if name in registry:
        if base_model is not None and base_model not in (name, registry[name].get('base_model')):
            raise ValueError('base_model is only used with a custom model path or repository.')
        return dict(registry[name])
    if base_model not in registry:
        raise ValueError('Custom checkpoints require a registered base_model.')
    path = Path(name).expanduser()
    if path.exists() and not path.is_dir():
        raise ValueError('model must be a complete model directory, not a file.')
    if not path.is_dir() and (name.startswith(('/', '.', '~')) or '/' not in name):
        raise ValueError(f'Model directory does not exist: {name}')
    base = dict(registry[base_model])
    base['config_name'] = base.get('config_name', base_model)
    base['source'] = str(path.resolve()) if path.is_dir() else name
    base['base_model'] = base_model
    return base


def validate_load_options(name, info, quantize, options):
    caps = capabilities(name, info)
    if quantize is not None and (type(quantize) is not int or quantize not in caps['quantization']):
        raise ValueError(f'quantize must be one of {caps["quantization"]}, or null.')
    if not isinstance(options, dict):
        raise ValueError('options must be an object.')
    unsupported = set(options) - set(caps['load_options'])
    if unsupported:
        raise ValueError(f'Unsupported load options for {name}: {", ".join(sorted(unsupported))}')
    for key in ('small_text_encoder', 'small_decoder', 'bake_lora', 'vae_full_precision'):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f'{key} must be a boolean.')
    for key in ('controlnet_path',):
        if key in options and (not isinstance(options[key], str) or not options[key].strip()):
            raise ValueError(f'{key} must be a nonempty path or repository.')
    paths = options.get('lora_paths', [])
    if not isinstance(paths, list) or any(not isinstance(p, str) or not p.strip() for p in paths):
        raise ValueError('lora_paths must be a list of nonempty paths or repository IDs.')
    scales = options.get('lora_scales', [1.0] * len(paths))
    if not isinstance(scales, list) or len(scales) != len(paths) or any(
            type(s) not in (int, float) or not math.isfinite(s) for s in scales):
        raise ValueError('lora_scales must contain one finite number for each LoRA.')
    if options.get('compute_precision', 'bfloat16') not in ('bfloat16', 'float16'):
        raise ValueError('compute_precision must be bfloat16 or float16.')
    if options.get('qwen21_backend', 'native') not in ('native', 'legacy'):
        raise ValueError('qwen21_backend must be native or legacy.')
    if options.get('qwen21_backend') == 'legacy' and (
            'turbo' in info.get('config_name', name).lower() or options.get('small_text_encoder')
            or options.get('lora_paths') or options.get('compute_precision', 'bfloat16') != 'bfloat16'):
        raise ValueError('The legacy Qwen backend supports the base checkpoint without adapters or a small encoder.')
    if 'text_encoder_quantize' in options and options['text_encoder_quantize'] is not None and (
            type(options['text_encoder_quantize']) is not int or options['text_encoder_quantize'] not in (4, 5, 6, 8)):
        raise ValueError('text_encoder_quantize must be 4, 5, 6, 8 or null.')
    if 'controlnet_strength' in options and (type(options['controlnet_strength']) not in (int, float)
            or not math.isfinite(options['controlnet_strength']) or not 0 <= options['controlnet_strength'] <= 10):
        raise ValueError('controlnet_strength must be finite and between 0 and 10.')
    return dict(options)


def validate_parameters(name, info, args, image_count, load_options=None):
    """Validate before queuing; unsupported explicit options are never silently dropped."""
    caps = capabilities(name, info)
    if (load_options or {}).get('qwen21_backend') == 'legacy':
        caps.update(dimension_multiple=16, min_dimension=16, mask=False)
    allowed = {'prompt', 'seed', 'width', 'height', 'steps', 'guidance', 'format', 'quality',
               'priority', 'init_image', 'init_images'} | GENERATION_FIELDS | set(IMAGE_FIELDS)
    unknown = set(args) - allowed
    if unknown:
        raise ValueError('Unknown generation fields: ' + ', '.join(sorted(unknown)))
    if args.get('negative_prompt') is not None and not isinstance(args['negative_prompt'], str):
        raise ValueError('negative_prompt must be a string or null.')
    supplied = set(args) & GENERATION_FIELDS
    bad = supplied - set(caps['generation_options'])
    # Empty negative prompts remain compatible with generic clients.
    if not args.get('negative_prompt'):
        bad.discard('negative_prompt')
    if bad:
        raise ValueError(f'Unsupported generation options for {name}: {", ".join(sorted(bad))}')
    width, height = args['width'], args['height']
    multiple = caps['dimension_multiple']
    if any(v < caps['min_dimension'] or v % multiple for v in (width, height)):
        raise ValueError(f'{name} requires dimensions >= {caps["min_dimension"]}, divisible by {multiple}.')
    if caps['max_dimension'] and max(width, height) > caps['max_dimension']:
        raise ValueError(f'{name} requires dimensions <= {caps["max_dimension"]}.')
    if not caps['min_steps'] <= args['steps'] or caps['max_steps'] and args['steps'] > caps['max_steps']:
        raise ValueError(f'Invalid steps for {name}; see /api/info for the limits.')
    if caps['fixed_steps'] is not None and args['steps'] != caps['fixed_steps']:
        raise ValueError(f'{name} has a fixed {caps["fixed_steps"]}-step sampler.')
    if not math.isfinite(args['guidance']) or args['guidance'] < 0:
        raise ValueError('guidance must be a finite nonnegative number.')
    if caps['guidance_fixed'] is not None and args['guidance'] != caps['guidance_fixed']:
        raise ValueError(f'{name} requires guidance {caps["guidance_fixed"]}.')
    if image_count < caps['min_init_images']:
        raise ValueError(f'{name} requires at least {caps["min_init_images"]} input image(s).')
    for field in IMAGE_FIELDS:
        if args.get(field) is not None and not isinstance(args[field], str):
            raise ValueError(f'{field} must be an encoded image string or null.')
        if args.get(field) and not caps[field if field != 'mask_image' else 'mask']:
            raise ValueError(f'{name} does not support {field}.')
    if args.get('mask_image') and image_count != 1:
        raise ValueError('mask_image requires exactly one source image.')
    loader = info['loader']
    if loader in ('qwen21', 'qwen21-controlnet') and args['guidance'] < 1:
        raise ValueError('Native Qwen 2.1 requires guidance >= 1.')
    if loader in ('qwen21', 'qwen21-controlnet') and args['guidance'] > 1 and args.get('negative_prompt') is None:
        raise ValueError('Guidance above 1 requires negative_prompt, which may be empty.')
    if loader == 'qwen21-controlnet':
        if not args.get('control_image') and not (image_count and args.get('mask_image')):
            raise ValueError('ControlNet requires control_image, or a source image and mask_image.')
        if bool(image_count) != bool(args.get('mask_image')):
            raise ValueError('ControlNet inpainting requires both a source image and mask_image.')
    if loader == 'krea2-controlnet' and not args.get('depth_image') and image_count != 1:
        raise ValueError('Depth ControlNet requires depth_image or one photograph to estimate depth.')
    if loader == 'krea2-controlnet' and args.get('depth_image') and image_count:
        raise ValueError('Choose depth_image or a photograph, not both.')
    if 'scheduler' in args and args['scheduler'] not in caps['schedulers']:
        raise ValueError(f'scheduler must be one of {caps["schedulers"]}.')
    for field in ('use_kv_cache', 'use_step_cache', 'enhance_prompt', 'verify', 'render_text_images',
                  'strict_caption_validation'):
        if field in args and type(args[field]) is not bool:
            raise ValueError(f'{field} must be a boolean.')
    ranges = {'step_cache_ratio': (0, 1, False), 'cfg_end': (0, 1, True),
              'strength': (0, 1, True), 'controlnet_strength': (0, 10, True),
              'step_cache_threshold': (0, 1, True)}
    for field, (low, high, inclusive) in ranges.items():
        if field not in args:
            continue
        value = args[field]
        if type(value) not in (int, float) or not math.isfinite(value) or not (
                (low <= value if field == 'controlnet_strength' else low < value)
                and (value <= high if inclusive else value < high)):
            raise ValueError(f'Invalid {field}; expected a finite value between {low} and {high}.')
    if 'verify_retries' in args and (type(args['verify_retries']) is not int or not 0 <= args['verify_retries'] <= 10):
        raise ValueError('verify_retries must be an integer between 0 and 10.')
    if 'output_resolution' in args and (type(args['output_resolution']) is not int
                                      or args['output_resolution'] < 32 or args['output_resolution'] % 32):
        raise ValueError('output_resolution must be a positive multiple of 32.')
    if 'auto_mask' in args and (not isinstance(args['auto_mask'], str) or not args['auto_mask'].strip()):
        raise ValueError('auto_mask must be a nonempty object description.')
    if loader == 'qwen21':
        if any(args.get(k) for k in ('auto_mask', 'enhance_prompt', 'verify')) or args.get('strength', 1) < 1:
            if not image_count:
                raise ValueError('auto_mask, enhance_prompt, verify and strength require a source image.')
        if args.get('use_step_cache') and not args.get('use_kv_cache', True):
            raise ValueError('use_step_cache requires use_kv_cache.')
        if (load_options or {}).get('small_text_encoder') and any(
                args.get(k) for k in ('auto_mask', 'enhance_prompt', 'verify')):
            raise ValueError('The small text encoder does not support auto_mask, enhance_prompt or verify.')
        if (load_options or {}).get('qwen21_backend') == 'legacy' and supplied - {'negative_prompt'}:
            raise ValueError('Advanced Qwen edit options require the native backend.')
        if 'turbo' in info.get('config_name', name).lower() and args.get('negative_prompt'):
            raise ValueError('Qwen 2.1 Turbo does not use negative conditioning.')
        if args.get('scheduler') == 'viggle_turbo' and not (load_options or {}).get('lora_paths'):
            raise ValueError('viggle_turbo requires a compatible distilled LoRA loaded with the model.')
        if args.get('scheduler') == 'viggle_turbo':
            from mflux.models.qwen21.model.qwen21_scheduler import ViggleTurboScheduler
            if args['steps'] != len(ViggleTurboScheduler.SIGMA_NODES):
                raise ValueError('viggle_turbo requires exactly six steps.')


def discover_saved_models(directory):
    root = Path(directory).expanduser()
    if not root.is_dir():
        return []
    return [str(path.resolve()) for path in sorted(root.iterdir())
            if path.is_dir() and any(path.rglob('*.safetensors'))]
