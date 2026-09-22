import os
import json
import time
import hashlib
import random
import requests
import argparse
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO

# mflux server Client
def mflux_generate_client(mfluxendpoint, prompt, width=1280, height=720, steps=4, count=4, seed=None, format="JPEG", quality=85, priority=False, init_images=None):
    mfluxendpoint = mfluxendpoint.rstrip('/')
    data = {
        "prompt": prompt,
        "height": height,
        "width": width,
        "steps": steps,
        "format": format,
        "quality": quality,
        "priority": priority
    }
    if seed is not None:
        data["seed"] = str(seed)
    if init_images:
        response = requests.get(mfluxendpoint + '/api/info', timeout=10)
        response.raise_for_status()
        info = response.json()
        limit = info.get('max_init_images', 0) if info.get('edit') is True else 0
        if info.get('multi_image_edit') is not True:
            limit = min(limit, 1)
        if len(init_images) > limit:
            raise ValueError(f"{info['model']} accepts at most {limit} input image(s).")
        data['init_images'] = [base64.b64encode(Path(path).read_bytes()).decode('ascii') for path in init_images]
    response = requests.post(mfluxendpoint + "/api/generate", json=data)
    response.raise_for_status()
    # parse the response and get the task_id
    json = response.json()
    task_id = json["task_id"]
    return task_id
    
def mflux_status_ready(mfluxendpoint, task_id):
    response = requests.get(mfluxendpoint.rstrip('/') + "/api/status?task_id=" + task_id)
    if response.status_code == 200:
        status = response.json()["status"]
        if status == "error":
            raise RuntimeError(response.json().get('error', 'Image generation failed.'))
        if status == "done":
            return 0
        else:
            # read the waiting time
            waiting_time = response.json().get("wait_remaining", 1)
            if waiting_time < 1: waiting_time = 1
            return waiting_time
    else:
        return -1

def mflux_get_image(mfluxendpoint, task_id):
    response = requests.get(mfluxendpoint.rstrip('/') + "/api/image?task_id=" + task_id + "&base64=false&delete=true")
    if response.status_code == 200:
        return response.content
    else:
        return None
    
def read_prompts(file_path):
    with open(file_path, 'r') as f:
        prompts = [json.loads(line.strip()) for line in f]
    return prompts

def select_random_prompt(prompts):
    return random.choice(prompts)

# 5120x2880 32'' 16:9 5K
# 5120x2160 34'' 21:9 UltraWide
# 5120x1440 34'' 21:9 Super UltraWide
# 3840x2160 32'' 16:9 4K UHD
# 3840x1600 34'' 21:9 UltraWide
# 3440x1440 34'' 21:9 UltraWide
# 2560x1600 30''  8:5 WQXGA   (fail)
# 2560x1440 27'' 16:9 Quad HD (Mac, fail)
# 1920x1200 23''  8:5 WUXGA   (106s/1 M1 Ultra)
# 1920x1080 21'' 16:9 Full HD ( 91s/1 M1 Ultra)
# 1280x1024 21''  5:4 SXGA    ( 50s/1 M1 Ultra)
# 1280x 960 20''  4:3 QuadVGA ( 49s/1 M1 Ultra)
# 1280x 800 20''  8:5 WXGA    ( 40s/1 M1 Ultra)
# 1280x 720 18'' 16:9 HD      ( 35s/1 151s/4 M1 Ultra)
# 1024x1024 16''  1:1 Dall.E  ( 40s/1 M1 Ultra)
# 1024x 768 15''  4:3 XGA     ( 31s/1 M1 Ultra)
#  800x 600  5''  4:3 SVGA    ( 21s/1 M1 Ultra)
#  768x 576  6''  4:3 PAL     ( 18s/1 M1 Ultra)
#  640x 480  4''  4:3 VGA     ( 14s/1 M1 Ultra)
#  640x 360  4'' 16:9 nHD     ( 11s/1 M1 Ultra)
#  320x 240  2''  4:3 QVGA    (  7s/1 M1 Ultra)
#  320x 200  2''  8:5 CGA     (  7s/1 M1 Ultra)
#  256x 256  1''  1:1         (  6s/1 M1 Ultra)

def generate_image(mfluxendpoint, prompt, width=1280, height=720, steps=4, count=1, init_images=None):
    print("### Generating Image(s)...")
    startt = time.time()
    task_id = mflux_generate_client(mfluxendpoint, prompt, width=width, height=height, steps=steps, count=count, init_images=init_images)
    for i in range(10000):
        waiting_time = mflux_status_ready(mfluxendpoint, task_id)
        print("Waiting time: ", waiting_time, " seconds")
        if waiting_time == 0: break
        nextsleep = max(min(waiting_time / 4, 10), 1)
        time.sleep(nextsleep)
    imageb = mflux_get_image(mfluxendpoint, task_id)    
    stopt = time.time()
    print("Time taken: ", stopt - startt, " seconds")
    # imageb is the image in bytes, convert to PIL image
    image = Image.open(BytesIO(imageb))
    return image

# compute a 4-digit hex hash to identify the image
def hash_image(image):
    return hashlib.md5(image.tobytes()).hexdigest()[:4]

# construct a filename for the image
def image_filename(tags, width, height, image):
    hash = hash_image(image)
    return f"wallpaper.cc0.photos_{tags}_{width}x{height}_{hash}_raw.jpg"

def save_images(path, out, tags):
    print("### Saving Image...")
    filename = image_filename(tags, out.width, out.height, out)
    with open(path + "/" + filename,"w+") as f:
        out.save(f)
        
#prompt = "A very fluffy, happy and smiling cat holding a beer bottle in one hand and a sign in the other hand that says 'Radeberger AI'"
#prompt = "Many happy yellow Rubber Ducks as a pattern on a green fabric. The ducks are smiling and have a red beak. The fabric is part of a shirt."
#prompt = "Many happy yellow Rubber Ducks as a pattern on a green t-shirt. The ducks are smiling and have a red beak. A teenage boy and a teenage girl are wearing the t-shirts with the Rubber ducks. In the background the is a school building."
prompt = "Ecology: Endangered Species, Mathematics: Combinatorics, Gadgets & Tech: Gaming Chairs"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from the factory prompt list.')
    parser.add_argument('--server', default='http://localhost:4030')
    parser.add_argument('--init-images', nargs='+', metavar='FILE', help='Ordered image inputs used for every prompt')
    args = parser.parse_args()
    app_path = os.path.dirname(os.path.abspath(__file__))
    prompts = read_prompts(app_path + "/image-generator-prompts.jsonlist")
    mfluxendpoint = args.server
    
    while True:
        selected = select_random_prompt(prompts)
        prompt = "photorealistic high-res picture, cinematic style with the following properties: " + selected["prompt"]
        #out = generate_image(mfluxendpoint, prompt, width=960, height=600, steps=4)
        out = generate_image(mfluxendpoint, prompt, width=1920, height=1200, steps=4, init_images=args.init_images)
        save_images(app_path + "/wallpapers", out, tags=selected["tags"])

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser(description='Load a model in Ollama.')
#    parser.add_argument('--stream', action='store_true', help='streamed loading') # only useful for debugging streams
#    parser.add_argument('--json', action='store_true', help='enable json parsing')
#    parser.add_argument('--seed', type=int, default=42, help='random seed')
#    parser.add_argument('--num_predict', type=int, default=300, help='number of tokens to predict')
#    parser.add_argument('--temperature', type=float, default=0.0, help='temperature for sampling')
#
#    args = parser.parse_args()
#
#    protocol = args.protocol
#    host = args.host
#    port = args.port
#    model = args.model
