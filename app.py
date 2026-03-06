import os
import tempfile
import base64
import requests
from flask import Flask, request, jsonify
import torch
import whisper
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import noisereduce as nr

app = Flask(__name__)

# ---------- 全局加载模型（只在启动时加载一次）----------
print("正在加载 Whisper 模型...")
whisper_model = whisper.load_model("small")  # 可根据需要选择模型大小

print("正在加载 ControlNet 模型...")
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_scribble",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

print("正在加载 Stable Diffusion 管道...")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "lambdalabs/sd-pokemon-diffusers",  # 可替换为你的风格模型
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

# 根据是否有 GPU 选择设备
if torch.cuda.is_available():
    pipe = pipe.to("cuda")
    print("使用 GPU 加速")
else:
    pipe = pipe.to("cpu")
    print("使用 CPU，速度可能较慢")

# 中文关键词映射（可扩充）
KEYWORD_MAP = {
    "狗": "dog", "猫": "cat", "鸟": "bird", "鱼": "fish", "兔子": "rabbit",
    "熊": "bear", "大象": "elephant", "猴子": "monkey", "汽车": "car",
    "房子": "house", "树": "tree", "花": "flower", "太阳": "sun",
    "云": "cloud", "可爱": "cute", "白色": "white", "黑色": "black",
    "小": "small", "大": "big",
}

@app.route('/ping')
def ping():
    return 'pong'

@app.route('/')
def index():
    return jsonify({'message': 'Hello from Doodle-Den!', 'status': 'running'})

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        sketch_url = data.get('sketch_url')
        voice_url = data.get('voice_url')
        if not sketch_url or not voice_url:
            return jsonify({'success': False, 'message': '缺少 sketch_url 或 voice_url'}), 400

        # 1. 下载草图
        sketch_resp = requests.get(sketch_url)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(sketch_resp.content)
            sketch_path = f.name

        # 2. 下载语音
        voice_resp = requests.get(voice_url)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(voice_resp.content)
            voice_path = f.name

        # 3. 处理草图提取线条
        sketch_img = Image.open(sketch_path).convert("RGB").resize((512, 512))
        image_np = np.array(sketch_img)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        condition_img = Image.fromarray(edges)

        # 4. 语音识别
        result = whisper_model.transcribe(voice_path, language="zh")
        user_text = result["text"]

        # 5. 构建提示词
        detected_keywords_en = []
        for ch, en in KEYWORD_MAP.items():
            if ch in user_text:
                detected_keywords_en.append(en)
        style_part = "Studio Pokemon style, anime, vibrant colors, soft lighting, magical atmosphere, highly detailed, cute, children's book illustration"
        if detected_keywords_en:
            keyword_part = ", ".join(detected_keywords_en)
            final_prompt = f"{user_text}, {keyword_part}, {style_part}"
        else:
            final_prompt = f"{user_text}, {style_part}"

        # 6. 生成图像
        negative_prompt = "blurry, messy, ugly, deformed, extra limbs, bad anatomy, black and white, line drawing, sketch, realistic, photo"
        generated_img = pipe(
            final_prompt,
            image=condition_img,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.5,
            controlnet_conditioning_scale=1.0,
        ).images[0]

        # 7. 转为 base64 返回
        img_io = io.BytesIO()
        generated_img.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')

        # 清理临时文件
        os.unlink(sketch_path)
        os.unlink(voice_path)

        return jsonify({
            'success': True,
            'image': img_base64,
            'prompt': final_prompt
        })

    except Exception as e:
        print(f"生成失败: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)