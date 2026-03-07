import os
import io
import tempfile
import base64
import traceback

import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

# AI 相关库
import torch
import whisper
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import noisereduce as nr

app = Flask(__name__)
CORS(app)  # 允许跨域请求（可选）

# ========== 全局加载模型（仅启动时加载一次）==========
print("正在加载 Whisper 模型...")
try:
    whisper_model = whisper.load_model("small")  # 可根据需要改为 "tiny" 或 "base"
    print("Whisper 模型加载成功")
except Exception as e:
    print(f"Whisper 模型加载失败: {e}")
    whisper_model = None

print("正在加载 ControlNet 模型...")
try:
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    print("ControlNet 模型加载成功")
except Exception as e:
    print(f"ControlNet 模型加载失败: {e}")
    controlnet = None

print("正在加载 Stable Diffusion 管道...")
try:
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "lambdalabs/sd-pokemon-diffusers",  # 可替换为其他风格模型
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
        print("使用 GPU 加速")
    else:
        pipe = pipe.to("cpu")
        print("使用 CPU 运行，速度可能较慢")
    print("Stable Diffusion 管道加载成功")
except Exception as e:
    print(f"Stable Diffusion 管道加载失败: {e}")
    pipe = None

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
    # 记录请求开始
    print("="*50)
    print("收到 /generate 请求")
    
    # 检查模型是否加载成功
    if whisper_model is None or controlnet is None or pipe is None:
        error_msg = "模型未正确加载，请检查服务器日志"
        print(error_msg)
        return jsonify({'success': False, 'message': error_msg}), 500

    try:
        data = request.get_json()
        if not data:
            print("错误：未收到 JSON 数据")
            return jsonify({'success': False, 'message': '请求体必须为 JSON'}), 400

        sketch_url = data.get('sketch_url')
        voice_url = data.get('voice_url')
        print(f"sketch_url: {sketch_url}")
        print(f"voice_url: {voice_url}")

        if not sketch_url or not voice_url:
            print("错误：缺少 sketch_url 或 voice_url")
            return jsonify({'success': False, 'message': '缺少 sketch_url 或 voice_url'}), 400

        # 1. 下载草图
        print("正在下载草图...")
        sketch_resp = requests.get(sketch_url, timeout=30)
        if sketch_resp.status_code != 200:
            print(f"草图下载失败，状态码: {sketch_resp.status_code}")
            return jsonify({'success': False, 'message': '草图下载失败'}), 400
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            f.write(sketch_resp.content)
            sketch_path = f.name
        print(f"草图保存至临时文件: {sketch_path}")

        # 2. 下载语音
        print("正在下载语音...")
        voice_resp = requests.get(voice_url, timeout=30)
        if voice_resp.status_code != 200:
            print(f"语音下载失败，状态码: {voice_resp.status_code}")
            os.unlink(sketch_path)
            return jsonify({'success': False, 'message': '语音下载失败'}), 400
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(voice_resp.content)
            voice_path = f.name
        print(f"语音保存至临时文件: {voice_path}")

        # 3. 处理草图，提取边缘作为 ControlNet 条件
        print("正在提取草图边缘...")
        sketch_img = Image.open(sketch_path).convert("RGB").resize((512, 512))
        image_np = np.array(sketch_img)
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = edges[:, :, None]
        edges = np.concatenate([edges, edges, edges], axis=2)
        condition_img = Image.fromarray(edges)
        print("边缘提取完成")

        # 4. 语音识别
        print("正在进行语音识别...")
        result = whisper_model.transcribe(voice_path, language="zh")
        user_text = result["text"]
        print(f"识别结果: {user_text}")

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
        print(f"最终提示词: {final_prompt}")

        # 6. 生成图像
        print("正在生成图像（这可能需要一段时间）...")
        negative_prompt = "blurry, messy, ugly, deformed, extra limbs, bad anatomy, black and white, line drawing, sketch, realistic, photo"
        generated_img = pipe(
            final_prompt,
            image=condition_img,
            negative_prompt=negative_prompt,
            num_inference_steps=40,
            guidance_scale=8.5,
            controlnet_conditioning_scale=1.0,
        ).images[0]
        print("图像生成完成")

        # 7. 转换为 base64
        img_io = io.BytesIO()
        generated_img.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        print("图像已转换为 base64")

        # 清理临时文件
        os.unlink(sketch_path)
        os.unlink(voice_path)
        print("临时文件已清理")

        # 8. 返回成功响应
        return jsonify({
            'success': True,
            'image': img_base64,
            'prompt': final_prompt
        })

    except Exception as e:
        # 打印详细错误信息到日志
        print("生成过程中发生异常:")
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port, debug=False)
