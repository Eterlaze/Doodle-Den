import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({
        'message': 'Hello from Doodle-Den!',
        'status': 'running'
    })

@app.route('/ping')
def ping():
    # 简单的健康检查接口
    return 'pong'

@app.route('/generate', methods=['POST'])
def generate():
    # 占位接口，后续再实现真正的生成逻辑
    return jsonify({
        'success': False,
        'message': 'Not implemented yet'
    })

if __name__ == '__main__':
    # 云托管要求监听 0.0.0.0 和端口 80
    port = int(os.environ.get('PORT', 80))  # 允许通过环境变量覆盖端口
    app.run(host='0.0.0.0', port=port, debug=False)