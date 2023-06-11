from flask import Flask, render_template, request
from io import BytesIO
from PIL import Image
from models.blip2 import recommend_images

import base64

app = Flask(__name__)

img = None


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        img_file = request.files['image']
        img = Image.open(img_file)

        image_path = 'temp/image.jpeg'
        img.save(image_path)

        img_data = BytesIO()
        img.save(img_data, format='JPEG')
        img_data.seek(0)

        return render_template('index.html', img_data=img_data)

    return render_template('index.html')


@app.route('/recommend_similar')
def recommend_similar():
    image_path = 'temp/image.jpeg'
    img = Image.open(image_path)

    if img is not None:
        images_output_list = []
        for image_path in recommend_images(img):
            images_output_list.append(open_image_bytesio(image_path))

        img_data = open_image_bytesio('temp/image.jpeg')
        return render_template('index.html', images_output_list=images_output_list, img_data=img_data)

    return render_template('index.html')


@app.route('/transfer_styles')
def transfer_styles():
    return render_template('index.html')


# 自定义过滤器
@app.template_filter('base64_encode')
def base64_encode(data):
    return base64.b64encode(data.getvalue()).decode('utf-8')


def open_image_bytesio(image_path):
    img = Image.open(image_path)
    img_output = BytesIO()
    img.save(img_output, format='jpeg')
    img_output.seek(0)

    return img_output


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
