import base64
import os
import shutil
from io import BytesIO

from PIL import Image
from flask import Flask, render_template, request
from flask_cors import CORS

from const.pathname import *
from style_transfer.S2WAT.style_transfer_model import do_style_transfer
# from models.blip2 import recommend_images_to_files_list, recommend_images_to_urls
from test.test import recommend_images_to_urls, recommend_images_to_files_list

app = Flask(__name__)

CORS(app)

img = None


@app.route('/')
def index():
    return render_template(INDEX_PAGE)


@app.route('/recommend', methods=['POST'])
def recommend():
    if request.method == 'POST':
        img_file = request.files['image']
        img = Image.open(img_file)

        image_path = os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME)
        img.save(image_path)

        img_data = BytesIO()
        img.save(img_data, format='JPEG')
        img_data.seek(0)

        return render_template(INDEX_PAGE, img_data=img_data)

    return render_template(INDEX_PAGE)


@app.route('/recommend_similar')
def recommend_similar():
    uploaded_image_path = os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME)
    img = Image.open(uploaded_image_path)

    if img is not None:
        image_show_list = []
        for image_path in recommend_images_to_files_list(img):
            image_show_list.append(open_image_bytesio(image_path))

        uploaded_image_data = open_image_bytesio(uploaded_image_path)
        return render_template(INDEX_PAGE, images_output_list=image_show_list, img_data=uploaded_image_data)

    return render_template(INDEX_PAGE)


@app.route('/transfer_styles')
def transfer_styles():
    uploaded_image_data = open_image_bytesio(os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    shutil.move(os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME),
                os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    do_style_transfer()
    image_show_list = []
    output_image_names = os.listdir(OUTPUT_IMAGE_DIR)
    for output_image_name in output_image_names:
        image_show_list.append(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, output_image_name)))

    return render_template(INDEX_PAGE, img_data=uploaded_image_data, images_output_list=image_show_list)


@app.route('/mobile_recommend', methods=['POST'])
def mobile_recommend():
    if 'image' not in request.files:
        return "No image file in the request", 400

    file = request.files['image']
    img = Image.open(file)

    recommended_image_urls = recommend_images_to_urls(img)
    return recommended_image_urls


@app.route('/mobile_transfer_styles', methods=['POST'])
def mobile_transfer_styles():
    if 'image' not in request.files:
        return "No image file in the request", 400

    img_file = request.files['image']
    img = Image.open(img_file)

    image_path = os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME)
    img.save(image_path)

    do_style_transfer()
    image_show_list = []
    output_image_names = os.listdir(OUTPUT_IMAGE_DIR)
    for output_image_name in output_image_names:
        image_show_list.append(base64.encodebytes(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, output_image_name))
                                                  .getvalue()).decode('utf-8'))
    return image_show_list


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

