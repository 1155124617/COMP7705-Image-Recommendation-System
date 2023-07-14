import base64
import os
import shutil
from io import BytesIO

from PIL import Image
from flask import Flask, render_template, request
from flask_cors import CORS

from const.pathname import *
# from rec_models.blip2 import recommend_images_to_files_list, recommend_images_to_urls, recommend_text_to_files_list
from test.test import recommend_images_to_urls, recommend_images_to_files_list, recommend_text_to_files_list
from style_transfer.S2WAT.style_transfer_model import do_style_transfer

app = Flask(__name__)
request_id = 0

CORS(app)

@app.route('/')
def index():
    return render_template(INDEX_PAGE)


@app.route('/recommend_page')
def recommend_page():
    return render_template(RECOMMEND_TRANSFER_PAGE)


@app.route('/given_style_page')
def given_style_page():
    return render_template(GIVEN_STYLE_TRANSFER_PAGE)


@app.route('/upload_single_image', methods=['POST'])
def upload_single_image():
    img_file = request.files['image']
    img = Image.open(img_file)

    image_path = os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME)
    img.save(image_path)

    img_data = BytesIO()
    img.save(img_data, format='JPEG')
    img_data.seek(0)

    return render_template(RECOMMEND_TRANSFER_PAGE, img_data=img_data)


@app.route('/recommend_similar')
def recommend_similar():
    uploaded_image_path = os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME)
    img = Image.open(uploaded_image_path)

    if img is not None:
        image_show_list = []
        for image_path in recommend_images_to_files_list(img):
            image_show_list.append(open_image_bytesio(image_path))

        uploaded_image_data = open_image_bytesio(uploaded_image_path)
        return render_template(RECOMMEND_TRANSFER_PAGE, images_output_list=image_show_list, img_data=uploaded_image_data)

    return render_template(RECOMMEND_TRANSFER_PAGE)


@app.route('/recommend_with_text', methods=['POST'])
def recommend_with_text():
    text = request.form['search_text']

    if text is not None:
        image_show_list = []
        for image_path in recommend_text_to_files_list(text):
            image_show_list.append(open_image_bytesio(image_path))

        return render_template(RECOMMEND_TRANSFER_PAGE, images_output_list=image_show_list)


@app.route('/transfer_styles')
def transfer_styles():
    uploaded_image_data = open_image_bytesio(os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    Image.open(os.path.join(UPLOADED_IMAGE_DIR, UPLOADED_IMAGE_NAME)).resize((200, 200)).save(os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME))

    # Clear output directory
    rm_rf_directory(OUTPUT_IMAGE_DIR)

    do_style_transfer()
    image_show_list = []
    output_image_names = os.listdir(OUTPUT_IMAGE_DIR)
    for output_image_name in output_image_names:
        image_show_list.append(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, output_image_name)))

    return render_template(RECOMMEND_TRANSFER_PAGE, img_data=uploaded_image_data, images_output_list=image_show_list)


@app.route('/upload_content_style_images', methods=['POST'])
def upload_content_style_images():
    content_image = Image.open(request.files['content_image'])
    style_image = Image.open(request.files['style_image'])

    content_image.resize((200, 200)).save(os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    style_image.resize((200, 200)).save(os.path.join(INPUT_STYLE_IMAGE_DIR, INPUT_STYLE_IMAGE_NAME))

    content_image_stream = open_image_bytesio(os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    style_image_stream = open_image_bytesio(os.path.join(INPUT_STYLE_IMAGE_DIR, INPUT_STYLE_IMAGE_NAME))

    return render_template(GIVEN_STYLE_TRANSFER_PAGE, content_image=content_image_stream, style_image=style_image_stream)


@app.route('/transfer_given_style')
def transfer_given_style():
    # Clear output directory
    rm_rf_directory(OUTPUT_IMAGE_DIR)

    do_style_transfer(INPUT_STYLE_IMAGE_DIR)
    image_show_list = []
    output_image_names = os.listdir(OUTPUT_IMAGE_DIR)
    for output_image_name in output_image_names:
        image_show_list.append(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, output_image_name)))

    content_image_stream = open_image_bytesio(os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    style_image_stream = open_image_bytesio(os.path.join(INPUT_STYLE_IMAGE_DIR, INPUT_STYLE_IMAGE_NAME))
    return render_template(GIVEN_STYLE_TRANSFER_PAGE, content_image=content_image_stream,
                           style_image=style_image_stream, images_output_list=image_show_list)


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
    img.resize((200, 200)).save(image_path)

    # Clear output directory
    rm_rf_directory(OUTPUT_IMAGE_DIR)

    do_style_transfer()
    image_show_list = []
    output_image_names = os.listdir(OUTPUT_IMAGE_DIR)
    for output_image_name in output_image_names:
        image_show_list.append(base64.encodebytes(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, output_image_name))
                                                  .getvalue()).decode('utf-8'))
    return image_show_list


@app.route('/mobile_transfer_given_style', methods=['POST'])
def mobile_transfer_given_style():
    if 'content_image' not in request.files and 'style_image' not in request.files:
        return "Please upload content image and style image", 400

    content_image = Image.open(request.files['content_image'])
    style_image = Image.open(request.files['style_image'])

    content_image.resize((200, 200)).save(os.path.join(INPUT_CONTENT_IMAGE_DIR, UPLOADED_IMAGE_NAME))
    style_image.resize((200, 200)).save(os.path.join(INPUT_STYLE_IMAGE_DIR, INPUT_STYLE_IMAGE_NAME))

    # Clear output directory
    rm_rf_directory(OUTPUT_IMAGE_DIR)

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


def rm_rf_directory(directory_path):
    dirs = os.listdir(directory_path)
    for directory in dirs:
        os.remove(os.path.join(directory_path, directory))


if __name__ == '__main__':
    # Remove all the files stored before
    # Remove recommendation uploaded files
    rm_rf_directory(UPLOADED_IMAGE_DIR)
    # Remove style transfer input images
    rm_rf_directory(INPUT_CONTENT_IMAGE_DIR)
    rm_rf_directory(INPUT_STYLE_IMAGE_DIR)
    # Remove output images
    rm_rf_directory(OUTPUT_REC_DIR)
    rm_rf_directory(OUTPUT_IMAGE_DIR)

    # Start App
    app.run(host='0.0.0.0', port=8000)

