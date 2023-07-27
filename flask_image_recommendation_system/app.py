import base64
import json
import os
import shutil

from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request
from flask_cors import CORS

from const.pathname import *
from style_transfer.S2WAT.style_transfer_model import do_style_transfer
# from rec_models.blip2 import recommend_images_to_files_list, recommend_images_to_urls, recommend_text_to_files_list, recommend_text_to_urls, get_random_image_urls
from test.test import recommend_images_to_urls, recommend_images_to_files_list, recommend_text_to_files_list, \
    recommend_text_to_urls, get_random_image_urls

app = Flask(__name__)
request_id = 0

CORS(app)


@app.route('/')
def index():
    images = get_random_image_urls()
    return render_template(INDEX_PAGE, image_wall=images)


@app.route('/recommend_page')
def recommend_page():
    return render_template(RECOMMEND_PAGE)


@app.route('/style_transfer_page')
def style_transfer_page():
    return render_template(STYLE_TRANSFER_PAGE)


@app.route('/recommend_similar', methods=['POST'])
def recommend_similar():
    data = request.json
    img = Image.open(BytesIO(base64.b64decode(data['image_data'])))

    if img is not None:
        image_show_list = recommend_images_to_urls(img)

        return json.dumps({'urls': image_show_list})

    print("img is None")
    return render_template(RECOMMEND_PAGE)


@app.route('/recommend_with_text', methods=['POST'])
def recommend_with_text():
    data = request.json
    text = data["text"]

    if text is not None:
        image_show_list = recommend_text_to_urls(text)

        return json.dumps({'urls': image_show_list})


@app.route('/transfer_styles', methods=['POST'])
def transfer_styles():
    data = request.json
    time_stamp = str(data['time_stamp'])

    # Prepare input directory
    if not os.path.exists(os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp)):
        os.makedirs(os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp))
    # Prepare output directory
    if not os.path.exists(os.path.join(OUTPUT_IMAGE_DIR, time_stamp)):
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, time_stamp))

    image_resize(Image.open(BytesIO(base64.b64decode(data['content_image_data'])))).save(
        os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp, UPLOADED_IMAGE_NAME))

    # Clear output directory
    # rm_rf_directory(OUTPUT_IMAGE_DIR)

    do_style_transfer(
        input_content_image_dir=os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp),
        input_style_image_dir=DEFAULT_STYLE_IMAGE_DIR,
        output_image_dir=os.path.join(OUTPUT_IMAGE_DIR, time_stamp)
    )

    image_show_list = []
    output_image_names = os.listdir(os.path.join(OUTPUT_IMAGE_DIR, time_stamp))
    for output_image_name in output_image_names:
        image_show_list.append(base64_encode(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, time_stamp, output_image_name))))

    return json.dumps({'urls': image_show_list})


@app.route('/transfer_given_style', methods=['POST'])
def transfer_given_style():
    # Clear output directory
    # rm_rf_directory(OUTPUT_IMAGE_DIR)
    data = request.json
    time_stamp = str(data['time_stamp'])

    # Prepare input directory
    if not os.path.exists(os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp)):
        os.makedirs(os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp))
    if not os.path.exists(os.path.join(INPUT_STYLE_IMAGE_DIR, time_stamp)):
        os.makedirs(os.path.join(INPUT_STYLE_IMAGE_DIR, time_stamp))
    # Prepare output directory
    if not os.path.exists(os.path.join(OUTPUT_IMAGE_DIR, time_stamp)):
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, time_stamp))

    image_resize(Image.open(BytesIO(base64.b64decode(data['content_image_data'])))).save(
        os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp, UPLOADED_IMAGE_NAME))
    image_resize(Image.open(BytesIO(base64.b64decode(data['style_image_data'])))).save(
        os.path.join(INPUT_STYLE_IMAGE_DIR, time_stamp, INPUT_STYLE_IMAGE_NAME))

    do_style_transfer(
        input_content_image_dir=os.path.join(INPUT_CONTENT_IMAGE_DIR, time_stamp),
        input_style_image_dir=os.path.join(INPUT_STYLE_IMAGE_DIR, time_stamp),
        output_image_dir=os.path.join(OUTPUT_IMAGE_DIR, time_stamp)
    )

    image_show_list = []
    output_image_names = os.listdir(os.path.join(OUTPUT_IMAGE_DIR, time_stamp))
    for output_image_name in output_image_names:
        image_show_list.append(
            base64_encode(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, time_stamp, output_image_name))))

    return json.dumps({'urls': image_show_list})


@app.route('/mobile_recommend', methods=['POST'])
def mobile_recommend():
    if 'image' not in request.files:
        return "No image file in the request", 400

    file = request.files['image']
    img = Image.open(file)

    recommended_image_urls = recommend_images_to_urls(img)
    return recommended_image_urls


@app.route('/mobile_recommend_with_text', methods=['POST'])
def mobile_recommend_with_text():
    text = request.form['search_text']

    return recommend_text_to_urls(text)


@app.route('/mobile_transfer_styles', methods=['POST'])
def mobile_transfer_styles():
    if 'image' not in request.files:
        return "No image file in the request", 400

    img_file = request.files['image']
    img = Image.open(img_file)

    # Prepare input directory
    if not os.path.exists(os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE')):
        os.makedirs(os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE'))
    # Prepare output directory
    if not os.path.exists(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE')):
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE'))

    image_path = os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE', UPLOADED_IMAGE_NAME)
    image_resize(img).save(image_path)

    # Clear output directory
    # rm_rf_directory(OUTPUT_IMAGE_DIR)

    do_style_transfer(
        input_content_image_dir=os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE'),
        input_style_image_dir=DEFAULT_STYLE_IMAGE_DIR,
        output_image_dir=os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE')
    )

    image_show_list = []
    output_image_names = os.listdir(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE'))
    for output_image_name in output_image_names:
        image_show_list.append(base64.encodebytes(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE', output_image_name))
                                                  .getvalue()).decode('utf-8'))
    return image_show_list


@app.route('/mobile_transfer_given_style', methods=['POST'])
def mobile_transfer_given_style():
    if 'content_image' not in request.files and 'style_image' not in request.files:
        return "Please upload content image and style image", 400

    content_image = Image.open(request.files['content_image'])
    style_image = Image.open(request.files['style_image'])

    # Prepare input directory
    if not os.path.exists(os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE')):
        os.makedirs(os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE'))
    if not os.path.exists(os.path.join(INPUT_STYLE_IMAGE_DIR, 'MOBILE')):
        os.makedirs(os.path.join(INPUT_STYLE_IMAGE_DIR, 'MOBILE'))
    # Prepare output directory
    if not os.path.exists(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE')):
        os.makedirs(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE'))

    image_resize(content_image).save(os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE', UPLOADED_IMAGE_NAME))
    image_resize(style_image).save(os.path.join(INPUT_STYLE_IMAGE_DIR, 'MOBILE', INPUT_STYLE_IMAGE_NAME))

    # Clear output directory
    # rm_rf_directory(OUTPUT_IMAGE_DIR)

    do_style_transfer(
        input_content_image_dir=os.path.join(INPUT_CONTENT_IMAGE_DIR, 'MOBILE', UPLOADED_IMAGE_NAME),
        input_style_image_dir=os.path.join(INPUT_STYLE_IMAGE_DIR, 'MOBILE', INPUT_STYLE_IMAGE_NAME),
        output_image_dir=os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE')
    )

    image_show_list = []
    output_image_names = os.listdir(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE'))
    for output_image_name in output_image_names:
        image_show_list.append(base64.encodebytes(open_image_bytesio(os.path.join(OUTPUT_IMAGE_DIR, 'MOBILE', output_image_name))
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
        shutil.rmtree(os.path.join(directory_path, directory))


def image_resize(im, base=200, resampling_method=Image.Resampling.LANCZOS):
    im.thumbnail((base, base), resampling_method)
    return im


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
