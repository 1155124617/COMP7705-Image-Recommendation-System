from io import BytesIO

from PIL import Image

from django.shortcuts import render

from models.blip2 import recommend_images

img = None

# Create your views here.
def index(request):
    return render(request, "index.html", {})


def recommend(request):
    if request.method == 'POST':
        img_file = request.FILES['image']
        img = Image.open(img_file)

        image_path = 'image_recommend/temp/image.jpeg'
        img.save(image_path)
        
        img_data = BytesIO()
        img.save(img_data, format='JPEG')
        img_data.seek(0)

        return render(request, 'index.html', {'img_data': img_data})
    return render(request, 'index.html')


def recommend_similar(request):
    image_path = 'image_recommend/temp/image.jpeg'
    img = Image.open(image_path)

    if img is not None:
        imgs = recommend_images(img)

        # naive test
        img_output = BytesIO()
        imgs.save(img_output, format='jpeg')
        img_output.seek(0)
        return render(request, 'index.html', {'img_output': img_output})

    return render(request, 'index.html')


def transfer_styles(request):
    return render(request, 'index.html')
