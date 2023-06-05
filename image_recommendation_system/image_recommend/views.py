from io import BytesIO

from PIL import Image

from django.shortcuts import render


# Create your views here.
def index(request):
    return render(request, "index.html", {})


def recommend(request):
    if request.method == 'POST':
        img_file = request.FILES['image']
        img = Image.open(img_file)
        img_data = BytesIO()
        img.save(img_data, format='JPEG')
        img_data.seek(0)

        return render(request, 'index.html', {'img_data': img_data})
    return render(request, 'index.html')


def recommend_similar(request):
    return render(request, 'index.html')


def transfer_styles(request):
    return render(request, 'index.html')
