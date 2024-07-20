from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO
import os

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            img_array = img_array.reshape((1, 224, 224, 3))
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)
            predictions = model.predict(img_array)
            decoded_predictions = decode_predictions(predictions, top=5)[0]
            formatted_predictions = [(pred[1], f"{pred[2] * 100:.2f}") for pred in decoded_predictions]
            img_data = request.POST.get('img_data')
            return render(request, 'home.html', {'form': form, 'prediction': formatted_predictions, 'img_data': img_data})
        else:
            form = ImageUploadForm()
            return render(request, 'home.html', {'form': form})