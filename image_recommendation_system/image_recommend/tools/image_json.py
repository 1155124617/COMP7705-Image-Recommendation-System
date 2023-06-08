from PIL import Image
import json

def image_to_json(image):
    # Convert the image to a byte array
    print('converting images to byte array')
    image_bytes = image.tobytes()

    # Create a dictionary with the necessary information
    data = {
        'mode': image.mode,
        'size': image.size,
        'data': image_bytes.decode('latin-1')
    }

    # Serialize the dictionary to JSON
    print('serializing the dictionary to JSON')
    json_data = json.dumps(data)
    return json_data


def json_to_image(json_data):
    # Parse the JSON string back to a dictionary
    data = json.loads(json_data)

    # Extract the necessary information
    mode = data['mode']
    size = tuple(data['size'])
    image_bytes = data['data'].encode('latin-1')

    # Create a PIL Image object from the byte array
    image = Image.frombytes(mode, size, image_bytes)
    return image
