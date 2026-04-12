from flask import Flask, request, jsonify, render_template
import requests
import base64

app = Flask(__name__)

# ── Put your Imagga keys here ──
API_KEY    = "acc_f80df2832dc1070"
API_SECRET = "b617b60799175cee59e0bc66671bbed4"

# Living things keywords
LIVING_KEYWORDS = [
    'person', 'man', 'woman', 'child', 'boy', 'girl', 'human', 'people',
    'dog', 'cat', 'bird', 'horse', 'elephant', 'bear', 'zebra', 'giraffe',
    'cow', 'sheep', 'monkey', 'lion', 'tiger', 'fish', 'shark', 'whale',
    'frog', 'snake', 'rabbit', 'deer', 'duck', 'eagle', 'parrot', 'wolf',
    'face', 'baby', 'adult', 'portrait', 'animal', 'pet', 'wildlife',
    'kid', 'teenager', 'toddler', 'infant', 'male', 'female', 'student'
]

def is_living(label):
    return any(word in label.lower() for word in LIVING_KEYWORDS)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file         = request.files['file']
    image_bytes  = file.read()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')

    # Call Imagga API
    response = requests.post(
        'https://api.imagga.com/v2/tags',
        auth=(API_KEY, API_SECRET),
        data={'image_base64': image_base64}
    )

    result = response.json()
    print("\n── Full API Response ──")
    print(result)

    if 'result' not in result:
        return jsonify({'error': 'API failed. Check your API key!'}), 500

    tags = result['result']['tags']

    # Print all tags in CMD
    print("\n── Imagga Tags ──")
    for tag in tags[:10]:
        print(f"  {tag['tag']['en']} → {tag['confidence']:.2f}%")

    # Separate into living and objects
    living_things = []
    objects       = []

    for tag in tags[:20]:
        label      = tag['tag']['en']
        confidence = f"{tag['confidence']:.2f}%"

        if is_living(label):
            living_things.append({
                'label':      label.title(),
                'confidence': confidence
            })
        else:
            objects.append({
                'label':      label.title(),
                'confidence': confidence
            })

    living_things = living_things[:3]
    objects       = objects[:3]

    return jsonify({
        'living_things': living_things,
        'objects':       objects
    })

if __name__ == '__main__':
    app.run(debug=True)