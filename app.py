from flask import Flask, request, jsonify
import joblib
import sys
import traceback

app = Flask(__name__)

@app.route('/', methods=['POST'])
def predict():
    try:
        json_ = request.json
        title = json_['title']  # Fixed: expect {'title': 'news headline'}
        
        # Load model (once, globally)
        if not hasattr(app, 'model'):
            app.model = joblib.load('model.pkl')
        
        prediction = app.model.predict([title])[0]
        return jsonify({
            'prediction': int(prediction),  # 0=Fake, 1=Real
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = 12345
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    
    print(f"[INFO] * Loading 'model.pkl' model")
    app.model = joblib.load('model.pkl')  # Load once
    print("[INFO] * Starting Flask server on port", port)
    app.run(host='0.0.0.0', port=port, debug=True)
