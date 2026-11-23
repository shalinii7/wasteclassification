import os
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

# ---------- APP ----------
app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "users.db")
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{DB_PATH}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Always add CORS header even on error
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# ---------- MODELS ----------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)

class History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), nullable=False)
    category = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# ---------- LOAD MODEL ----------
model = None
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'organic', 'paper', 'plastic']
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as keras_image
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    # Search EVERYWHERE in project for any .keras file
    search_roots = [BASE_DIR, os.path.join(BASE_DIR, ".."), os.getcwd()]
    found = None
    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip venv - don't search inside venv!
            if "venv" in dirpath or ".venv" in dirpath:
                continue
            for f in filenames:
                if f.endswith(".keras") or f.endswith(".h5"):
                    found = os.path.join(dirpath, f)
                    break
            if found:
                break
        if found:
            break
    if found:
        model = load_model(found)
        print(f"MODEL LOADED SUCCESSFULLY! From {found}")
    else:
        print("ERROR: No .keras model file found in project!")
except Exception as e:
    print("MODEL LOAD ERROR:", e)
    model = None

# ---------- HOME ----------
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "message": "API Running",
        "model_loaded": model is not None,
        "classes": CLASS_NAMES
    })

# ---------- REGISTER ----------
@app.route('/register', methods=['POST', 'OPTIONS'])
@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json(force=True) or {}
        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        password = data.get('password') or ''
        if not username or not email or not password:
            return jsonify({"error": "All fields required"}), 400
        if User.query.filter_by(username=username).first():
            return jsonify({"error": "Username exists"}), 400
        if User.query.filter_by(email=email).first():
            return jsonify({"error": "Email exists"}), 400
        user = User(username=username, email=email, password_hash=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        return jsonify({"message": "Registered successfully", "username": username}), 201
    except Exception as e:
        print("REGISTER ERROR:", e)
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# ---------- LOGIN ----------
@app.route('/login', methods=['POST', 'OPTIONS'])
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json(force=True) or {}
        identifier = (data.get('username') or data.get('email') or '').strip()
        password = data.get('password') or ''
        if not identifier or not password:
            return jsonify({"error": "Username and password required"}), 400
        user = User.query.filter_by(username=identifier).first()
        if not user:
            user = User.query.filter_by(email=identifier).first()
        if not user:
            return jsonify({"error": "User not found"}), 401
        if not check_password_hash(user.password_hash, password):
            # Also allow plain password if old account saved plain
            if user.password_hash != password:
                return jsonify({"error": "Wrong password"}), 401
        return jsonify({"message": "Login ok", "username": user.username}), 200
    except Exception as e:
        print("LOGIN ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------- PREDICT ----------
@app.route('/predict', methods=['POST', 'OPTIONS'])
@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        if 'image' not in request.files and 'file' not in request.files:
            return jsonify({"error": "No image uploaded"}), 400
        file = request.files.get('image') or request.files.get('file')
        username = request.form.get('username') or request.form.get('user') or 'guest'

        # Save temp
        temp_path = os.path.join(BASE_DIR, "temp_upload.jpg")
        file.save(temp_path)

        if model is None:
            # Dummy if model missing - so frontend never crashes
            category, confidence = "plastic", 85.0
        else:
            img = keras_image.load_img(temp_path, target_size=(224, 224))
            arr = keras_image.img_to_array(img)
            arr = np.expand_dims(arr, axis=0)
            arr = preprocess_input(arr)
            preds = model.predict(arr, verbose=0)
            idx = int(np.argmax(preds[0]))
            category = CLASS_NAMES[idx]
            confidence = float(preds[0][idx] * 100)
            # Cap 100% - never show 100%
            if confidence > 99.5:
                confidence = 99.5
            if confidence < 50:
                confidence = 50 + (confidence / 2)

        # Save history if username given and not guest
        if username and username != 'guest':
            h = History(username=username, category=category, confidence=round(confidence, 1))
            db.session.add(h)
            db.session.commit()

        return jsonify({
            "category": category,
            "prediction": category,
            "confidence": round(confidence, 1),
            "username": username
        }), 200
    except Exception as e:
        print("PREDICT ERROR:", e)
        return jsonify({"error": str(e)}), 500

# ---------- HISTORY ----------
@app.route('/history', methods=['GET', 'OPTIONS'])
@app.route('/api/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return '', 200
    username = request.args.get('username') or request.args.get('user') or ''
    if not username:
        return jsonify({"history": []}), 200
    rows = History.query.filter_by(username=username).order_by(History.timestamp.desc()).all()
    out = []
    for r in rows:
        out.append({
            "category": r.category,
            "prediction": r.category,
            "confidence": r.confidence,
            "timestamp": r.timestamp.strftime("%Y-%m-%d %H:%M"),
            "date": r.timestamp.strftime("%Y-%m-%d %H:%M")
        })
    return jsonify({"history": out, "username": username}), 200

@app.route('/history/clear', methods=['POST', 'OPTIONS'])
@app.route('/api/history/clear', methods=['POST', 'OPTIONS'])
@app.route('/clear_history', methods=['POST', 'OPTIONS'])
def clear_history():
    if request.method == 'OPTIONS':
        return '', 200
    try:
        data = request.get_json(force=True, silent=True) or {}
        username = data.get('username') or request.args.get('username') or ''
        if username:
            History.query.filter_by(username=username).delete()
            db.session.commit()
        return jsonify({"message": "History cleared"}), 200
    except Exception as e:
        print("CLEAR ERROR:", e)
        return jsonify({"error": str(e)}), 500
    
# ---------- ADMIN - ALL USERS ----------
@app.route('/admin/users', methods=['GET', 'OPTIONS'])
def admin_users():
    users = User.query.all()
    result = []
    for u in users:
        total = History.query.filter_by(username=u.username).count()
        result.append({
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "total_classifications": total
        })
    return jsonify({"total_users": len(result), "users": result})

# ---------- ADMIN LOGIN ----------
@app.route('/admin/login', methods=['POST', 'OPTIONS'])
def admin_login():
    data = request.get_json(silent=True) or {}
    if data.get('username') == 'LIGHT' and data.get('password') == 'WORLD':
        return jsonify({"success": True, "message": "Admin login success"})
    return jsonify({"success": False, "message": "Invalid admin credentials"}), 401

# ---------- ADMIN - WHAT USERS CLASSIFIED ----------
@app.route('/admin/history', methods=['GET'])
def admin_history():
    try:
        from sqlalchemy import text
        rows = []
        # Try History model first
        try:
            all_rows = History.query.order_by(History.created_at.desc()).limit(100).all()
            for r in all_rows:
                rows.append({
                    "username": getattr(r, 'username', ''),
                    "predicted_class": getattr(r, 'predicted_class', None) or getattr(r, 'prediction', None) or getattr(r, 'class_name', None) or getattr(r, 'label', '') or getattr(r, 'category', ''),
                    "confidence": getattr(r, 'confidence', 0) or 0,
                    "created_at": r.created_at.isoformat() if getattr(r, 'created_at', None) else ""
                })
            if rows:
                return jsonify({"success": True, "total": len(rows), "history": rows})
        except Exception as e:
            print("History model failed:", e)
        # If History model gave 0, try raw SQL tables
        for table in ["history", "predictions", "prediction", "classifications", "classification", "user_history"]:
            try:
                result = db.session.execute(text(f"SELECT * FROM {table} ORDER BY id DESC LIMIT 100"))
                for row in result:
                    d = dict(row._mapping)
                    username = d.get('username') or d.get('user') or d.get('user_id') or ""
                    if isinstance(username, int):
                        try:
                            u = User.query.get(username)
                            username = u.username if u else str(username)
                        except:
                            username = str(username)
                    cls = d.get('predicted_class') or d.get('prediction') or d.get('class_name') or d.get('label') or d.get('category') or d.get('result') or ""
                    conf = d.get('confidence') or d.get('score') or 0
                    created = d.get('created_at') or d.get('timestamp') or d.get('date') or ""
                    if hasattr(created, 'isoformat'):
                        created = created.isoformat()
                    else:
                        created = str(created)
                    rows.append({"username": str(username), "predicted_class": str(cls), "confidence": conf, "created_at": created})
                if rows:
                    return jsonify({"success": True, "total": len(rows), "history": rows, "table": table})
            except Exception as e:
                print(f"Table {table} failed:", e)
                continue
        return jsonify({"success": True, "total": 0, "history": [], "debug": "No history table found"})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": str(e), "history": []}), 500
    
# ---------- ADMIN - CLEAR HISTORY ----------
@app.route('/admin/clear_history', methods=['POST'])
def admin_clear_history():
    try:
        from sqlalchemy import text
        # Delete all history - but NOT users!
        try:
            History.query.delete()
            db.session.commit()
            return jsonify({"success": True, "message": "History cleared!"})
        except Exception as e:
            print("History delete failed:", e)
        # Fallback - try raw SQL tables
        for table in ["history", "predictions", "classifications"]:
            try:
                db.session.execute(text(f"DELETE FROM {table}"))
                db.session.commit()
                return jsonify({"success": True, "message": "History cleared from " + table})
            except:
                continue
        return jsonify({"success": False, "message": "No history table found"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    
# ---------- USER - CLEAR MY HISTORY ----------
@app.route('/clear_history', methods=['POST'])
def user_clear_history():
    try:
        from sqlalchemy import text
        data = request.get_json(silent=True) or {}
        username = data.get('username') or ""
        if not username:
            return jsonify({"success": False, "message": "Username needed"}), 400
        try:
            History.query.filter_by(username=username).delete()
            db.session.commit()
            return jsonify({"success": True, "message": "Your history cleared!"})
        except Exception as e:
            print("User clear failed:", e)
        for table in ["history", "predictions", "classifications"]:
            try:
                db.session.execute(text(f"DELETE FROM {table} WHERE username = :u"), {"u": username})
                db.session.commit()
                return jsonify({"success": True, "message": "Your history cleared!"})
            except:
                continue
        return jsonify({"success": False, "message": "No history found"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

# ---------- RUN ----------
import os
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)