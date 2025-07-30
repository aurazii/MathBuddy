
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import firebase_admin
from firebase_admin import credentials, firestore, auth
from datetime import datetime
import re
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import pytesseract

app = Flask(__name__)
app.secret_key = 'secretkey'

# Initialize Firebase
cred = credentials.Certificate('firebase_config.json')
firebase_admin.initialize_app(cred)
db = firestore.client()

# Test Firebase connection
try:
    # Test write
    test_doc = db.collection('test').document('connection')
    test_doc.set({'status': 'connected', 'timestamp': datetime.now()})
    print("✅ Firebase connected successfully!")
except Exception as e:
    print(f"❌ Firebase connection failed: {e}")

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_math_image(image_path):
    """Preprocessing yang lebih fokus untuk soal matematika dengan eksponen"""
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Resize untuk OCR yang lebih baik - lebih besar untuk detail
    height, width = gray.shape
    if height < 600 or width < 600:
        scale_factor = max(800/height, 800/width)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        gray = cv2.resize(gray, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Noise reduction yang lebih halus
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Enhance contrast untuk teks matematika
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # Sharpening untuk ketajaman
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    # Multiple preprocessing methods
    methods = []
    
    # Method 1: OTSU dengan morphology yang lebih halus
    _, thresh1 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    thresh1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    methods.append(thresh1)
    
    # Method 2: Adaptive threshold dengan parameter yang lebih baik
    thresh2 = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10)
    methods.append(thresh2)
    
    # Method 3: Mean adaptive threshold
    thresh3 = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
    methods.append(thresh3)
    
    # Method 4: Manual threshold dengan nilai yang berbeda
    _, thresh4 = cv2.threshold(sharpened, 140, 255, cv2.THRESH_BINARY)
    methods.append(thresh4)
    
    # Method 5: Inverted untuk background gelap
    _, thresh5 = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    methods.append(thresh5)
    
    return methods

def extract_math_text_multiple(image_path):
    """Extract teks dengan fokus khusus pada eksponen dan operator matematika"""
    try:
        processed_images = preprocess_math_image(image_path)
        
        # Konfigurasi Tesseract yang lebih spesifik untuk matematika
        configs = [
            # Config khusus untuk matematika dengan whitelist yang lebih lengkap
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-*/=^().,x÷×√∫∑∏πesincostan',
            r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789+-*/=^().,x÷×√∫∑∏πesincostan',
            r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789+-*/=^().,x÷×√∫∑∏πesincostan',
            # Config untuk single line math
            r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789+-*/=^().,x÷×√',
            # Config untuk single word
            r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789+-*/=^().,',
            # Config dengan preserve_interword_spaces
            r'--oem 3 --psm 7 -c preserve_interword_spaces=1 tessedit_char_whitelist=0123456789+-*/=^().,x',
            # Config tanpa whitelist untuk backup
            r'--oem 3 --psm 8',
            r'--oem 3 --psm 7',
            r'--oem 3 --psm 6',
            # Config dengan different OEM
            r'--oem 1 --psm 8 -c tessedit_char_whitelist=0123456789+-*/=^().,',
            r'--oem 2 --psm 8 -c tessedit_char_whitelist=0123456789+-*/=^().,',
        ]
        
        results = []
        confidence_scores = []
        
        for img_idx, img in enumerate(processed_images):
            for config_idx, config in enumerate(configs):
                try:
                    # Get text with confidence
                    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
                    
                    # Filter high confidence words
                    words = []
                    total_conf = 0
                    valid_words = 0
                    
                    for i in range(len(data['text'])):
                        if int(data['conf'][i]) > 30:  # Confidence threshold
                            word = data['text'][i].strip()
                            if word and any(c in word for c in '0123456789+=^*/-×÷'):
                                words.append(word)
                                total_conf += int(data['conf'][i])
                                valid_words += 1
                    
                    if words:
                        text = ' '.join(words).replace(' ', '')
                        avg_conf = total_conf / valid_words if valid_words > 0 else 0
                        
                        # Prioritas hasil yang mengandung karakter matematika
                        if any(c in text for c in '0123456789+=^*/-×÷'):
                            results.append(text)
                            confidence_scores.append(avg_conf)
                            print(f"Method {img_idx}-{config_idx}: '{text}' (conf: {avg_conf:.1f})")
                    
                    # Fallback: regular text extraction
                    text = pytesseract.image_to_string(img, config=config)
                    if text.strip():
                        cleaned = text.strip().replace('\n', '').replace(' ', '')
                        if cleaned and any(c in cleaned for c in '0123456789+=^*/-×÷'):
                            results.append(cleaned)
                            confidence_scores.append(50)  # Default confidence
                            
                except Exception as e:
                    print(f"OCR config error: {e}")
                    continue
        
        if results:
            # Pilih hasil terbaik berdasarkan confidence dan konten
            best_results = []
            
            # Prioritas 1: Hasil dengan eksponen
            exponent_results = [(r, c) for r, c in zip(results, confidence_scores) if '^' in r]
            if exponent_results:
                best_results.extend(exponent_results)
            
            # Prioritas 2: Hasil dengan operator matematika lengkap
            math_results = [(r, c) for r, c in zip(results, confidence_scores) 
                           if re.search(r'\d.*[+\-*/^=×÷].*\d', r)]
            if math_results:
                best_results.extend(math_results)
            
            # Prioritas 3: Hasil dengan angka dan operator
            number_results = [(r, c) for r, c in zip(results, confidence_scores) 
                             if re.search(r'\d', r) and any(op in r for op in '+-*/=^×÷')]
            if number_results:
                best_results.extend(number_results)
            
            if best_results:
                # Pilih yang confidence tertinggi
                best_result = max(best_results, key=lambda x: x[1])
                return best_result[0]
            
            # Fallback: pilih yang terpanjang
            return max(results, key=len)
        
        return None
        
    except Exception as e:
        print(f"Error in OCR: {e}")
        return None

# Tambahkan fungsi ini di atas clean_math_expression
def convert_unicode_superscripts(text):
    superscript_map = str.maketrans({
        '¹': '^1',
        '²': '^2',
        '³': '^3',
        '⁴': '^4',
        '⁵': '^5',
        '⁶': '^6',
        '⁷': '^7',
        '⁸': '^8',
        '⁹': '^9',
        '⁰': '^0'
    })
    return text.translate(superscript_map)

def clean_math_expression(text):
    """Pembersihan dan deteksi eksponen yang lebih pintar"""
    if not text:
        return None
    
    print(f"Raw OCR text: '{text}'")
    
    # Normalisasi karakter matematika
    text = text.replace('×', '*').replace('÷', '/').replace('x', '*')
    
    # Hapus karakter yang tidak diinginkan tapi pertahankan struktur
    text = re.sub(r'[^0-9+\-*/=^().\s]', '', text)
    text = re.sub(r'\s+', '', text)
    
    print(f"After cleaning: '{text}'")
    
    # Perbaikan pola umum OCR error
    corrections = [
        (r'(\d)(\d)=', r'\1^\2='),  # 23= -> 2^3=
        (r'^(\d)(\d)$', r'\1^\2'),  # 23 -> 2^3
        (r'(\d)(\d)\+', r'\1^\2+'), # 23+ -> 2^3+
        (r'(\d)(\d)\-', r'\1^\2-'), # 23- -> 2^3-
        (r'(\d)(\d)\*', r'\1^\2*'), # 23* -> 2^3*
        (r'(\d)(\d)/', r'\1^\2/'),  # 23/ -> 2^3/
        (r'(\d)o(\d)', r'\1+\2'),   # 2o3 -> 2+3 (OCR error)
        (r'(\d)O(\d)', r'\1+\2'),   # 2O3 -> 2+3 (OCR error)
        (r'(\d)l(\d)', r'\1+\2'),   # 2l3 -> 2+3 (OCR error)
        (r'(\d)I(\d)', r'\1+\2'),   # 2I3 -> 2+3 (OCR error)
        (r'(\d)S(\d)', r'\1+\2'),   # 2S3 -> 2+3 (OCR error)
        (r'(\d)s(\d)', r'\1+\2'),   # 2s3 -> 2+3 (OCR error)
    ]
    
    for pattern, replacement in corrections:
        if re.search(pattern, text):
            old_text = text
            text = re.sub(pattern, replacement, text)
            print(f"Correction applied: '{old_text}' -> '{text}'")
            break
    
    # Validasi hasil
    if re.match(r'^[\d+\-*/=^().]+$', text):
        return text
    
    return text if text else None

def validate_and_complete_expression(text):
    """Validasi dan auto-complete yang lebih pintar untuk eksponen"""
    if not text:
        return None
    
    print(f"Validating: '{text}'")
    
    # Deteksi dan lengkapi eksponen
    if re.match(r'^\d\^\d$', text):  # 2^3
        try:
            base, exp = text.split('^')
            result = int(base) ** int(exp)
            completed = f"{text}={result}"
            print(f"Completed exponent: {completed}")
            return completed
        except:
            return text
    
    # Deteksi eksponen dengan equals: 2^3=
    if re.match(r'^\d\^\d=$', text):  # 2^3=
        try:
            expr = text[:-1]  # Hapus =
            base, exp = expr.split('^')
            result = int(base) ** int(exp)
            completed = f"{expr}={result}"
            print(f"Completed exponent with equals: {completed}")
            return completed
        except:
            return text
    
    # Pola aritmatika lainnya
    patterns = {
        r'^(\d+)\+(\d+)$': lambda m: f"{m.group(0)}={int(m.group(1)) + int(m.group(2))}",
        r'^(\d+)\-(\d+)$': lambda m: f"{m.group(0)}={int(m.group(1)) - int(m.group(2))}",
        r'^(\d+)\*(\d+)$': lambda m: f"{m.group(0)}={int(m.group(1)) * int(m.group(2))}",
        r'^(\d+)/(\d+)$': lambda m: f"{m.group(0)}={int(m.group(1)) // int(m.group(2))}" if int(m.group(2)) != 0 else m.group(0),
    }
    
    for pattern, func in patterns.items():
        match = re.match(pattern, text)
        if match:
            try:
                result = func(match)
                print(f"Completed arithmetic: {result}")
                return result
            except:
                return text
    
    # Jika berakhir dengan =, hapus
    if text.endswith('=') and not re.search(r'=\d+$', text):
        return text[:-1]
    
    return text

def detect_math_patterns(text):
    """Deteksi pola-pola matematika dengan fokus pada eksponen"""
    patterns = {
        'exponent': r'.*\^.*',  # Tambahkan deteksi eksponen
        'equation': r'.*=.*',
        'quadratic': r'.*x\^?2.*',
        'cubic': r'.*x\^?3.*',
        'power': r'.*\d+\^\d+.*',
        'fraction': r'\d+/\d+',
        'trigonometry': r'.*(sin|cos|tan).*',
        'logarithm': r'.*(log|ln).*',
        'calculus': r'.*(∫|∑|∂|lim).*',
        'inequality': r'.*(≤|≥|<|>).*',
        'geometry': r'.*(π|°|radius|diameter|area|volume).*',
        'algebra': r'.*[a-z].*[+\-*/=].*',
        'arithmetic': r'^\s*\d+\s*[+\-*/]\s*\d+.*'
    }
    
    detected = []
    for pattern_name, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(pattern_name)
    
    return detected

# Process image route
@app.route('/process-image', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file'})
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract teks matematika
            raw_text = extract_math_text_multiple(filepath)
            
            if raw_text:
                # Bersihkan ekspresi
                cleaned_text = clean_math_expression(raw_text)
                
                if cleaned_text:
                    # Validasi dan lengkapi ekspresi
                    final_text = validate_and_complete_expression(cleaned_text)
                    
                    # Deteksi pola matematika
                    patterns = detect_math_patterns(final_text)
                    
                    # Clean up file
                    os.remove(filepath)
                    
                    return jsonify({
                        'success': True, 
                        'text': final_text,
                        'raw_text': raw_text,
                        'patterns': patterns,
                        'message': f'Soal matematika berhasil diproses!'
                    })
                else:
                    os.remove(filepath)
                    return jsonify({
                        'success': False, 
                        'error': 'Tidak dapat mengenali soal matematika dari gambar. Pastikan gambar berisi angka dan operator matematika yang jelas.'
                    })
            else:
                os.remove(filepath)
                return jsonify({
                    'success': False, 
                    'error': 'Tidak dapat membaca teks dari gambar. Coba ambil foto yang lebih jelas dengan pencahayaan yang baik.'
                })
            
        except Exception as e:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'error': f'Error processing image: {str(e)}'})
    
    return jsonify({'success': False, 'error': 'Invalid file type. Gunakan PNG, JPG, JPEG, atau GIF.'})

@app.route('/')
def index():
    return render_template('calculator.html', user=session.get('user'))

# Temporary signup without Firebase (for testing)
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        phone = request.form['phone']
        password = request.form['password']
        confirm_password = request.form['confirmPassword']
        
        # Validasi
        if len(password) < 6:
            return render_template('signup.html', error='Password minimal 6 karakter')
        
        has_letter = any(c.isalpha() for c in password)
        has_number = any(c.isdigit() for c in password)
        
        if not (has_letter and has_number):
            return render_template('signup.html', error='Password harus kombinasi huruf dan angka')
        
        if password != confirm_password:
            return render_template('signup.html', error='Password tidak cocok')
        
        try:
            # Coba Firebase dulu
            # Format phone number untuk Indonesia
            formatted_phone = None
            if phone:
                if phone.startswith('+'):
                    formatted_phone = phone
                elif phone.startswith('0'):
                    formatted_phone = f'+62{phone[1:]}'
                else:
                    formatted_phone = f'+62{phone}'
            
            print(f"Creating user with email: {email}, phone: {formatted_phone}")
            
            user_record = auth.create_user(
                email=email,
                password=password,
                phone_number=formatted_phone if formatted_phone else None
            )
            
            print(f"User created successfully!")
            print(f"UID: {user_record.uid}")
            print(f"Email: {user_record.email}")
            print(f"Phone: {user_record.phone_number}")
            
            db.collection('users').document(user_record.uid).set({
                'email': email,
                'phone': phone,
                'formatted_phone': formatted_phone,
                'created_at': datetime.now()
            })
            
            session['signup_success'] = True
            return redirect(url_for('signin'))
            
        except Exception as e:
            print(f"Firebase error: {e}")
            print(f"Error type: {type(e)}")
            # Fallback: simpan ke session untuk testing
            session['temp_users'] = session.get('temp_users', {})
            session['temp_users'][email] = {
                'password': password, 
                'phone': phone,
                'formatted_phone': formatted_phone if 'formatted_phone' in locals() else None
            }
            session['signup_success'] = True
            return redirect(url_for('signin'))
    
    return render_template('signup.html')

# Temporary signin without Firebase (for testing)
@app.route('/signin', methods=['GET', 'POST'])
def signin():
    success_message = None
    if session.pop('signup_success', False):
        success_message = 'Akun berhasil dibuat! Silakan login.'
    
    if request.method == 'POST':
        email_or_phone = request.form['email']
        password = request.form['password']
        
        try:
            user = None
            
            # Cek apakah input adalah email atau nomor telepon
            if '@' in email_or_phone:
                # Input adalah email
                print(f"Searching by email: {email_or_phone}")
                user = auth.get_user_by_email(email_or_phone)
            else:
                # Input adalah nomor telepon, coba berbagai format
                phone_formats = []
                
                # Format 1: +62xxxxxxxxx
                if email_or_phone.startswith('0'):
                    phone_formats.append(f'+62{email_or_phone[1:]}')
                elif not email_or_phone.startswith('+'):
                    phone_formats.append(f'+62{email_or_phone}')
                else:
                    phone_formats.append(email_or_phone)
                
                # Format 2: Tambah format alternatif
                if email_or_phone.startswith('08'):
                    phone_formats.append(f'+628{email_or_phone[2:]}')
                
                print(f"Trying phone formats: {phone_formats}")
                
                # Coba setiap format
                for phone_format in phone_formats:
                    try:
                        print(f"Searching by phone: {phone_format}")
                        user = auth.get_user_by_phone_number(phone_format)
                        print(f"Found user with phone: {phone_format}")
                        break
                    except auth.UserNotFoundError:
                        print(f"Not found with format: {phone_format}")
                        continue
                    except Exception as e:
                        print(f"Error with format {phone_format}: {e}")
                        continue
            
            if user:
                session['user'] = user.uid
                session['email'] = user.email
                return redirect(url_for('index'))
            else:
                # Fallback: cek session untuk testing
                temp_users = session.get('temp_users', {})
                for stored_email, user_data in temp_users.items():
                    if (stored_email == email_or_phone or 
                        user_data.get('phone') == email_or_phone or
                        user_data.get('formatted_phone') == email_or_phone):
                        session['user'] = 'temp_user'
                        session['email'] = stored_email
                        return redirect(url_for('index'))
                
                return render_template('signin.html', error='Email atau nomor telepon tidak ditemukan', success=success_message)
            
        except auth.UserNotFoundError:
            return render_template('signin.html', error='Email atau nomor telepon tidak ditemukan', success=success_message)
        except Exception as e:
            print(f"Signin error: {e}")
            return render_template('signin.html', error='Email atau password salah', success=success_message)
    
    return render_template('signin.html', success=success_message)

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/debug-users')
def debug_users():
    try:
        # List semua users (hanya untuk debugging)
        users = auth.list_users()
        user_list = []
        for user in users.users:
            user_list.append({
                'uid': user.uid,
                'email': user.email,
                'phone': user.phone_number,
                'created': user.user_metadata.creation_timestamp if hasattr(user, 'user_metadata') else 'N/A'
            })
        return jsonify(user_list)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
