from flask import Flask, render_template, request
import os
import re
import pickle
import numpy as np
import pdfplumber
import pytesseract

from PIL import Image, ImageEnhance, ImageFilter
from werkzeug.utils import secure_filename


# ============================================================
# FLASK APP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024


ALLOWED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png",
    "bmp",
    "tiff",
    "webp"
}


# ============================================================
# TESSERACT
# ============================================================

# Windows local system
tesseract_path = os.environ.get(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

if os.path.exists(tesseract_path):
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model.pkl"
)

try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    print("Model loaded successfully!")

except Exception as e:
    model = None
    print("Model loading error:", e)


# ============================================================
# CHECK ALLOWED FILE
# ============================================================

def allowed_file(filename):

    if not filename or "." not in filename:
        return False

    extension = filename.rsplit(
        ".",
        1
    )[1].lower()

    return extension in ALLOWED_EXTENSIONS


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):

    try:

        image = image.convert("L")

        enhancer = ImageEnhance.Contrast(
            image
        )

        image = enhancer.enhance(2.5)

        image = image.filter(
            ImageFilter.SHARPEN
        )

        width, height = image.size

        if width < 1800:

            scale = 1800 / width

            image = image.resize(
                (
                    int(width * scale),
                    int(height * scale)
                )
            )

        return image

    except Exception as e:

        print(
            "Image preprocessing error:",
            e
        )

        return image


# ============================================================
# OCR IMAGE
# ============================================================

def ocr_image(image):

    try:

        image = preprocess_image(
            image
        )

        text = pytesseract.image_to_string(
            image,
            config="--psm 6"
        )

        return text

    except Exception as e:

        print(
            "OCR error:",
            e
        )

        return ""


# ============================================================
# EXTRACT TEXT FROM IMAGE
# ============================================================

def extract_text_from_image(
    file_path
):

    try:

        image = Image.open(
            file_path
        )

        return ocr_image(
            image
        )

    except Exception as e:

        print(
            "Image extraction error:",
            e
        )

        return ""


# ============================================================
# EXTRACT TEXT FROM PDF
# ============================================================

def extract_text_from_pdf(
    file_path
):

    text = ""

    try:

        with pdfplumber.open(
            file_path
        ) as pdf:

            for page in pdf.pages:

                page_text = (
                    page.extract_text()
                )

                if page_text:

                    text += (
                        page_text + "\n"
                    )

    except Exception as e:

        print(
            "PDF extraction error:",
            e
        )

    return text


# ============================================================
# MAIN REPORT TEXT EXTRACTION
# ============================================================

def extract_report_text(
    file_path
):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == ".pdf":

        text = extract_text_from_pdf(
            file_path
        )

        return text

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    if extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tiff",
        ".webp"
    }:

        return extract_text_from_image(
            file_path
        )

    return ""


# ============================================================
# CLEAN TEXT
# ============================================================

def clean_text(text):

    text = text.replace(
        "\r",
        "\n"
    )

    text = re.sub(
        r"[\t ]+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PATIENT NAME
# ============================================================

def extract_patient_name(text):

    patterns = [

        r"Patient\s+Name\s*[:\-]\s*"
        r"([A-Za-z][A-Za-z .'-]{1,60})",

        r"Patient\s+Name\s+"
        r"([A-Za-z][A-Za-z .'-]{1,60})",

        r"Name\s*[:\-]\s*"
        r"([A-Za-z][A-Za-z .'-]{1,60})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            name = match.group(
                1
            ).strip()

            name = re.split(
                r"\s{2,}|Age|Gender|Phone|Mobile|Patient ID",
                name,
                flags=re.IGNORECASE
            )[0].strip()

            return name

    return ""


# ============================================================
# AGE
# ============================================================

def extract_age(text):

    patterns = [

        r"Age\s*/\s*Gender\s*[:\-]?\s*(\d{1,3})",

        r"Age\s*[:\-]\s*(\d{1,3})",

        r"Age\s+(\d{1,3})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return match.group(1)

    return ""


# ============================================================
# GENDER
# ============================================================

def extract_gender(text):

    patterns = [

        r"Gender\s*[:\-]\s*"
        r"(Male|Female|Other)",

        r"Sex\s*[:\-]\s*"
        r"(Male|Female|Other)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            ).capitalize()

    return ""


# ============================================================
# PHONE
# ============================================================

def extract_phone(text):

    patterns = [

        r"Patient\s+Phone\s*[:\-]\s*"
        r"(\+?\d[\d\s\-()]{8,18})",

        r"Patient\s+Mobile\s*[:\-]\s*"
        r"(\+?\d[\d\s\-()]{8,18})",

        r"Phone\s+Number\s*[:\-]\s*"
        r"(\+?\d[\d\s\-()]{8,18})",

        r"Mobile\s+Number\s*[:\-]\s*"
        r"(\+?\d[\d\s\-()]{8,18})",

        r"Mobile\s*[:\-]\s*"
        r"(\+?\d[\d\s\-()]{8,18})"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            return match.group(
                1
            ).strip()

    return ""


# ============================================================
# NUMBER CLEANING
# ============================================================

def clean_number(value):

    if value is None or value == "":
        return None

    value = str(value)

    value = value.replace(
        ",",
        ""
    )

    value = value.replace(
        "O",
        "0"
    )

    value = value.replace(
        "o",
        "0"
    )

    try:

        return float(value)

    except (
        ValueError,
        TypeError
    ):

        return None


# ============================================================
# RADIUS
# ============================================================

def extract_radius(text):

    patterns = [

        r"(?:Mean\s+)?Radius\s*[:=\-]?"
        r"\s*([0-9Oo]+(?:\.[0-9Oo]+)?)",

        r"radius_mean\s*[:=\-]?"
        r"\s*([0-9Oo]+(?:\.[0-9Oo]+)?)",

        r"Radius\s+"
        r"([0-9Oo]+(?:\.[0-9Oo]+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = clean_number(
                match.group(1)
            )

            if value is not None:
                return value

    return None


# ============================================================
# PERIMETER
# ============================================================

def extract_perimeter(text):

    patterns = [

        r"(?:Mean\s+)?Perimeter\s*[:=\-]?"
        r"\s*([0-9Oo]+(?:\.[0-9Oo]+)?)",

        r"perimeter_mean\s*[:=\-]?"
        r"\s*([0-9Oo]+(?:\.[0-9Oo]+)?)",

        r"Perimeter\s+"
        r"([0-9Oo]+(?:\.[0-9Oo]+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = clean_number(
                match.group(1)
            )

            if value is not None:
                return value

    return None


# ============================================================
# AREA
# ============================================================

def extract_area(text):

    patterns = [

        r"(?:Mean\s+)?Area\s*[:=\-]?"
        r"\s*([0-9Oo]+(?:\.[0-9Oo]+)?)",

        r"area_mean\s*[:=\-]?"
        r"\s*([0-9Oo]+(?:\.[0-9Oo]+)?)",

        r"Area\s+"
        r"([0-9Oo]+(?:\.[0-9Oo]+)?)"
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            value = clean_number(
                match.group(1)
            )

            if value is not None:
                return value

    return None


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# UPLOAD PAGE
# ============================================================

@app.route("/upload")
def upload():

    return render_template(
        "upload.html"
    )


# ============================================================
# ANALYZE
# ============================================================

@app.route(
    "/analyze",
    methods=["POST"]
)
def analyze():

    uploaded_file = request.files.get(
        "report"
    )

    if not uploaded_file:

        return render_template(
            "analyze.html",
            error="Please upload a medical report."
        )

    if uploaded_file.filename == "":

        return render_template(
            "analyze.html",
            error="Please select a medical report."
        )

    if not allowed_file(
        uploaded_file.filename
    ):

        return render_template(
            "analyze.html",
            error=(
                "Invalid file type. "
                "Please upload a PDF or image file."
            )
        )

    filename = secure_filename(
        uploaded_file.filename
    )

    if not filename:

        return render_template(
            "analyze.html",
            error="Invalid file name."
        )

    file_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        filename
    )

    uploaded_file.save(
        file_path
    )

    print(
        "Report uploaded:",
        filename
    )

    # --------------------------------------------------------
    # EXTRACT TEXT
    # --------------------------------------------------------

    report_text = extract_report_text(
        file_path
    )

    report_text = clean_text(
        report_text
    )

    print(
        "Extracted report text:"
    )

    print(
        report_text
    )

    # --------------------------------------------------------
    # PATIENT DETAILS
    # --------------------------------------------------------

    patient_name = (
        extract_patient_name(
            report_text
        )
    )

    age = extract_age(
        report_text
    )

    gender = extract_gender(
        report_text
    )

    phone = extract_phone(
        report_text
    )

    # --------------------------------------------------------
    # ML FEATURES
    # --------------------------------------------------------

    radius = extract_radius(
        report_text
    )

    perimeter = extract_perimeter(
        report_text
    )

    area = extract_area(
        report_text
    )

    print(
        "Name:",
        patient_name
    )

    print(
        "Age:",
        age
    )

    print(
        "Gender:",
        gender
    )

    print(
        "Phone:",
        phone
    )

    print(
        "Radius:",
        radius
    )

    print(
        "Perimeter:",
        perimeter
    )

    print(
        "Area:",
        area
    )

    # --------------------------------------------------------
    # SEND TO ANALYZE PAGE
    # --------------------------------------------------------

    return render_template(

        "analyze.html",

        patient_name=patient_name,

        age=age,

        gender=gender,

        phone=phone,

        radius=radius,

        perimeter=perimeter,

        area=area,

        report_text=report_text,

        error=None
    )


# ============================================================
# PREDICT
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    # --------------------------------------------------------
    # GET VALUES
    # --------------------------------------------------------

    patient_name = request.form.get(
        "patient_name",
        ""
    )

    age = request.form.get(
        "age",
        ""
    )

    gender = request.form.get(
        "gender",
        ""
    )

    phone = request.form.get(
        "phone",
        ""
    )

    radius_text = request.form.get(
        "radius",
        ""
    )

    perimeter_text = request.form.get(
        "perimeter",
        ""
    )

    area_text = request.form.get(
        "area",
        ""
    )

    report_text = request.form.get(
        "report_text",
        ""
    )

    # --------------------------------------------------------
    # CONVERT VALUES
    # --------------------------------------------------------

    radius = clean_number(
        radius_text
    )

    perimeter = clean_number(
        perimeter_text
    )

    area = clean_number(
        area_text
    )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    missing = []

    if radius is None:
        missing.append(
            "Radius"
        )

    if perimeter is None:
        missing.append(
            "Perimeter"
        )

    if area is None:
        missing.append(
            "Area"
        )

    if missing:

        return render_template(

            "analyze.html",

            error=(
                "Please enter valid values for: "
                + ", ".join(missing)
            ),

            patient_name=patient_name,

            age=age,

            gender=gender,

            phone=phone,

            radius=radius,

            perimeter=perimeter,

            area=area,

            report_text=report_text
        )

    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = None

    probability = None

    if model is None:

        prediction = (
            "Model Not Loaded"
        )

    else:

        try:

            # ------------------------------------------------
            # CREATE INPUT FEATURES
            # ------------------------------------------------

            features = np.array(
                [[
                    radius,
                    perimeter,
                    area
                ]],
                dtype=float
            )

            # ------------------------------------------------
            # MODEL PREDICTION
            # ------------------------------------------------

            result = model.predict(
                features
            )[0]

            # 0 = Malignant
            # 1 = Benign

            if int(result) == 0:

                prediction = (
                    "Malignant"
                )

            else:

                prediction = (
                    "Benign"
                )

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        features
                    )[0]
                )

                probability = round(
                    max(probabilities) * 100,
                    2
                )

        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            prediction = (
                "Prediction Error"
            )

    # ========================================================
    # RESULT PAGE
    # ========================================================

    return render_template(

        "predict.html",

        prediction=prediction,

        probability=probability,

        radius=radius,

        perimeter=perimeter,

        area=area
    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        debug=False,

        host="0.0.0.0",

        port=port
    )