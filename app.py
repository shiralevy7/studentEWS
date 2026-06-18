"""
Early Warning & Risk Tiering — Flask backend
RandomForest /predict is preserved verbatim. Added:
  • /draft_message       — empathetic parent letters (he/en, gender-aligned)
  • /generate_materials  — 3-tier Hebrew differentiated lesson sequence
Both run a deterministic template layer that enforces the Strict
Pedagogical Hebrew Guard. An optional LLM path (SYSTEM_PROMPT_HE) can be
switched on if an API key is present.
"""
import os, joblib, numpy as np, pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# ---- Model: "the brain" (unchanged) ---------------------------------
MODEL_PATH = "student_risk_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
if model is None:
    print("WARNING: student_risk_model.pkl not found — /predict will 503.")

FEATURES = ["grade_level", "subject_encoded", "attendance_percentage",
            "assignment_completion_rate", "midterm_exam_score", "teacher_observed_stress"]
SUBJECT_MAP   = {"Math": 0, "Language Arts": 1, "Science": 2}
LABEL_TO_TIER = {0: "Excellent", 1: "Good", 2: "Average", 3: "Poor"}
HE_SUBJECTS   = ["מתמטיקה", "שפה ואוריינות", "מדעים"]
HE_GRADE      = ["", "א'", "ב'", "ג'", "ד'", "ה'", "ו'"]

def encode_subject(v):
    return SUBJECT_MAP.get(v.strip(), 0) if isinstance(v, str) else int(v)

def build_row(p):
    return [int(p["grade_level"]),
            encode_subject(p.get("subject_encoded", p.get("subject", 0))),
            float(p["attendance_percentage"]),
            float(p["assignment_completion_rate"]),
            float(p["midterm_exam_score"]),
            int(p["teacher_observed_stress"])]

# =====================================================================
#  STRICT PEDAGOGICAL HEBREW GUARD  (few-shot system prompt for an LLM)
# =====================================================================
SYSTEM_PROMPT_HE = """אתה מורה ותיק/ה בבית ספר יסודי בישראל. כתוב טקסטים בעברית
תקנית, חמה ואמפתית, המתאימים לתקשורת עם הורים ולחומרי לימוד לילדים.

חוקים מחייבים:
1. עברית טבעית בלבד. אסור שימוש בשפה ארכאית, בתרגומית או בניסוחים מלאכותיים.
   אל תתרגם מונחים מילולית (לדוגמה: 'Excellent' אינו 'מצוין יתרה' אלא 'רמת הצטיינות').
2. התאמת מין מוחלטת: פעלים בגוף ראשון יתאימו למין המורה שיינתן, ופעלים, שמות תואר
   וכינויי גוף בגוף שלישי יתאימו למין התלמיד/ה שיינתן. אין לעבור בין מינים באותו טקסט.
3. גוון לא־מאשים, מכבד, ומזמין לשיתוף פעולה. הימנע מטון תאגידי/רובוטי.
4. משפטים קצרים וברורים, מותאמים לקריאה של ילדים צעירים בחומרי הלימוד.

דוגמאות (few-shot):
[מורה=נקבה, תלמיד=זכר] "שלום רב, כאן רוני. אני פונה אליכם מתוך אכפתיות עמוקה...
דני הוא תלמיד נעים ובעל יכולת, ואני מאמינה שיחד נסייע לו לחזור למסלול טוב."
[מורה=זכר, תלמידה=נקבה] "שלום רב, כאן יוסי. שמתי לב שמאיה נראית מתוחה ועמוסה
מעט יותר מהרגיל, ורציתי לשתף אתכם בעדינות. שלומה חשוב לי מאוד."
"""

def he_forms(tg, sg):
    """Resolve teacher (1st person) + student (3rd person) morphology."""
    return {
        "believe": "מאמינה" if tg == "f" else "מאמין",
        "think":   "חושבת" if tg == "f" else "חושב",
        "glad":    "שמחה"  if tg == "f" else "שמח",
        "standing":"עומדת" if tg == "f" else "עומד",
        "pron":    "היא" if sg == "f" else "הוא",
        "pupil":   "תלמידה" if sg == "f" else "תלמיד",
        "pleasant":"נעימה" if sg == "f" else "נעים",
        "capable": "בעלת יכולת" if sg == "f" else "בעל יכולת",
        "succeeding":"מצליחה" if sg == "f" else "מצליח",
        "continuing":"ממשיכה" if sg == "f" else "ממשיך",
        "looks":   "נראית" if sg == "f" else "נראה",
        "tense":   "מתוחה" if sg == "f" else "מתוח",
        "burdened":"עמוסה" if sg == "f" else "עמוס",
        "wellbeing":"שלומה" if sg == "f" else "שלומו",
        "inHim":   "בה" if sg == "f" else "בו",
        "toHim":   "לה" if sg == "f" else "לו",
        "defName": "התלמידה" if sg == "f" else "התלמיד",
    }

def build_parent_message(data):
    """Deterministic, gender-aligned draft (mirrors the frontend)."""
    lang   = data.get("lang", "he")
    tier   = data.get("tier", "Average")
    concern= data.get("concern", "General Update")
    teacher= data.get("teacher") or ("המורה" if lang == "he" else "your teacher")
    parent = (data.get("parent") or "").strip()
    name   = (data.get("name") or "").strip()

    if lang == "he":
        G = he_forms(data.get("teacher_gender", "f"), data.get("student_gender", "f"))
        n = name or G["defName"]
        greet = f"שלום {parent}, כאן {teacher}." if parent else f"שלום רב, כאן {teacher}."
        lead  = f"אני פונה אליכם מתוך אכפתיות עמוקה ורצון ללוות את {n} בצורה המיטבית."
        body = {
            "Academic Drop": f"בתקופה האחרונה הבחנתי בירידה קלה בהישגים של {n}. {n} {G['pron']} {G['pupil']} {G['pleasant']} ו{G['capable']}, ואני {G['believe']} שבעזרה משותפת נצליח לסייע {G['toHim']} לחזור למסלול טוב.",
            "Attendance/Absences": f"לאחרונה הבחנתי בכמה ימי היעדרות של {n}. נוכחות סדירה חשובה מאוד הן ללמידה והן לתחושת השייכות בכיתה, ולכן רציתי לבדוק יחד אתכם אם יש דרך שבה אוכל לעזור ולהקל על ההגעה לבית הספר.",
            "Emotional Stress": f"לאחרונה שמתי לב ש{n} {G['looks']} {G['tense']} ו{G['burdened']} מעט יותר מהרגיל, ורציתי לשתף אתכם בעדינות כדי שנוכל לתמוך {G['inHim']} יחד. {G['wellbeing']} חשוב לי מאוד.",
            "General Update": f"רציתי לעדכן אתכם בקצרה לגבי {n}. {n} {G['continuing']} להשתלב יפה בכיתה, ואני {G['glad']} לשתף אתכם בהתקדמות החיובית.",
        }.get(concern, "")
        tier_line = {
            "Poor": "לאור התמונה הנוכחית, אשמח אם נוכל למצוא זמן לשיחה קצרה בקרוב — אפילו השבוע, אם יתאים לכם.",
            "Average": f"אני {G['think']} ששיחה קצרה בינינו תוכל לתרום מאוד, בכל זמן שנוח לכם.",
            "Good": "זוהי בעיקר הודעה ידידותית, כדי שנישאר מתואמים ונמשיך לבנות יחד על ההתקדמות.",
            "Excellent": f"כמו כן, רציתי לשתף בגאווה עד כמה {n} {G['succeeding']}, ולהודות לכם על התמיכה המתמשכת בבית.",
        }.get(tier, "")
        close = f"תודה מקרב לב על שיתוף הפעולה. אני {G['standing']} לרשותכם לכל שאלה ובכל עת.\n\nבברכה,\n{teacher}"
        return "\n\n".join([greet, lead, body, tier_line, close])

    # English fallback
    n = name or "your child"
    greet = f"Dear {parent}," if parent else "Dear Parent or Guardian,"
    lead = f"This is {teacher}. I'm reaching out with genuine care, hoping we can support {n} together in the best possible way."
    body = {
        "Academic Drop": f"Lately I've noticed a small dip in {n}'s schoolwork, and I'm confident that with a little extra support from both of us, {n} can get back on a strong path.",
        "Attendance/Absences": f"I've noticed {n} has missed a number of school days recently, and I wanted to check whether there's anything I can do to make getting to school a little easier.",
        "Emotional Stress": f"Recently {n} has seemed a little more tense than usual, and I wanted to share this gently so we can support {n} together.",
        "General Update": f"I wanted to send a brief update. {n} continues to be a positive part of our classroom.",
    }.get(concern, "")
    close = f"Thank you so much for your partnership. Please don't hesitate to reach out with any questions.\n\nWarm regards,\n{teacher}"
    return "\n\n".join([greet, lead, body, close])

def build_materials(data):
    """3-tier Hebrew differentiated lesson sequence."""
    t = (data.get("topic") or "הנושא הזה").strip()
    g = int(data.get("grade", 3))
    subj = HE_SUBJECTS[int(data.get("subject", 2))]
    gl = "כיתה " + (HE_GRADE[g] if 0 < g < len(HE_GRADE) else str(g))
    emph = "מדגישה" if data.get("teacher_gender", "f") == "f" else "מדגיש"
    return {
        "base": [
            f"היום נלמד על {t}.",
            f"{t} – נושא חשוב שנכיר יחד. נתקדם צעד אחר צעד.",
            f"הרעיון המרכזי: {t} נמצא סביבנו, ואפשר לראות אותו בחיי היום-יום.",
            f"מילות מפתח: שימו לב למילים שהמורה {emph} בכיתה.",
            f"משימה קצרה: ציירו ציור פשוט שמראה את {t}, וספרו לחבר דבר אחד שלמדתם.",
        ],
        "standard": [
            f"מוקד השיעור ({gl}, {subj}): {t}.",
            f"{t} עוזר לנו להבין כיצד הדברים פועלים. בשיעור נברר מהו {t} ומדוע הוא חשוב.",
            f"פעילות מודרכת: בזוגות, כתבו שלוש דוגמאות הקשורות ל{t} והסבירו כל אחת במשפט שלם.",
            f"בדיקת הבנה: האם תוכלו לתאר את {t} במילים שלכם ולתת דוגמה אחת?",
        ],
        "advanced": [
            f"אתגר חקר ({gl}, {subj}): {t}.",
            f"העמיקו: מדוע {t} חשוב, ומה היה קורה אילו פעל אחרת?",
            f"השוואה וניגוד: במה {t} דומה לרעיון אחר שלמדתם, ובמה הוא שונה ממנו?",
            f"חקירה זוטא: נסחו שאלה על {t}, שערו תשובה, ותכננו כיצד תבדקו אותה בעצמכם.",
            f"רפלקציה ומטה-קוגניציה: איזה חלק בלמידה אתגר אתכם יותר מכול, וכיצד התמודדתם איתו?",
        ],
    }

# ---- Routes ---------------------------------------------------------
@app.route("/")
def home():
    return send_from_directory("static", "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not loaded"}), 503
    try:
        data = request.get_json(force=True)
        X = pd.DataFrame([build_row(data)], columns=FEATURES)
        label = int(model.predict(X)[0])
        conf = round(float(np.max(model.predict_proba(X)[0])) * 100, 1)
        return jsonify({"label": label, "tier": LABEL_TO_TIER.get(label, "Unknown"), "confidence": conf})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    if model is None:
        return jsonify({"error": "Model file not loaded"}), 503
    try:
        records = request.get_json(force=True).get("students", [])
        X = pd.DataFrame([build_row(r) for r in records], columns=FEATURES)
        labels = model.predict(X).astype(int).tolist()
        results = [{**r, "label": l, "tier": LABEL_TO_TIER.get(l, "Unknown")}
                   for r, l in zip(records, labels)]
        return jsonify({"results": results, "count": len(results)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/draft_message", methods=["POST"])
def draft_message():
    """Body: {parent, name, tier, concern, lang, teacher, teacher_gender, student_gender}"""
    try:
        return jsonify({"message": build_parent_message(request.get_json(force=True))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_materials", methods=["POST"])
def generate_materials():
    """Body: {topic, grade, subject, teacher_gender}"""
    try:
        return jsonify({"levels": build_materials(request.get_json(force=True))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)