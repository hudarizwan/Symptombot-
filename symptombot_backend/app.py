from flask import Flask, request, jsonify   
from flask_cors import CORS                 
import traceback
import joblib  

# === Initialize Flask App ===
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# === Load ML Model and Data ===
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(script_dir, "disease_model.pkl"))
mlb = joblib.load(os.path.join(script_dir, "symptom_encoder.pkl"))
precautions_dict = joblib.load(os.path.join(script_dir, "precautions_dict.pkl"))

# === Hardcoded Medicine Suggestions ===
hardcoded_medicines = {
    "Malaria": ["Chloroquine", "Artemisinin", "Mefloquine"],
    "Common Cold": ["Paracetamol", "Cetirizine", "Steam inhalation"],
    "Allergy": ["Loratadine", "Cetirizine", "Antihistamine cream"],
    "Hepatitis A": ["Rest", "ORS", "Avoid alcohol"],
    "Hypoglycemia": ["Glucose tablets", "Sugar drink", "Fruit juice"],
    "Acne": ["Benzoyl peroxide", "Salicylic acid", "Retinoid cream"],
    "Fungal infection": ["Clotrimazole", "Ketoconazole", "Terbinafine"],
    "Drug Reaction": ["Stop medication", "Consult doctor", "Hydration therapy"],
    "Diabetes": ["Metformin", "Insulin", "Low-sugar diet"],
    "Hypertension": ["Amlodipine", "Lifestyle change", "Low sodium diet"],
    "Asthma": ["Inhaled corticosteroids", "Albuterol", "Avoid triggers"],
    "Tuberculosis": ["Isoniazid", "Rifampicin", "Ethambutol"],
    "Bronchitis": ["Rest", "Cough syrup", "Antibiotics if bacterial"],
    "Pneumonia": ["Antibiotics", "Rest", "Fever reducer"],
    "COVID-19": ["Paracetamol", "Antivirals", "Oxygen therapy"],
    "Migraine": ["Ibuprofen", "Sumatriptan", "Caffeine"],
    "Anxiety": ["Cognitive Behavioral Therapy", "SSRIs", "Relaxation techniques"],
    "Depression": ["SSRIs", "Therapy", "Lifestyle change"],
    "Thyroid Disorder": ["Levothyroxine", "Anti-thyroid drugs", "Iodine therapy"],
    "Cholera": ["ORS", "Doxycycline", "Hydration"],
    "Typhoid": ["Ciprofloxacin", "Azithromycin", "Rest"],
    "Food Poisoning": ["ORS", "Antiemetics", "Rest"],
    "UTI": ["Ciprofloxacin", "Nitrofurantoin", "Hydration"],
    "PCOS": ["Metformin", "Hormonal therapy", "Lifestyle change"],
    "Heart Attack": ["Aspirin", "Nitroglycerin", "Clot busters"],
    "Stroke": ["Clot busters", "Physical therapy", "Aspirin"],
    "Cancer (Generic)": ["Chemotherapy", "Radiation", "Surgery"],
    "Chickenpox": ["Calamine lotion", "Antihistamines", "Rest"],
    "Measles": ["Vitamin A", "Rest", "Fever reducer"],
    "Dengue": ["ORS", "Paracetamol", "Avoid NSAIDs"],
    "Zika Virus": ["Rest", "Fluids", "Paracetamol"],
    "Eczema": ["Moisturizer", "Hydrocortisone cream", "Avoid irritants"],
    "Psoriasis": ["Steroid cream", "Phototherapy", "Methotrexate"],
    "Arthritis": ["NSAIDs", "Physical therapy", "DMARDs"],
    "Gout": ["Allopurinol", "NSAIDs", "Low purine diet"],
    "Anemia": ["Iron supplements", "Vitamin B12", "Folic acid"],
    "Obesity": ["Diet plan", "Exercise", "Orlistat"],
    "Insomnia": ["Melatonin", "Sleep hygiene", "CBT-I"],
    "GERD": ["Antacids", "PPI", "Lifestyle changes"],
    "Peptic Ulcer": ["PPI", "Antibiotics", "Avoid spicy food"],
    "Appendicitis": ["Surgery", "Painkillers", "IV fluids"],
    "Gallstones": ["Surgery", "Pain management", "Low-fat diet"],
    "Kidney Stones": ["Painkillers", "Hydration", "Lithotripsy"],
    "Epilepsy": ["Valproate", "Lamotrigine", "Carbamazepine"],
    "Parkinson's Disease": ["Levodopa", "Dopamine agonists", "Physical therapy"],
    "Alzheimer's Disease": ["Donepezil", "Memantine", "Cognitive therapy"],
    "HIV/AIDS": ["Antiretroviral therapy", "Regular monitoring", "Preventive care"],
    "Lupus": ["Steroids", "Immunosuppressants", "NSAIDs"],
    "Rheumatoid Arthritis": ["Methotrexate", "Biologics", "NSAIDs"],
    "Sinusitis": ["Nasal spray", "Antibiotics", "Steam inhalation"],
    "Otitis Media": ["Amoxicillin", "Pain relief", "Warm compress"],
    "Hypothyroidism": ["Levothyroxine", "Iodine supplements", "Regular monitoring"],
    "Chronic cholestasis": ["Ursodeoxycholic acid", "Fat-soluble vitamins", "Low-fat diet"],
    "Osteoarthritis": ["NSAIDs", "Physical therapy", "Glucosamine supplements"],
    "(vertigo) Paroxysmal Positional Vertigo": ["Vestibular exercises", "Meclizine", "Epley maneuver"],
    "Impetigo": ["Topical mupirocin", "Oral antibiotics", "Proper hygiene"],
    "Dimorphic hemorrhoids (piles)": ["Sitz bath", "Laxatives", "Hydrocortisone cream"],
    "Cervical spondylosis": ["Painkillers", "Physical therapy", "Neck exercises"],
    "Hyperthyroidism": ["Methimazole", "Beta blockers", "Radioiodine therapy"],
    "Varicose veins": ["Compression stockings", "Sclerotherapy", "Vein stripping surgery"],
    "Paralysis (brain hemorrhage)": ["Physiotherapy", "Anti-clot medications", "Rehabilitation care"],
    "Hepatitis D": ["Interferon alfa", "Liver monitoring", "Antivirals (if coinfected with B)"],
    "Gastroenteritis": ["ORS", "Zinc tablets", "Antiemetics"],
    "Hepatitis B": ["Antivirals", "Entecavir", "Tenofovir"],
    "Hepatitis C": ["Direct-acting antivirals (DAAs)", "Ribavirin", "Peginterferon"]
}



# === Health Check Route ===
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

# === Diagnosis Endpoint ===
@app.route('/diagnose', methods=['POST'])
def diagnose():
    try:
        symptoms_text = request.form.get('symptoms') or (request.json and request.json.get('symptoms'))
        if not symptoms_text:
            return jsonify({"error": "No symptoms provided"}), 400

        symptoms = [s.strip().lower().replace(" ", "_") for s in symptoms_text.split(',')]
        known_symptoms = set(mlb.classes_)
        valid_symptoms = [s for s in symptoms if s in known_symptoms]
        unknown_symptoms = [s for s in symptoms if s not in known_symptoms]

        if not valid_symptoms:
            return jsonify({
                "error": "No valid known symptoms provided",
                "unknown": unknown_symptoms
            }), 400

        input_vector = mlb.transform([valid_symptoms])
        prediction = model.predict(input_vector)[0]

        precautions = precautions_dict.get(prediction, {})
        precautions = list(filter(None, precautions.values())) if precautions else []

        severe_diseases = ["Malaria", "Hepatitis A", "Hypoglycemia"]
        mild_diseases = ["Fungal infection", "Common Cold", "Allergy", "Acne"]

        if prediction in severe_diseases:
            recommendation = "Consult a doctor immediately."
        elif prediction in mild_diseases:
            recommendation = "Rest and use OTC medicine."
        else:
            recommendation = "Rest or visit a doctor if symptoms persist."

        medicine_suggestions = hardcoded_medicines.get(prediction, ["Paracetamol","panadol"])

        return jsonify({
            "diagnosis": prediction,
            "recommendation": recommendation,
            "precautions": precautions,
            "alternatives": medicine_suggestions,
            "progression_risk": f"{round((len(valid_symptoms) / 10) * 100)}%",
            "unknown_symptoms": unknown_symptoms
        })

    except Exception as e:
        print("❌ Error in /diagnose:", traceback.format_exc())
        return jsonify({"error": "Something went wrong on the server."}), 500

# === Start Server ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)