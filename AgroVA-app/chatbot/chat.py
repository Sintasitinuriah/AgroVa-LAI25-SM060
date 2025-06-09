import json
import random
from string import Formatter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load intents
with open(r'intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Data tanaman dan penyakit
plant_data = {
    "plant": ["padi", "gandum", "singkong", "kentang"],
    "disease_list": {
        "padi": ["blast", "tungro", "hawar daun bakteri", "daun layu"],
        "gandum": ["bercak daun", "karat daun"],
        "singkong": ["mosaik", "busuk akar"],
        "kentang": ["busuk daun", "bercak daun"]
    },
    "growth_stage": {
        "padi": ["vegetatif", "generatif"],
        "kentang": ["pertumbuhan awal", "berbunga"],
        "singkong": ["awal tanam", "panen"],
        "gandum": ["semai", "pembungaan"]
    },
    "weather_condition": {
        "padi": "kelembapan tinggi dan hujan terus-menerus",
        "gandum": "angin kencang dan kelembapan sedang",
        "singkong": "cuaca lembap",
        "kentang": "hujan deras dan tanah basah"
    },
    "season": {
        "padi": "musim hujan",
        "gandum": "musim semi",
        "singkong": "musim panas",
        "kentang": "musim hujan"
    },
    "side_effects": {
        "blast": "penurunan hasil panen",
        "tungro": "pertumbuhan terhambat",
        "mosaik": "daun keriting",
        "busuk daun": "tanaman cepat mati",
        "hawar daun bakteri": "mengeringnya daun"
    },
    "varieties": {
        "padi": ["IR64", "Inpari 32"],
        "kentang": ["Granola", "Atlantik"],
        "singkong": ["Malang 4", "Adira 1"],
        "gandum": ["Gandum Merah", "Gandum Putih"]
    },
    "control": {
        "blast": "gunakan fungisida dan varietas tahan penyakit",
        "tungro": "pengendalian vektor dengan insektisida",
        "busuk daun": "semprot fungisida secara rutin",
        "karat daun": "rotasi tanaman dan fungisida",
        "mosaik": "gunakan benih bebas virus"
    },
    "pathogen": {
        "blast": "jamur Pyricularia oryzae",
        "tungro": "virus tungro",
        "busuk daun": "jamur Phytophthora infestans",
        "karat daun": "jamur Puccinia graminis",
        "mosaik": "virus mosaik"
    },
    "symptoms": {
        "blast": "bercak putih abu-abu pada daun",
        "tungro": "daun menguning dan kerdil",
        "busuk daun": "daun menghitam dan rontok",
        "karat daun": "bintik-bintik coklat kemerahan",
        "mosaik": "daun belang hijau muda"
    },
    "impact": {
        "blast": "penurunan produktivitas hingga 50%",
        "tungro": "kerusakan parah pada fase vegetatif",
        "mosaik": "kualitas umbi menurun",
        "busuk daun": "gagal panen"
    },
    "natural_treatment": {
        "blast": "ekstrak bawang putih sebagai fungisida alami",
        "tungro": "penggunaan tanaman refugia",
        "mosaik": "tanam varietas tahan dan rotasi tanaman",
        "busuk daun": "pestisida nabati dari daun mimba"
    },
    "rare_disease_list": {
        "padi": ["daun bergaris kuning"],
        "gandum": ["bengkak akar"],
        "singkong": ["busuk batang"],
        "kentang": ["bintik ungu daun"]
    }
}

# Persiapkan data penyakit untuk lookup
disease_data = {}
for plant, diseases in plant_data["disease_list"].items():
    for disease in diseases:
        disease_data[disease] = {
            "plant": plant,
            "symptoms": plant_data["symptoms"].get(disease, "-"),
            "pathogen": plant_data["pathogen"].get(disease, "-"),
            "impact": plant_data["impact"].get(disease, "-"),
            "side_effects": plant_data["side_effects"].get(disease, "-"),
            "control": plant_data["control"].get(disease, "-"),
            "natural_treatment": plant_data["natural_treatment"].get(disease, "-"),
        }

# Preprocessing
def preprocess(text):
    return text.lower()

# Cosine similarity-based intent matching
def match_intent(text):
    text = preprocess(text)
    all_examples = []
    intent_map = []

    for intent_obj in data['intents']:
        for example in intent_obj['pattern']:
            all_examples.append(preprocess(example))
            intent_map.append(intent_obj)

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_examples + [text])
    input_vector = vectors[-1]
    example_vectors = vectors[:-1]

    similarities = cosine_similarity(input_vector, example_vectors).flatten()
    max_score = max(similarities)
    best_match_index = similarities.argmax()

    if max_score < 0.3:  # Threshold kemiripan minimum
        return None
    return intent_map[best_match_index]

def extract_intent(text):
    intent_obj = match_intent(text)
    if intent_obj:
        return intent_obj.get('intent') or intent_obj.get('tag') or intent_obj.get('name')
    else:
        return None

def generate_response(intent_obj, text):
    if not intent_obj:
        return "Maaf, saya belum mengerti pertanyaan Anda. Bisa dijelaskan lagi?"

    text = preprocess(text)
    found_entities = {}

    # Cek penyakit
    detected_disease = None
    for disease in disease_data:
        if disease in text:
            detected_disease = disease
            found_entities.update(disease_data[disease])
            found_entities["disease"] = disease
            break

    # Cek tanaman
    detected_plant = None
    for plant in plant_data["plant"]:
        if plant in text:
            detected_plant = plant
            break

    if not detected_plant and detected_disease:
        detected_plant = disease_data[detected_disease]["plant"]

    if detected_plant:
        found_entities["plant"] = detected_plant
        found_entities["disease_list"] = ', '.join(plant_data["disease_list"].get(detected_plant, []))
        found_entities["growth_stage"] = ', '.join(plant_data["growth_stage"].get(detected_plant, []))
        found_entities["weather_condition"] = plant_data["weather_condition"].get(detected_plant, "")
        found_entities["season"] = plant_data["season"].get(detected_plant, "")
        found_entities["rare_disease_list"] = ', '.join(plant_data["rare_disease_list"].get(detected_plant, []))
        found_entities["varieties"] = ', '.join(plant_data["varieties"].get(detected_plant, []))
    else:
        found_entities["plant"] = "tanaman ini"

    response_template = random.choice(intent_obj['responses'])
    required_keys = [field_name for _, field_name, _, _ in Formatter().parse(response_template) if field_name]

    for key in required_keys:
        if key not in found_entities:
            found_entities[key] = "-"

    return response_template.format(**found_entities)

def chatbot_response(user_input):
    text = preprocess(user_input)
    intent_obj = match_intent(text)
    response = generate_response(intent_obj, text)
    intent_name = extract_intent(text)
    return intent_name, response

if __name__ == "__main__":
    print("Chatbot Prediksi Penyakit Tanaman. Ketik 'exit' untuk keluar.")
    while True:
        user_input = input("Anda: ")
        if user_input.lower() == 'exit':
            print("Terima kasih! Sampai jumpa.")
            break
        intent_name, reply = chatbot_response(user_input)
        print(f"Intent: {intent_name}")
        print("Bot:", reply)
