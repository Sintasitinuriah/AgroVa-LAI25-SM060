import json
from googletrans import Translator

# Inisialisasi penerjemah
translator = Translator()

# Muat file intents.json
with open('intents.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Fungsi untuk menerjemahkan list teks
def translate_list(texts, dest='id'):
    return [translator.translate(text, dest=dest).text for text in texts]

# Terjemahkan patterns dan responses
for intent in data["intents"]:
    if "patterns" in intent:
        intent["patterns_translated"] = translate_list(intent["patterns"])
    if "responses" in intent:
        intent["responses_translated"] = translate_list(intent["responses"])

# Simpan hasil ke file baru
with open('translated_intents.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Terjemahan selesai dan disimpan ke 'translated_intents.json'")
