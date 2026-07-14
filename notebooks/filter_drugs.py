# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

import re
import urllib.request
import os
import json

def download_file(url):
    print(f"Downloading {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        print(f"Error downloading: {e}")
        return ""

def main():
    # 1. Download English blacklist (370k words)
    blacklist_url = "https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt"
    blacklist_data = download_file(blacklist_url)
    blacklist = set(blacklist_data.lower().splitlines())
    
    # Add common medical non-drug terms to blacklist
    additional_blacklist = {
        "nausea", "vomiting", "diarrhea", "constipation", "headache", "fever", "pain", 
        "cough", "infection", "disease", "syndrome", "symptom", "treatment", "therapy",
        "patient", "doctor", "nurse", "hospital", "clinic", "pharmacy", "medicine",
        "effect", "side", "reaction", "allergy", "warning", "precaution", "dose", "dosage",
        "tablet", "capsule", "pill", "injection", "blood", "urine", "liver", "kidney",
        "heart", "lung", "brain", "stomach", "intestine", "skin", "cell", "tissue", "organ",
        "system", "body", "health", "clinical", "medical", "study", "research", "data",
        "receptor", "receptors", "enzyme", "enzymes", "protein", "proteins", "gene", "genes",
        "dna", "rna", "virus", "bacteria", "fungus", "parasite", "hormone", "hormones",
        "acid", "acids", "base", "bases", "salt", "salts", "water", "food", "diet", "nutrition",
        "weight", "age", "sex", "gender", "male", "female", "child", "adult", "elderly",
        "pregnancy", "lactation", "breastfeeding", "pregnant", "fetus", "infant", "newborn",
        "week", "month", "year", "day", "hour", "minute", "second", "time", "rate", "level",
        "concentration", "clearance", "absorption", "distribution", "metabolism", "excretion",
        "elimination", "half-life", "bioavailability", "solubility", "stability", "ph",
        "temperature", "pressure", "volume", "flow", "resistance", "capacity", "index",
        "ratio", "percentage", "fraction", "coefficient", "constant", "parameter", "variable",
        "factor", "cause", "mechanism", "pathway", "target", "marker", "indicator",
        "test", "assay", "analysis", "measurement", "detection", "quantification",
        "identification", "characterization", "purification", "extraction", "isolation",
        "synthesis", "production", "manufacture", "formulation", "preparation",
        "solution", "suspension", "emulsion", "gel", "ointment", "cream", "paste",
        "powder", "granule", "pellet", "bead", "matrix", "carrier", "excipient",
        "diluent", "binder", "disintegrant", "lubricant", "glidant", "plasticizer",
        "solvent", "buffer", "preservative", "antioxidant", "surfactant", "emulsifier",
        "stabilizer", "thickener", "gelling", "coating", "polymer", "starch", "cellulose",
        "lactose", "sucrose", "glucose", "mannitol", "sorbitol", "xylitol", "glycerol",
        "glycol", "water", "ethanol", "alcohol", "ether", "chloroform", "benzene",
        "trigger", "zone", "chemoreceptor", "medulla", "oblongata", "dopamine",
        "serotonin", "histamine", "acetylcholine", "epinephrine", "norepinephrine",
        "gaba", "glutamate", "glycine", "adenosine", "atp", "camp", "cgmp", "nitric", "oxide",
        "folic", "folate", "dihydrofolate", "tetrahydrofolate", "reductase", "synthetase",
        "folylpolyglutamate", "polyglutamat", "polyglutamate", "thiol", "sulfhydryl",
        "barrier", "mucosa", "mucus", "bicarbonate", "gastrin", "pepsin", "prostaglandin",
        "cyclooxygenase", "cox", "thromboxane", "leukotriene", "arachidonic", "phospholipase",
        "membrane", "lipid", "cholesterol", "triglyceride", "lipoprotein", "ldl", "hdl",
        "vldl", "chylomicron", "apolipoprotein", "agonist", "antagonist",
        "blocker", "inhibitor", "activator", "modulator", "channel", "transporter",
        "pump", "atpase", "sodium", "potassium", "calcium", "chloride", "magnesium",
        "phosphate", "salt", "salts", "sulfate", "carbonate", "ammonium", "manganese", "cobalt", "nickel",
        "chromium", "fluoride", "iodide", "selenium", "vitamin", "vitamins", "caroten",
        "inosit", "leucin", "natri", "section", "times", "donaipharm", "opc",
        "and", "the", "for", "with", "from", "that", "this", "these", "those",
        "actions", "activated", "activating", "active", "activity", "associated", "association",
        "based", "complex", "complexes", "contain", "containing", "derivative", "derivatives",
        "induced", "induction", "levels", "measured", "normal", "observed", "reduced", "reduction"
    }
    blacklist.update(additional_blacklist)
    
    # 2. Build Whitelist of known drugs/chemicals from multiple sources
    whitelist = set()
    
    # Source A: AutoNER BC5CDR Chemical dictionary
    autoner_url = "https://raw.githubusercontent.com/shangjingbo1226/AutoNER/master/data/BC5CDR/dict_core.txt"
    autoner_data = download_file(autoner_url)
    if autoner_data:
        for line in autoner_data.splitlines():
            if not line.strip():
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                entity_type, name = parts[0], parts[1]
                if entity_type.lower() == "chemical":
                    for word in name.lower().split():
                        word_clean = re.sub(r'[^a-z]', '', word)
                        if len(word_clean) > 2 and word_clean not in blacklist:
                            whitelist.add(word_clean)
                            
        print(f"Loaded {len(whitelist)} chemical words in whitelist.")
    
    # Source B: NLM Drugs (from corpora repo)
    nlm_url = "https://raw.githubusercontent.com/dariusk/corpora/master/data/medicine/drugs.json"
    nlm_data = download_file(nlm_url)
    if nlm_data:
        try:
            data = json.loads(nlm_data)
            drugs_list = data.get('drugs', [])
            for drug in drugs_list:
                for word in drug.lower().split():
                    word_clean = re.sub(r'[^a-z]', '', word)
                    if len(word_clean) > 2 and word_clean not in blacklist:
                        whitelist.add(word_clean)
        except Exception as e:
            print(f"Error parsing NLM drugs: {e}")
            
    # Some absolute whitelisted Vietnamese pharmaceutical terms
    viet_whitelist = {
        "panadol", "decolgen", "tiffy", "hapacol", "boganic", "efferalgan", "eferalgan",
        "solumedrol", "depomedrol", "prednisolone", "methylprednisolone", "paracetamol",
        "amoxicillin", "ampicillin", "cephalexin", "cefalexin", "ceftriaxone", "cefuroxime",
        "cefaclor", "cefixime", "ciprofloxacin", "ofloxacin", "levofloxacin", "azithromycin",
        "erythromycin", "clarithromycin", "tetracycline", "doxycycline", "metronidazole",
        "aspirin", "ibuprofen", "diclofenac", "meloxicam", "piroxicam", "celecoxib",
        "etoricoxib", "omeprazole", "pantoprazole", "rabeprazole", "lansoprazole",
        "eszopiclone", "diazepam", "lorazepam", "alprazolam", "clonazepam", "bromazepam",
        "salbutamol", "albuterol", "fluticasone", "budesonide", "tiotropium", "ipratropium",
        "montelukast", "prednisone", "hydrocortisone", "dexamethasone", "betamethasone",
        "insulin", "metformin", "gliclazide", "glimepiride", "pioglitazone", "acarbose",
        "atorvastatin", "simvastatin", "rosuvastatin", "fenofibrate", "gemfibrozil",
        "amlodipine", "nifedipine", "felodipine", "diltiazem", "verapamil", "captopril",
        "enalapril", "lisinopril", "perindopril", "losartan", "valsartan", "irbesartan",
        "telmisartan", "bisoprolol", "metoprolol", "atenolol", "propranolol", "carvedilol",
        "spironolactone", "furosemide", "hydrochlorothiazide", "ranitidine", "famotidine",
        "cimetidine", "domperidone", "metoclopramide", "ondansetron", "loperamide",
        "promethazine", "chlorpheniramine", "loratadine", "cetirizine", "fexofenadine",
        "levocetirizine", "desloratadine", "acyclovir", "valacyclovir", "oseltamivir",
        "ganciclovir", "entecavir", "tenofovir", "lamivudine", "efavirenz", "nevirapine",
        "amphotericin", "fluconazole", "itraconazole", "ketoconazole", "clotrimazole",
        "miconazole", "nystatin", "albendazole", "mebendazole", "ivermectin", "chloroquine",
        "artemisinin", "quinine", "methotrexate", "azathioprine", "cyclosporine",
        "tacrolimus", "mycophenolate", "cyclophosphamide", "doxorubicin", "paclitaxel",
        "docetaxel", "cisplatin", "carboplatin", "oxaliplatin", "fluorouracil", "gemcitabine",
        "capecitabine", "imatinib", "erlotinib", "gefitinib", "vemurafenib", "rituximab",
        "trastuzumab", "bevacizumab", "cetuximab", "infliximab", "adalimumab", "etanercept",
        "warfarin", "heparin", "enoxaparin", "clopidogrel", "aspirin", "dipyridamole",
        "toptana", "streptokinase", "urokinase", "alteplase", "epoetin", "filgrastim",
        "sildenafil", "tadalafil", "vardenafil", "alprostadil", "testosterone", "estradiol",
        "progesterone", "medroxyprogesterone", "levonorgestrel", "etanercept", "interferon"
    }
    whitelist.update(viet_whitelist)
    
    print(f"Loaded total of {len(whitelist)} drug/chemical words in whitelist.")
    
    # 3. Read and split drugs.txt
    input_file = 'drugs.txt'
    output_file = 'drugs_candidates.txt'
    
    print("Regenerating raw drugs.txt using extract_drugs.py...")
    import extract_drugs
    extract_drugs.main()
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return
        
    with open(input_file, 'r', encoding='utf-8') as f:
        phrases = [line.strip() for line in f if line.strip()]
        
    # Split phrases into unique single words
    words_set = set()
    for phrase in phrases:
        words_set.update(phrase.split())
        
    words = sorted(list(words_set))
    print(f"Original unique single words count: {len(words)}")
    
    # Drug suffixes regex
    suffix_regex = re.compile(
        r'(ol|in|ine|one|ide|vir|pam|lam|pril|sartan|statin|cef|cin|cillin|mab|zole|pine|'
        r'ate|ole|tadine|zine|mine|drine|tinib|ciclib|parib|asone|olone|nide|setron|'
        r'gen|dol|col|gan|can|bar|bac|pin|tin)$'
    )
    
    filtered_words = []
    for w in words:
        w_lower = w.lower()
        w_clean = re.sub(r'[^a-z]', '', w_lower)
        
        # 1. Filter out words containing digits
        if re.search(r'[0-9]', w_lower):
            continue
            
        # 2. Filter out short words
        if len(w_clean) < 3:
            continue
            
        # 3. Whitelist check (ALWAYS KEEP if in whitelist)
        if w_clean in whitelist:
            filtered_words.append(w)
            continue
            
        # 4. Blacklist check (REMOVE if in blacklist)
        if w_clean in blacklist:
            continue
            
        # 5. Suffix check (KEEP if it matches typical drug ending)
        if suffix_regex.search(w_clean):
            filtered_words.append(w)
            continue
            
    # Sort and unique
    sorted_filtered = sorted(list(set(filtered_words)))
    
    # Overwrite drugs.txt
    with open(input_file, 'w', encoding='utf-8') as f:
        for w in sorted_filtered:
            f.write(w + '\n')
            
    with open(output_file, 'w', encoding='utf-8') as f:
        for w in sorted_filtered:
            f.write(w + '\n')
                
    print(f"Done! Overwrote drugs.txt with {len(sorted_filtered)} highly refined drug/medical terms.")

if __name__ == '__main__':
    main()
