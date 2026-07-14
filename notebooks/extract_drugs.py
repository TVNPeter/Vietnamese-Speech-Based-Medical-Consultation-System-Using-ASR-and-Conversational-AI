# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///

import json
import re

def build_vn_regex():
    onsets = ["", "b", "c", "ch", "d", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n", "ng", "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x"]
    rhymes = [
        "a", "ai", "ao", "au", "ay", "am", "an", "ang", "anh", "ap", "at", "ac", "ach",
        "e", "eo", "em", "en", "eng", "ep", "et", "ec",
        "i", "ia", "iu", "im", "in", "inh", "ip", "it", "ich",
        "o", "oa", "oac", "oach", "oai", "oam", "oan", "oang", "oanh", "oap", "oat", "oay",
        "oe", "oec", "oem", "oen", "oeng", "oeo", "oep", "oet",
        "oi", "om", "on", "ong", "oong", "op", "ot", "oc", "ooc",
        "u", "ua", "uac", "uach", "uai", "uam", "uan", "uang", "uanh", "uap", "uat", "uay",
        "ue", "uec", "uem", "uen", "ueng", "ueu", "uep", "uet",
        "ui", "um", "un", "ung", "up", "ut", "uc",
        "uoi", "uom", "uon", "uong", "uop", "uot", "uoc",
        "y", "ya", "yec", "yem", "yen", "yeng", "yeu", "yep", "yet"
    ]
    onset_pattern = "|".join(onsets)
    rhyme_pattern = "|".join(rhymes)
    pattern = f"^({onset_pattern})({rhyme_pattern})$"
    return re.compile(pattern)

def is_foreign(word, vn_regex):
    word = word.lower()
    # Require alphanumeric ascii
    if not re.match(r'^[a-z0-9]+$', word):
        return False
    # Pure numbers are not foreign words
    if word.isdigit():
        return False
    # Single letters treated as foreign (e.g. vitamin a)
    if len(word) == 1:
        return True
    # If it matches valid unaccented Vietnamese, it's not foreign
    if vn_regex.match(word):
        return False
    return True

def main():
    input_file = 'randomqa.txt'
    output_file = 'drugs.txt'
    
    vn_regex = build_vn_regex()
    extracted_terms = set()
    
    with open(input_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            text = data.get('question', '') + " " + data.get('answer', '')
            
            # Find all words
            words = re.findall(r'[a-zA-Z0-9\u00C0-\u1EF9]+', text)
            
            current_term = []
            for w in words:
                if is_foreign(w, vn_regex):
                    current_term.append(w.lower())
                else:
                    if current_term:
                        term_str = " ".join(current_term)
                        if len(term_str) > 1: # exclude single isolated letters
                            extracted_terms.add(term_str)
                        current_term = []
            if current_term:
                term_str = " ".join(current_term)
                if len(term_str) > 1:
                    extracted_terms.add(term_str)
                    
    # Basic filtering to remove pure numbers or common noise
    final_terms = []
    for term in extracted_terms:
        # Avoid things like "1 2", "500 mg" (wait, mg is fine, but numbers alone in sequence no)
        if not re.match(r'^[0-9\s]+$', term): 
            final_terms.append(term)
            
    sorted_terms = sorted(final_terms)
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        for term in sorted_terms:
            fout.write(term + '\n')
            
    print(f"Extracted {len(sorted_terms)} medical/drug terms to {output_file}")

if __name__ == '__main__':
    main()
