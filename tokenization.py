import json
import spacy

def get_target_vector(prompt: str, vocab: set) -> dict:    

    # STEP 2 – Define fuzzy modifiers
    fuzzy_modifiers = {
        "not": 0.0,
        "barely": 0.1,
        "a bit": 0.2,
        "a little": 0.3,
        "slightly": 0.35,
        "somewhat": 0.5,
        "kind of": 0.55,
        "pretty": 0.6,
        "fairly": 0.65,
        "moderately": 0.7,
        "rather": 0.75,
        "quite": 0.8,
        "really": 0.9,
        "very": 1.0,
        "super": 1.0,
        "extremely": 1.0,
        "incredibly": 1.0,
        "absolutely": 1.0,
        "totally": 1.0,
        "completely": 1.0,
        "just a bit": 0.2,
        "a tad": 0.25,
        "not very": 0.2,
        "not really": 0.1,
        "not quite": 0.2
    }

    category_weights = {
        "core": 1.0,
        "modifier": 0.9,
        "accessory": 0.4,
        "background": 0.5,
    }

    tokens = prompt.lower().split()
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(prompt)

    raw_keywords = set()
    skip_next = False
    for i in range(len(tokens)):
        if skip_next:
            skip_next = False
            continue
        word = tokens[i]
        if i < len(tokens) - 1:
            two_word = f"{word} {tokens[i+1]}"
            if two_word in fuzzy_modifiers:
                if i + 2 < len(tokens):
                    target = tokens[i+2]
                    if target in vocab:
                        raw_keywords.add(target)
                        skip_next = True
                        continue
        if word in fuzzy_modifiers:
            if i + 1 < len(tokens) and tokens[i+1] in vocab:
                raw_keywords.add(tokens[i+1])
                skip_next = True
            continue
        if word in vocab:
            raw_keywords.add(word)

    core_keywords = set()
    modifier_keywords = set()
    accessory_keywords = set()

    for token in doc:
        print(f"{token.text:<10} | POS: {token.pos_:<6} | Dep: {token.dep_:<10} | Head: {token.head.text:<10} | Lemma: {token.lemma_}")

    for token in doc:
        lemma = token.lemma_.lower()
        if lemma not in raw_keywords:
            continue
        if token.pos_ in {"ADJ", "ADV"}:
            modifier_keywords.add(lemma)
        elif token.dep_ in {"pobj", "dobj", "prep"}:
            accessory_keywords.add(lemma)
        elif token.dep_ in {"nsubj", "ROOT"}:
            core_keywords.add(lemma)
        elif token.pos_ == "NOUN":
            core_keywords.add(lemma)
        else:
            accessory_keywords.add(lemma)

    role_map = {
        "core": list(core_keywords),
        "modifier": list(modifier_keywords),
        "accessory": list(accessory_keywords)
    }

    fuzzy_vector = {}
    i = 0
    used = set()

    while i < len(tokens):
        word = tokens[i]
        modifier_value = 1.0
        keyword = None

        if i < len(tokens) - 1:
            two_word = f"{word} {tokens[i+1]}"
            if two_word in fuzzy_modifiers:
                modifier_value = fuzzy_modifiers[two_word]
                if i + 2 < len(tokens):
                    keyword = tokens[i+2]
                    i += 2
            elif word in fuzzy_modifiers and tokens[i+1] in vocab:
                modifier_value = fuzzy_modifiers[word]
                keyword = tokens[i+1]
                i += 1
        elif word in vocab:
            keyword = word

        if keyword and keyword not in used:
            for category, keywords in role_map.items():
                if keyword in keywords:
                    category_weight = category_weights[category]
                    fuzzy_vector[keyword] = round(modifier_value * category_weight, 3)
                    used.add(keyword)
                    break
        i += 1

    for category, keywords in role_map.items():
        for keyword in keywords:
            if keyword not in fuzzy_vector:
                fuzzy_vector[keyword] = category_weights[category]

    # Extract spatial relations dynamically
    spatial_map = []

    for token in doc:
        if token.pos_ == "VERB":
            verb = token.lemma_
            subject = None
            obj = None

            # Try to find an explicit subject
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass"}:
                    subject = child.lemma_
                elif child.dep_ == "dobj":
                    obj = child.lemma_

            # Inherit subject from the root verb if this is a conj
            if token.dep_ == "conj" and not subject:
                for ancestor in token.ancestors:
                    if ancestor.pos_ == "VERB":
                        for child in ancestor.children:
                            if child.dep_ in {"nsubj", "nsubjpass"}:
                                subject = child.lemma_
                                break

            if subject and obj:
                spatial_map.append((subject, verb, obj))

    # Detect background using prep → pobj pattern
    background = None
    for token in doc:
        if token.dep_ == "pobj" and token.head.dep_ == "prep":
            background = token.lemma_.lower()

    # Add background to fuzzy vector
    if background not in fuzzy_vector:
            fuzzy_vector[background] = category_weights["background"]

    return fuzzy_vector, role_map, spatial_map, background