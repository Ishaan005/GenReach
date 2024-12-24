import re
import numpy as np
import textstat
from textblob import TextBlob
import spacy
from sentence_transformers import SentenceTransformer

semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

def calculate_word_count_score(content):
    word_count = len(content.split())
    return (min(word_count / 1000, 1)) * 100

def calculate_relevance_score(content, query):
    query_terms = query.lower().split()
    content_terms = content.lower().split()
    match_count = sum(1 for term in query_terms if term in content_terms)
    return ((match_count / len(query_terms)) * 100) if query_terms else 0

def calculate_subjective_impression(content):
    text_blob = TextBlob(content)
    polarity = text_blob.sentiment.polarity
    subjectivity = text_blob.sentiment.subjectivity
    fluency_score = len(re.findall(r'\.', content)) / len(content.split())
    return np.mean([polarity, subjectivity, fluency_score]) * 100

def calculate_readability_score(content):
    return textstat.flesch_reading_ease(content)

def calculate_structure_score(content):
    header_count = len(re.findall(r'<h[1-6]>.*?</h[1-6]>', content, re.IGNORECASE))
    bullet_count = len(re.findall(r'<li>.*?</li>', content, re.IGNORECASE))
    paragraph_count = len(re.findall(r'<p>.*?</p>', content, re.IGNORECASE))
    structure_score = min((header_count + bullet_count + paragraph_count) / 10, 1)
    return structure_score

def calculate_semantic_similarity(content, query):
    content_embedding = semantic_model.encode(content)
    query_embedding = semantic_model.encode(query)
    similarity = (
        np.dot(content_embedding, query_embedding) /
        (np.linalg.norm(content_embedding) * np.linalg.norm(query_embedding))
    )
    return similarity

def calculate_ner_score(content, query):
    content_doc = nlp(content)
    query_doc = nlp(query)
    content_entities = {ent.text.lower() for ent in content_doc.ents}
    query_entities = {ent.text.lower() for ent in query_doc.ents}
    matching_entites = content_entities.intersection(query_entities)
    return (len(matching_entites) / len(query_entities)) if query_entities else 0

def analyze_tone(content):
    tone_keywords = {
        "formal": ["therefore", "hence", "moreover", "however", "thus"],
        "casual": ["hey", "cool", "awesome", "pretty", "yeah"],
        "persuasive": ["imagine", "guarantee", "proven", "effective", "success"]
    }
    content_lower = content.lower()
    tone_scores = {
        tone: sum(content_lower.count(keyword) for keyword in keywords)
        for tone, keywords in tone_keywords.items()
    }
    predominant_tone = max(tone_scores, key=tone_scores.get)
    return {"predominant_tone": predominant_tone, "tone_scores": tone_scores}

def calculate_diversity_score(content):
    sentences = [sent.text for sent in nlp(content).sents]
    unique_sentence_structures = len(set(sentences)) / len(sentences) if sentences else 0
    words = content.split()
    unique_words = len(set(words)) / len(words) if words else 0
    return np.mean([unique_sentence_structures, unique_words])