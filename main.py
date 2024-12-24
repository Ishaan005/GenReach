from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import re
import numpy as np
from textblob import TextBlob
import textstat
from sentence_transformers import SentenceTransformer
import spacy

app = FastAPI()

#Load Models
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load("en_core_web_sm")

#Input Schema
class ContentQuery(BaseModel):
    content: str
    query: str

#Caclulate normalised workd count
def calculate_word_count_score(content):
    word_count = len(content.split())
    return (min(word_count / 1000 , 1)) * 100

#CHeck how many query terms are present in the content
def calculate_relevance_score(content, query):
    query_terms = query.lower().split()
    content_terms = content.lower().split()
    match_count = sum(1 for term in query_terms if term in content_terms)
    return ((match_count / len(query_terms)) * 100) if query_terms else 0

def calculate_subjective_imporession(content):
    text_blob = TextBlob(content)
    polarity = text_blob.sentiment.polarity # Sentiment Ploarity
    subjectivity = text_blob.sentiment.subjectivity
    fluency_score = len(re.findall(r'\.', content)) / len(content.split())

    return np.mean([polarity, subjectivity, fluency_score]) * 100

#TODO: Find a better way to calculate readability score
#Calculate the readability score using Flesch-Kincaid.
def calculate_readability_score(content):
    return textstat.flesch_reading_ease(content)

#TODO: Find a better way to calculate structure score
#Calculate the structure score
def calculate_structure_score(content):
    """Analyze content structure, looking for headers, bullet points, and paragraphs."""
    header_count = len(re.findall(r'<h[1-6]>.*?</h[1-6]>', content, re.IGNORECASE))
    bullet_count = len(re.findall(r'<li>.*?</li>', content, re.IGNORECASE))
    paragraph_count = len(re.findall(r'<p>.*?</p>', content, re.IGNORECASE))
    structure_score = min((header_count + bullet_count + paragraph_count) / 10, 1)  # Normalize
    return structure_score

#Calculate the semantic similarity between content and query
def calculate_semantic_similarity(content, query):
    content_embedding = semantic_model.encode(content)
    query_embedding = semantic_model.encode(query)
    similarity = np.dot(content_embedding, query_embedding) / (np.linalg.norm(content_embedding) * np.linalg.norm(query_embedding))
    return similarity

#Check if entites in the query appear in the content
def calculate_ner_score(content, query):
    content_doc = nlp(content)
    query_doc = nlp(query)

    content_entities = {ent.text.lower() for ent in content_doc.ents}
    query_entities = {ent.text.lower() for ent in query_doc.ents}
    
    matching_entites = content_entities.intersection(query_entities)

    return (len(matching_entites) / len(query_entities)) if query_entities else 0


@app.post("/geo_score")
def geo_score(data:ContentQuery):
    try:
        content = data.content
        query = data.query
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    word_count_score = calculate_word_count_score(content)
    relevance_score = calculate_relevance_score(content, query)
    subjective_score = calculate_subjective_imporession(content)
    readability_score = calculate_readability_score(content)
    structure_score = calculate_structure_score(content)
    semantic_score = calculate_semantic_similarity(content, query)
    ner_score = calculate_ner_score(content, query)

    #TODO: Adjust weights
    geo_score = (0.3 * word_count_score +
                 0.2 * relevance_score +
                 0.1 * subjective_score +
                 0.1 * readability_score +
                 0.1 * structure_score + 
                 0.1 * semantic_score +
                 0.1 * ner_score)

    return {
        "word_count_score": word_count_score,
        "relevance_score": relevance_score, 
        "subjective_score": subjective_score,
        "readability_score": readability_score,
        "structure_score": structure_score,
        "semantic_score": semantic_score,
        "ner_score": ner_score,
        "geo_score": geo_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)