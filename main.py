from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError
import re
import numpy as np
from textblob import TextBlob
import textstat

app = FastAPI()

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

    geo_score = (0.4 * word_count_score) + (0.4 * relevance_score) + (0.2 * subjective_score)

    return {
        "word_count_score": word_count_score,
        "relevance_score": relevance_score,
        "subjective_score": subjective_score,
        "geo_score": geo_score
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)