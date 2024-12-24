from fastapi import FastAPI, Depends, HTTPException
from pydantic import ValidationError
import uvicorn

from dependencies import verify_api_key
from schemas import ContentQuery
from scoring_utils import (
    calculate_word_count_score,
    calculate_relevance_score,
    calculate_subjective_impression,
    calculate_readability_score,
    calculate_structure_score,
    calculate_semantic_similarity,
    calculate_ner_score,
    analyze_tone,
    calculate_diversity_score
)

app = FastAPI()

@app.post("/geo_score")
def geo_score(data: ContentQuery, _=Depends(verify_api_key)):
    try:
        content = data.content
        query = data.query
        weights = data.weights
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    word_count_score = calculate_word_count_score(content)
    relevance_score = calculate_relevance_score(content, query)
    subjective_score = calculate_subjective_impression(content)
    readability_score = calculate_readability_score(content)
    structure_score = calculate_structure_score(content)
    semantic_score = calculate_semantic_similarity(content, query)
    ner_score = calculate_ner_score(content, query)
    diversity_score = calculate_diversity_score(content)

    tone = analyze_tone(content)

    final_geo_score = (
        weights["word_count_score"] * word_count_score +
        weights["relevance_score"] * relevance_score +
        weights["subjective_score"] * subjective_score +
        weights["readability_score"] * readability_score +
        weights["structure_score"] * structure_score +
        weights["semantic_score"] * semantic_score +
        weights["ner_score"] * ner_score +
        weights["diversity_score"] * diversity_score
    )

    response = {
        "word_count_score": word_count_score,
        "relevance_score": relevance_score,
        "subjective_score": subjective_score,
        "readability_score": readability_score,
        "structure_score": structure_score,
        "semantic_score": semantic_score,
        "ner_score": ner_score,
        "diversity_score": diversity_score,
        "tone": tone,
        "geo_score": final_geo_score
    }

    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)