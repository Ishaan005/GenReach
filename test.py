import unittest
from fastapi.testclient import TestClient
from main import app, calculate_word_count_score, calculate_relevance_score, calculate_subjective_imporession

client = TestClient(app)

class TestMain(unittest.TestCase):

    def test_calculate_word_count_score(self):
        content = "This is a test content with a certain number of words."
        score = calculate_word_count_score(content)
        self.assertEqual(score, (min(len(content.split()) / 1000, 1)) * 100)

    def test_calculate_relevance_score(self):
        content = "This is a test content with a certain number of words."
        query = "test content"
        score = calculate_relevance_score(content, query)
        self.assertEqual(score, ((2 / 2) * 100))

    def test_calculate_subjective_imporession(self):
        content = "This is a test content with a certain number of words."
        score = calculate_subjective_imporession(content)
        self.assertIsInstance(score, float)

    def test_geo_score(self):
        response = client.post("/geo_score", json={"content": "This is a test content with a certain number of words.", "query": "test content"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("word_count_score", data)
        self.assertIn("relevance_score", data)
        self.assertIn("subjective_score", data)
        self.assertIn("geo_score", data)

    def test_geo_score_missing_content(self):
        response = client.post("/geo_score", json={"content": "", "query": "test content"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Content and Query are required"})

    def test_geo_score_missing_query(self):
        response = client.post("/geo_score", json={"content": "This is a test content with a certain number of words.", "query": ""})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Content and Query are required"})

if __name__ == '__main__':
    unittest.main()