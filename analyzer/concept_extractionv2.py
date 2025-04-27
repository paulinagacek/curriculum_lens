import pandas as pd
from sentence_transformers import SentenceTransformer
from gemini_client import GeminiClient
import time

class EducationalConceptExtractor:
    def __init__(self, curriculum_file: str):
        self.curriculum_file = curriculum_file
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gemini_client = GeminiClient()
    
    def extract_covered_concepts(self):

        df = pd.read_csv(self.curriculum_file, index_col=0)
        for _, row in df.iterrows():
            course_name, course_content = row['course_name'], row['program_content']
            covered_concepts = self.gemini_client.extract_covered_concepts(course_name, course_content)
            print(covered_concepts)
            print(f"Total time: {self.gemini_client.total_time}")