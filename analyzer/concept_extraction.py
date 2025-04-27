import pandas as pd
import numpy as np
from flashtext import KeywordProcessor
import fasttext
from sklearn.metrics.pairwise import cosine_similarity
import fasttext.util


class EducationalConceptExtractor:
    def __init__(self, concepts_file: str, curriculum_file: str, similarity_threshold: float = 0.7):
        """
        Initializes the extractor with concept and curriculum files.
        :param concepts_file: Path to the wikidata concepts CSV.
        :param curriculum_file: Path to the curriculum CSV.
        :param fasttext_model_path: Path to the pretrained FastText model.
        :param similarity_threshold: Cosine similarity threshold for filtering concepts.
        """
        self.concepts_file = concepts_file
        self.curriculum_file = curriculum_file
        self.similarity_threshold = similarity_threshold

        # Load FastText model
        self.fasttext_model = fasttext.load_model('/data/cc.en.300.bin')
        self.vector_size = 300

        # Initialize keyword processor
        self.keyword_processor = KeywordProcessor()

        # Load concepts and curriculum
        self.load_concepts()
        self.load_curriculum()
        print("Extractor initialized")

    def load_concepts(self):
        """Loads the concept names and their fields, then precomputes field embeddings."""
        concepts_df = pd.read_csv(self.concepts_file)

        # Store concepts and their multiple fields
        self.concept_to_fields = {}
        self.concepts = set()

        for _, row in concepts_df.iterrows():
            concept_name = row["conceptName"].lower()
            field_name = str(row["fieldName"]).lower()

            if concept_name not in self.concept_to_fields:
                self.concept_to_fields[concept_name] = []
            
            self.concept_to_fields[concept_name].append(field_name)
            self.concepts.add(concept_name)

        # Precompute embeddings for each field (grouped by concept)
        self.field_embeddings = {
            concept: [self.text_to_embedding(field) for field in fields]
            for concept, fields in self.concept_to_fields.items()
        }

        # Add concepts to keyword processor for fast matching
        self.keyword_processor.add_keywords_from_list(list[self.concepts])

    def load_curriculum(self):
        """Loads the curriculum CSV into a DataFrame."""
        self.curriculum_df = pd.read_csv(self.curriculum_file)

    def text_to_embedding(self, text: str) -> np.ndarray:
        """Converts text into a FastText vector by averaging word embeddings."""
        words = text.split()
        word_vectors = [self.fasttext_model[word] for word in words if word in self.fasttext_model]

        if not word_vectors:  # If no words are found in FastText
            return np.zeros(self.vector_size)

        return np.mean(word_vectors, axis=0)

    def get_filtered_concepts(self, extracted_concepts: set, course_name: str) -> list:
        """
        Filters concepts based on cosine similarity between the course name and the multiple fields of a concept.
        :param extracted_concepts: Set of concepts extracted using fast keyword matching.
        :param course_name: The name of the course.
        :return: List of relevant concepts after filtering.
        """
        if not extracted_concepts:
            return []

        course_embedding = self.text_to_embedding(course_name)

        filtered_concepts = []
        for concept in extracted_concepts:
            field_embeddings = self.field_embeddings.get(concept, [])

            if not field_embeddings:
                continue  # Skip if no field embeddings exist

            # Compute similarity with all associated fields and take the highest
            similarities = [cosine_similarity([course_embedding], [field_emb])[0][0] for field_emb in field_embeddings]
            max_similarity = max(similarities, default=0)

            if max_similarity >= self.similarity_threshold:
                filtered_concepts.append(concept)

        return filtered_concepts

    def process_courses(self) -> pd.DataFrame:
        """Processes each course and extracts relevant concepts."""
        extracted_data = []

        for _, row in self.curriculum_df.iterrows():
            course_name = row["course_name"]
            
            # Step 1: Fast keyword extraction
            text_to_search = " ".join(str(row[col]) for col in ["prerequisites", "effects", "program_content", "program_content_ensuring_outcomes"] if col in row)
            extracted_concepts = set(self.keyword_processor.extract_keywords(text_to_search.lower()))

            # Step 2: Semantic filtering
            relevant_concepts = self.get_filtered_concepts(extracted_concepts, course_name)

            # Store results
            extracted_data.append({
                "course_name": course_name,
                "relevant_concepts": relevant_concepts,
            })
        
        return pd.DataFrame(extracted_data)

    def save_results(self, output_file: str):
        """Saves extracted concepts to a CSV file."""
        result_df = self.process_courses()
        result_df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

# Example Usage
# extractor = EducationalConceptExtractor("wikidata_concepts.csv", "curriculum.csv", "cc.en.300.vec")
# extractor.save_results("extracted_concepts.csv")
