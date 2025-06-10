import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from gemini_client import GeminiClient
from wikidata import WikidataMatcher
from knowledge_graph import KnowledgeGraph


class EducationalConceptExtractor:
    def __init__(self, curriculum_file: str):
        self.curriculum_file = curriculum_file
        self.gemini_client = GeminiClient()
        self.wikidata_matcher = WikidataMatcher()

        self.covered_concepts_file = (
            f"{curriculum_file.removesuffix('.csv')}_covered_concepts.csv"
        )
        self.matched_covered_concepts_file = (
            f"{curriculum_file.removesuffix('.csv')}_covered_concepts_matched.csv"
        )
        self.prerequisites_concepts_file = (
            f"{curriculum_file.removesuffix('.csv')}_prerequisites_concepts.csv"
        )
        self.matched_prerequisites_concepts_file = (
            f"{curriculum_file.removesuffix('.csv')}_prerequisites_concepts_matched.csv"
        )

        self.__extract_covered_concepts()
        self.__match_covered_concepts_to_wikidata()
        self.__extract_prerequisites_concepts()
        self.__match_prerequisites_concepts_to_wikidata()

        self.kg = KnowledgeGraph(
            self.matched_covered_concepts_file, self.matched_prerequisites_concepts_file, self.curriculum_file
        )

    def __extract_covered_concepts(self):
        if os.path.exists(self.covered_concepts_file):
            print(f"Covered concepts already extracted. Skipping extraction.")
            return pd.read_csv(self.covered_concepts_file)

        df = pd.read_csv(self.curriculum_file, index_col=0)
        extracted_concepts = []

        for _, row in df.iterrows():
            course_name, course_content = row["course_name"], row["program_content"]
            covered_concepts = self.gemini_client.extract_covered_concepts(
                course_name, course_content
            )
            extracted_concepts.append(
                {"course_name": course_name, "covered_concepts": covered_concepts}
            )
            print(f"Extracted concepts for {course_name}: {covered_concepts}")
            print(f"Total time: {self.gemini_client.total_time}")

        concepts_df = pd.DataFrame(extracted_concepts)
        concepts_df.to_csv(self.covered_concepts_file, index=False)
        return concepts_df
    
    def __extract_prerequisites_concepts(self):
        if os.path.exists(self.prerequisites_concepts_file):
            print(f"Prerequisites concepts already extracted. Skipping extraction.")
            return pd.read_csv(self.prerequisites_concepts_file)

        df = pd.read_csv(self.curriculum_file, index_col=0)
        extracted_concepts = []

        for _, row in df.iterrows():
            course_name, course_content = row["course_name"], row["prerequisites"]
            if pd.isna(course_content) or isinstance(course_content, float) or course_content.strip() == "":
                print(f"NO prerequisites concepts for {course_name}")
                continue

            prerequisites_concepts = self.gemini_client.extract_prerequisite_concepts(
                course_name, course_content
            )
            if len(prerequisites_concepts)==0 or prerequisites_concepts[0]=="none":
                print(f"NO prerequisites concepts for {course_name}")
                continue

            extracted_concepts.append(
                {"course_name": course_name, "prerequisites_concepts": prerequisites_concepts}
            )
            print(f"Extracted prerequisites concepts for {course_name}: {prerequisites_concepts}")
            print(f"Total time: {self.gemini_client.total_time}")

        concepts_df = pd.DataFrame(extracted_concepts)
        concepts_df.to_csv(self.prerequisites_concepts_file, index=False)
        return concepts_df

    def __match_covered_concepts_to_wikidata(self):
        if os.path.exists(self.matched_covered_concepts_file):
            print(f"Concepts already matched to Wikidata. Skipping matching.")
            return pd.read_csv(self.matched_covered_concepts_file)

        concepts_df = pd.read_csv(self.covered_concepts_file)

        matched_rows = []

        for _, row in concepts_df.iterrows():
            course_name = row["course_name"]
            concepts = (
                eval(row["covered_concepts"])
                if isinstance(row["covered_concepts"], str)
                else row["covered_concepts"]
            )

            for concept in concepts:
                entity = self.wikidata_matcher.search_entity(concept, course_name)
                print(entity)

                if entity:
                    matched_rows.append(
                        {
                            "course_name": course_name,
                            "concept": concept,
                            "wikidata_qid": entity.qid,
                            "wikidata_label": entity.label,
                            "wikidata_description": entity.description,
                            "wikidata_url": entity.url,
                        }
                    )
                else:
                    matched_rows.append(
                        {
                            "course_name": course_name,
                            "concept": concept,
                            "wikidata_qid": None,
                            "wikidata_label": None,
                            "wikidata_description": None,
                            "wikidata_url": None,
                        }
                    )

        matched_df = pd.DataFrame(matched_rows)
        matched_df.to_csv(self.matched_covered_concepts_file, index=False)
    
    def __match_prerequisites_concepts_to_wikidata(self):
        if os.path.exists(self.matched_prerequisites_concepts_file):
            print(f"Concepts already matched to Wikidata. Skipping matching.")
            return pd.read_csv(self.matched_prerequisites_concepts_file)

        concepts_df = pd.read_csv(self.prerequisites_concepts_file)

        matched_rows = []

        for _, row in concepts_df.iterrows():
            course_name = row["course_name"]
            concepts = (
                eval(row["prerequisites_concepts"])
                if isinstance(row["prerequisites_concepts"], str)
                else row["prerequisites_concepts"]
            )

            for concept in concepts:
                entity = self.wikidata_matcher.search_entity(concept, course_name)
                print(entity)

                if entity:
                    matched_rows.append(
                        {
                            "course_name": course_name,
                            "concept": concept,
                            "wikidata_qid": entity.qid,
                            "wikidata_label": entity.label,
                            "wikidata_description": entity.description,
                            "wikidata_url": entity.url,
                        }
                    )
                else:
                    matched_rows.append(
                        {
                            "course_name": course_name,
                            "concept": concept,
                            "wikidata_qid": None,
                            "wikidata_label": None,
                            "wikidata_description": None,
                            "wikidata_url": None,
                        }
                    )

        matched_df = pd.DataFrame(matched_rows)
        matched_df.to_csv(self.matched_prerequisites_concepts_file, index=False)