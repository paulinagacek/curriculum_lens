import pandas as pd
from concept_extractionv2 import EducationalConceptExtractor

CURRICULUM_CSV = "/data/en_Informatyka_i_Systemy_Inteligentne_curriculum.csv"
WIKIDATA_CONCEPTS_CSV = "/data/wikidata_concepts.csv"

if __name__ == "__main__":
    print("Analysis in progres...")

    extractor = EducationalConceptExtractor(CURRICULUM_CSV)
    extractor.extract_covered_concepts()