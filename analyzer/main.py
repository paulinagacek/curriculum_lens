from concept_extraction import EducationalConceptExtractor

CURRICULUM_CSV = "/data/en_Informatyka_i_Systemy_Inteligentne_curriculum.csv"

if __name__ == "__main__":
    print("Analysis in progres...")
    extractor = EducationalConceptExtractor(CURRICULUM_CSV)