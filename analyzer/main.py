import pandas as pd

CURRICULUM_CSV = "/data/en_Informatyka_i_Systemy_Inteligentne_curriculum.csv"

if __name__ == "__main__":
    print("Analysis in progres...")
    print(pd.read_csv(CURRICULUM_CSV).columns)