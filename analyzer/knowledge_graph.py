from neo4j import GraphDatabase

URI = "bolt://memgraph:7687"
AUTH = ("testuser123", "t123")


class KnowledgeGraph:
    def __init__(self, covered_concepts_file: str, prerequisites_concepts_file: str, curriculum_file:str):
        self.add_covered_concepts(covered_concepts_file)
        self.add_prerequisites_concepts(prerequisites_concepts_file)
        self.add_course_details(curriculum_file)
        self.clean()

    def add_covered_concepts(self, covered_concepts_file: str):
        query = f"""LOAD CSV FROM '{covered_concepts_file}' WITH HEADER AS row
            MERGE (c:Course {{name: row.course_name}})

            MERGE (concept:Concept {{wikidata_qid: row.wikidata_qid}})
            ON CREATE SET
                concept.name = row.wikidata_label,
                concept.description = row.wikidata_description,
                concept.wikidata_url = row.wikidata_url

            MERGE (c)-[:COVERS]->(concept);"""
        self.__execute_query(query)

    def add_prerequisites_concepts(self, prerequisites_concepts_file: str):
        query = f"""LOAD CSV FROM '{prerequisites_concepts_file}' WITH HEADER AS row
            MERGE (c:Course {{name: row.course_name}})

            MERGE (concept:Concept {{wikidata_qid: row.wikidata_qid}})
            ON CREATE SET
                concept.name = row.wikidata_label,
                concept.description = row.wikidata_description,
                concept.wikidata_url = row.wikidata_url

            MERGE (c)-[:HAS_PREREQUISITE]->(concept);"""
        self.__execute_query(query)
    
    def add_course_details(self, curriculum_file:str):
        query = f"""LOAD CSV FROM '{curriculum_file}' WITH HEADER AS row
            MATCH (c:Course {{name: row.course_name}})
            SET c.term = toInteger(row.semester),
                c.ects = toInteger(row.ects);
            """
        self.__execute_query(query)
    
    def clean(self):
        self.__execute_query("MATCH (p) WHERE p.name = "" DETACH DELETE p;")

    def __execute_query(self, query):
        try:
            with GraphDatabase.driver(URI, auth=AUTH) as client:
                with client.session() as session:
                    print("Running query:", query)
                    result = session.run(query)
                    print(result.single())

                    session.run("FREE MEMORY")

        except BaseException as e:
            print("Failed to execute transaction")
            raise e
