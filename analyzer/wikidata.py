import os
import json
import time
import requests
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class WikidataEntity:
    def __init__(self, qid: str, label: str, description: str = ""):
        self.qid = qid
        self.label = label
        self.description = description
        self.url = f"https://www.wikidata.org/wiki/{qid}"

    def __str__(self):
        return f"Qid: {self.qid}, label: {self.label}, description: {self.description}"

    def __repr__(self):
        return f"WikidataEntity('{self.qid}', '{self.label}', '{self.description}')"

    def to_dict(self):
        return {"qid": self.qid, "label": self.label, "description": self.description}

    @staticmethod
    def from_dict(data):
        return WikidataEntity(data["qid"], data["label"], data.get("description", ""))


class WikidataMatcher:
    def __init__(
        self, similarity_threshold=0.5, cache_file: str = "/data/wikidata_cache.json"
    ):
        self.cache_file = cache_file
        self.cache = self.load_cache()
        self.max_nr_of_trials = 3
        self.similarity_threshold = similarity_threshold
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: WikidataEntity.from_dict(v) for k, v in data.items()}
        else:
            return {}

    def save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(
                {k: v.to_dict() for k, v in self.cache.items()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    def search_entity(self, query: str, course_name: str) -> Optional[WikidataEntity]:
        # Check local cache first
        if query in self.cache:
            return self.cache[query]

        # If not found, query Wikidata API
        entities = self._search_wikidata(query, course_name)

        if entities:
            entity = entities[0]  # Take the best matching entity
            if entity:
                self.cache[query] = entity
            self.save_cache()
            return entity
        else:
            return None

    def _search_wikidata(self, query: str, course_name:str) -> List[WikidataEntity]:
        search_url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 5,
            "srnamespace": "0",
        }
        trial_nr = 1
        while trial_nr <= self.max_nr_of_trials:
            response = requests.get(search_url, params=params)
            if response.status_code != 200:
                trial_nr += 1
                print("sleeping...")
                time.sleep(5)
                continue
                # response.raise_for_status()

            data = response.json()
            if "query" not in data or "search" not in data["query"]:
                return []

            titles = [item["title"] for item in data["query"]["search"]]
            entities = self._get_entities_details(titles)

            if not entities:
                return []
            
            filtered_entities = self._filter_entities(entities)

            if not filtered_entities:
                print(f"No valid Wikidata entities after filtering for query '{query}'.")
                return []

            return [self._get_best_match(query, course_name, filtered_entities)]

    def _get_best_match(
        self, query: str, course_name:str, candidates: List[WikidataEntity]
    ) -> WikidataEntity | None:
        query_embedding = self.embedding_model.encode(f"{query}")
        course_embedding = self.embedding_model.encode(f"{course_name}")

        scored = []
        for entity in candidates:
            text = (
                f"{entity.label}: {entity.description}"
                if entity.description
                else entity.label
            )

            candidate_embedding = self.embedding_model.encode(text)
            query_similarity = cosine_similarity([query_embedding], [candidate_embedding])[0][
                0
            ]
            course_similarity = cosine_similarity([course_embedding], [candidate_embedding])[0][
                0
            ]
            if query_similarity >= self.similarity_threshold or course_similarity >= self.similarity_threshold:
                scored.append((entity, query_similarity, course_similarity))
        print(scored)
        # Sort descending by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        if len(scored) == 0: return None

        best_match, best_query_score, best_course_score = scored[0]
        print(
            f"Best match for '{query}' = '{best_match.label}' (score: {best_query_score:.2f})"
        )

        return best_match

    def _get_entities_details(self, qids: List[str]) -> List[WikidataEntity]:
        if not qids:
            return []

        sparql_url = "https://query.wikidata.org/sparql"
        ids_formatted = " ".join(f"wd:{qid}" for qid in qids)
        query = f"""
        SELECT DISTINCT ?item ?itemLabel ?itemDescription
        WHERE {{
            VALUES ?item {{ {ids_formatted} }} 
            FILTER NOT EXISTS {{
                ?item wdt:P31 ?type .
                FILTER (?type IN (wd:Q7725634, wd:Q18918145, wd:Q13442814, wd:Q1368848))
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        trial_nr = 1
        while trial_nr <= self.max_nr_of_trials:
            r = requests.get(sparql_url, params={"format": "json", "query": query})
            if r.status_code != 200:
                trial_nr += 1
                print("sleeping sparql ...")
                time.sleep(5)
                continue

            data = r.json()
            entities = {}

            for item in data["results"]["bindings"]:
                label = item.get("itemLabel", {}).get("value", "")
                description = item.get("itemDescription", {}).get("value", "")
                qid = (
                    item.get("item", {})
                    .get("value")
                    .replace("http://www.wikidata.org/entity/", "")
                )
                entities[qid] = WikidataEntity(qid, label, description)

            return [entities[qid] for qid in qids if qid in entities]
    
    def _filter_entities(self, entities: List[WikidataEntity]) -> List[WikidataEntity]:
        filtered = []
        for e in entities:
            label = e.label or ""
            description = (e.description or "").lower()
            if label == "" or description == "": continue

            if "category:" in label.lower():
                continue
            if len(label.split()) > 4:
                continue
            if "scientific article" in description or "scholarly article" in description:
                continue

            filtered.append(e)
        return filtered