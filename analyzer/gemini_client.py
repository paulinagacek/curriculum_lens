import yaml
import time
from enum import Enum
from dataclasses import dataclass
from google import genai
import os


@dataclass
class GeminiConfig:
    model: str
    temperature: float
    max_nr_of_trials: int
    prompt_version: str
    prompts_file: str


class ExtractionTaskType(Enum):
    COVERED_CONCEPTS = "covered_concepts"
    PREREQUISITE_CONCEPTS = "prerequisite_concepts"


class GeminiClient:
    def __init__(self):
        self.config_file_path = "/config/gemini_config.yaml"
        self.config = self.__load_config()
        with open("/config/gemini_key.txt", "r") as f:
            self.client = genai.Client(api_key=f.readline())
        self.prompts_dir = "prompts"
        self.total_time = 0

    def extract_covered_concepts(self, course_name: str, description: str):
        return self.__generate_content(
            course_name, description, ExtractionTaskType.COVERED_CONCEPTS
        )

    def extract_prerequisite_concepts(self, course_name: str, description: str):
        return self.__generate_content(
            course_name, description, ExtractionTaskType.PREREQUISITE_CONCEPTS
        )

    def __generate_content(
        self, course_name: str, description: str, task_type: ExtractionTaskType
    ) -> str:
        nr_of_trials = 0
        while nr_of_trials < self.config.max_nr_of_trials:
            try:
                start = time.time()
                response = self.client.models.generate_content(
                    model=self.config.model,
                    contents=self.__create_prompt(course_name, description, task_type),
                    config={"temperature": self.config.temperature},
                )
                self.total_time += time.time()-start
                return self.__normalize_concepts(response.text)
            except Exception as e:
                nr_of_trials += 1
                print(e)
                time.sleep(30)

    def __create_prompt(
        self, course_name: str, description: str, task_type: ExtractionTaskType
    ):
        prompt_template = self.__read_prompt_template(task_type)
        return prompt_template.replace("<course_name>", course_name).replace(
            "<description>", description
        )

    def __read_prompt_template(self, task_type: ExtractionTaskType) -> str:
        version = self.config.prompt_version
        task_key = task_type.value

        prompt_path = os.path.join(self.prompts_dir, version, f"{task_key}.txt")

        if not os.path.exists(prompt_path):
            raise FileNotFoundError(f"Prompt template not found: {prompt_path}")

        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()


    def __load_config(self) -> GeminiConfig:
        with open(self.config_file_path, "r") as f:
            data = yaml.safe_load(f)
        return GeminiConfig(**data)
    
    def __normalize_concepts(self, concept_str):
        if isinstance(concept_str, str):
            return [c.strip().lower() for c in concept_str.split(';') if c.strip()]
        else:
            return []
