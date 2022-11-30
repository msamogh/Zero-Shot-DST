from typing import *
from dataclasses import dataclass
import json


@dataclass
class Ontology:
    domain: Text
    slots: Dict[Text, List[Text]]
    descriptions: Dict[Text, Text]

    @classmethod
    def from_files(cls, domain_name, ontology_file, description_file):
        ontology = json.load(open(ontology_file, "r"))
        descriptions = json.load(open(description_file, "r"))
        return cls(
            domain=domain_name, slots=ontology, descriptions=descriptions
        )  # type: ignore
