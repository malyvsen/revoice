from typing import Dict
from dataclasses import dataclass
from epitran import Epitran
from .ipa import mapping as default_ipa_mapping


@dataclass(frozen=True)
class Transcriber:
    epitran: Epitran
    ipa_mapping: Dict[str, str]

    @classmethod
    def from_language_code(
        cls, language_code, ipa_mapping=default_ipa_mapping
    ) -> "Transcriber":
        return cls(epitran=Epitran(language_code), ipa_mapping=ipa_mapping)

    def transcribe(self, text: str) -> str:
        ipa_text = self.epitran.transliterate(text)
        result = []
        idx = 0
        while idx < len(ipa_text):
            match, transcription = self.match_start(ipa_text[idx:])
            idx += len(match)
            result.append(transcription)
        return "".join(result)

    @property
    def prioritized_ipa_mapping(self):
        return {
            ipa: self.ipa_mapping[ipa]
            for ipa in sorted(
                self.ipa_mapping.keys(), key=lambda ipa: len(ipa), reverse=True
            )
        }

    def match_start(self, ipa_text):
        for ipa, transcription in self.prioritized_ipa_mapping.items():
            if ipa_text.startswith(ipa):
                return ipa, transcription
        return ipa_text[0], ipa_text[0]
