"""metadata.py — JSON-LD scholarly metadata (schema.org/ScholarlyArticle).

Outputs to: metadata/paper.jsonld
Format: JSON-LD
Visibility: public
"""

import json
from datetime import datetime
from pathlib import Path

from orchestrator.deliverable_handlers.base import DeliverableHandler


class MetadataHandler(DeliverableHandler):
    """
    Generate JSON-LD scholarly metadata using schema.org/ScholarlyArticle.

    Most of this can be built deterministically from paper config without
    AI; we use Claude only for keywords and abstract refinement.
    """

    DEFAULT_MAX_TOKENS = 1500
    DEFAULT_TEMPERATURE = 0.3

    def run(self) -> Path:
        """Generate JSON-LD deterministically from paper config."""
        output_path = self.resolve_output_path()

        metadata = {
            "@context": "https://schema.org",
            "@type": "ScholarlyArticle",
            "name": self.paper.title,
            "headline": self.paper.title,
            "author": [self._author_block()],
            "isPartOf": self._publication_block(),
            "datePublished": self._date_published(),
            "dateModified": datetime.now().isoformat(timespec='seconds'),
            "inLanguage": self.paper.drafting_context.get('default_language', 'en'),
            "license": self._license_url(),
            "isAccessibleForFree": True,
            "keywords": self._keywords(),
            "abstract": self.paper.drafting_context.get('abstract', ''),
            "identifier": {
                "@type": "PropertyValue",
                "propertyID": "internal-id",
                "value": self.paper.id,
            },
            "subjectOf": {
                "@type": "Dataset",
                "name": f"Simulation outputs for {self.paper.id}",
                "license": "https://creativecommons.org/licenses/by/4.0/",
            },
            "_orchestrator_version": "0.1.0",
            "_generated": datetime.now().isoformat(timespec='seconds'),
        }

        # Validate it's actually serializable JSON
        text = json.dumps(metadata, indent=2, ensure_ascii=False)

        # Validation skipped for JSON (no forbidden patterns useful here)
        self.write_output(text, output_path)
        return output_path

    def _author_block(self) -> dict:
        return {
            "@type": "Person",
            "name": self.paper.drafting_context.get('authors', 'Worth, J. B.'),
            "givenName": "James B.",
            "familyName": "Worth",
            "email": "bworth@substrate.ai",
            "identifier": {
                "@type": "PropertyValue",
                "propertyID": "ORCID",
                "value": self.paper.drafting_context.get(
                    'orcid', '0009-0005-5000-9497'),
                "url": f"https://orcid.org/{self.paper.drafting_context.get('orcid', '0009-0005-5000-9497')}",
            },
            "affiliation": {
                "@type": "Organization",
                "name": "Substrate AI",
                "address": {
                    "@type": "PostalAddress",
                    "addressLocality": "Valencia",
                    "addressCountry": "ES",
                },
            },
        }

    def _publication_block(self) -> dict:
        target_journal = self.paper.target_journal
        journal_def = self.config.get_journal(target_journal)
        return {
            "@type": "Periodical",
            "name": journal_def.get('display_name', target_journal),
            "issn": journal_def.get('issn', ''),
            "publisher": journal_def.get('publisher', ''),
        }

    def _date_published(self) -> str:
        # Use today as a placeholder; real publication date set when paper accepted
        return datetime.now().date().isoformat()

    def _license_url(self) -> str:
        # CC-BY 4.0 is the standard for accompanying data and metadata
        return "https://creativecommons.org/licenses/by/4.0/"

    def _keywords(self) -> list:
        kws = self.paper.drafting_context.get('keywords', [])
        if isinstance(kws, str):
            kws = [k.strip() for k in kws.split(',') if k.strip()]
        # Add some defaults if empty
        if not kws:
            kws = ['plasma physics', 'particle-in-cell simulation',
                   'magnetic reconnection']
        return kws
