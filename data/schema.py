"""
data/schema.py
==============
Single source of truth for the resume output schema.

This Pydantic model is used in THREE places:
  1. Dataset generation — every training example's "output" is validated here
  2. Inference post-processing — model output is validated/rejected here
  3. Evaluation — field-by-field diffing uses this stable shape

NEVER duplicate this schema in another file. Import from here.

Rule: Missing fields must be null / empty list, NEVER hallucinated.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, field_validator


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class EducationEntry(BaseModel):
    """One education block (degree + institution + dates)."""

    degree: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    model_config = {"extra": "ignore"}


class ExperienceEntry(BaseModel):
    """One work-experience block."""

    title: Optional[str] = None
    company: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
class ProjectEntry(BaseModel):
    """One project block (title/name, description, technologies/skills used, link)."""

    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = []
    link: Optional[str] = None

    model_config = {"extra": "ignore"}


# ---------------------------------------------------------------------------
# Root schema
# ---------------------------------------------------------------------------

class ResumeSchema(BaseModel):
    """
    Structured output for a parsed resume.

    Design decisions:
    - All scalar fields are Optional[str] → null when absent, not invented.
    - List fields default to [] → never null, never hallucinated items.
    - phone is a list because candidates sometimes list multiple numbers.
    - links captures GitHub, LinkedIn, portfolio URLs separately from email.
    - projects captures hands-on AI/engineering projects.
    """

    name: Optional[str] = None
    email: Optional[str] = None
    phone: List[str] = []
    location: Optional[str] = None
    summary: Optional[str] = None
    education: List[EducationEntry] = []
    experience: List[ExperienceEntry] = []
    projects: List[ProjectEntry] = []
    skills: List[str] = []
    certifications: List[str] = []
    links: List[str] = []

    model_config = {"extra": "ignore"}

    # ------------------------------------------------------------------
    # Validators — light normalization, not heavy transformation
    # ------------------------------------------------------------------

    @field_validator("name", "location", "summary", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Optional[str]) -> Optional[str]:
        """Strip leading/trailing whitespace from scalar strings."""
        if v is None:
            return None
        v = v.strip()
        return v if v else None  # empty string → None

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, v: Optional[str]) -> Optional[str]:
        """Lowercase and strip email; return None if empty."""
        if v is None:
            return None
        v = v.strip().lower()
        return v if v else None

    @field_validator("skills", "certifications", "links", "phone", mode="before")
    @classmethod
    def deduplicate_list(cls, v: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        if not v:
            return []
        seen = set()
        result = []
        for item in v:
            item = item.strip() if isinstance(item, str) else item
            if item and item not in seen:
                seen.add(item)
                result.append(item)
        return result


# ---------------------------------------------------------------------------
# Utility helpers used by dataset generation, evaluation, and inference
# ---------------------------------------------------------------------------

def parse_resume_output(raw: dict) -> ResumeSchema:
    """
    Parse a raw dict (e.g. from json.loads(model_output)) into a
    validated ResumeSchema.

    Raises:
        pydantic.ValidationError — if the dict does not conform to the schema.
    """
    return ResumeSchema.model_validate(raw)


def to_json(resume: ResumeSchema, *, indent: int = 2) -> str:
    """Serialize a ResumeSchema to a JSON string (for storage / output)."""
    return resume.model_dump_json(indent=indent)


def empty_resume() -> ResumeSchema:
    """Return a fully-null / fully-empty ResumeSchema (useful for testing)."""
    return ResumeSchema()


# ---------------------------------------------------------------------------
# Schema export — used in system prompt construction during training/inference
# ---------------------------------------------------------------------------

SCHEMA_DESCRIPTION = """
{
  "name":           "string or null",
  "email":          "string or null",
  "phone":          ["string"],
  "location":       "string or null",
  "summary":        "string or null",
  "education": [
    {
      "degree":       "string or null",
      "institution":  "string or null",
      "start_date":   "string or null",
      "end_date":     "string or null"
    }
  ],
  "experience": [
    {
      "title":        "string or null",
      "company":      "string or null",
      "start_date":   "string or null",
      "end_date":     "string or null",
      "description":  "string or null"
    }
  ],
  "projects": [
    {
      "name":         "string or null",
      "description":  "string or null",
      "technologies": ["string"],
      "link":         "string or null"
    }
  ],
  "skills":          ["string"],
  "certifications":  ["string"],
  "links":           ["string"]
}
""".strip()

SYSTEM_PROMPT = (
    "You are a resume parser. Extract information from the resume text and "
    "return it as a single valid JSON object matching this schema exactly:\n\n"
    + SCHEMA_DESCRIPTION
    + "\n\nRules:\n"
    "- Return ONLY the JSON object. No explanation, no markdown, no code fences.\n"
    "- If a field is missing from the resume, set it to null (scalars) or [] (lists).\n"
    "- Never invent information that is not present in the resume.\n"
    "- Do not add extra keys beyond those in the schema."
)


# ---------------------------------------------------------------------------
# Quick self-test (python data/schema.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    sample = {
        "name": "  John Doe  ",
        "email": "JOHN@EXAMPLE.COM",
        "phone": ["+91-9876543210", "+91-9876543210"],  # duplicate
        "location": "Bangalore, India",
        "summary": None,
        "education": [
            {"degree": "B.Tech Computer Science", "institution": "IIT Delhi",
             "start_date": "2018", "end_date": "2022"}
        ],
        "experience": [
            {"title": "Software Engineer", "company": "Acme Corp",
             "start_date": "Jun 2022", "end_date": "Present",
             "description": "Built REST APIs with FastAPI."}
        ],
        "skills": ["Python", "FastAPI", "SQL", "Python"],  # duplicate
        "certifications": ["AWS Certified Developer"],
        "links": ["https://github.com/johndoe"]
    }

    parsed = parse_resume_output(sample)
    print("Schema self-test passed.")
    print(json.loads(to_json(parsed)))
    print()
    print("SYSTEM_PROMPT preview:")
    print(SYSTEM_PROMPT[:300], "...")
