"""
config.py — Load and validate program_config.yaml.

The orchestrator is YAML-driven. This module:
  - Loads program_config.yaml
  - Validates structural requirements
  - Provides typed access via PaperConfig and ProgramConfig dataclasses
  - Resolves paths (program root, paper repos, runs)

Usage:
    from orchestrator.config import load_config
    cfg = load_config('program_config.yaml')
    paper = cfg.get_paper('A1')
    print(paper.title)
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. Install with: pip install pyyaml", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised for invalid or incomplete program_config.yaml."""
    pass


def _require(d: dict, key: str, context: str) -> Any:
    """Fetch a required key, raising ConfigError with context if missing."""
    if key not in d:
        raise ConfigError(f"Missing required key '{key}' in {context}")
    return d[key]


# ---------------------------------------------------------------------------
# Per-paper config wrapper
# ---------------------------------------------------------------------------

@dataclass
class PaperConfig:
    """Paper definition extracted from YAML, with typed accessors."""
    id: str                           # e.g. "A1"
    paper_num: Optional[int]          # legacy paper number, may be None
    tag: str                          # e.g. "p1_static_p11b"
    label: str
    title: str
    repo_name: str
    target_journal: str
    status: str
    raw: dict                         # full YAML dict for handlers

    @property
    def deliverables(self) -> list:
        """List of deliverable type names (strings) or override dicts."""
        return self.raw.get('deliverables', [])

    @property
    def has_deliverables(self) -> bool:
        return bool(self.deliverables)

    @property
    def deliverable_names(self) -> list[str]:
        """Just the type names, even if some entries are override dicts."""
        out = []
        for d in self.deliverables:
            if isinstance(d, str):
                out.append(d)
            elif isinstance(d, dict) and 'type' in d:
                out.append(d['type'])
        return out

    @property
    def simulation(self) -> dict:
        return self.raw.get('simulation', {})

    @property
    def analysis(self) -> dict:
        return self.raw.get('analysis', {})

    @property
    def drafting_context(self) -> dict:
        return self.raw.get('drafting_context', {})

    @property
    def citation_strategy(self) -> dict:
        return self.raw.get('citation_strategy', {})

    @property
    def internal_patent_strategy(self) -> dict:
        return self.raw.get('internal_patent_strategy', {})

    @property
    def is_analysis_only(self) -> bool:
        """Paper 7 pattern — no simulation, just cross-paper analysis."""
        return bool(self.simulation.get('analysis_only'))

    @property
    def requires_attorney_review(self) -> bool:
        return bool(self.internal_patent_strategy.get(
            'requires_attorney_review_before_public', False))


# ---------------------------------------------------------------------------
# Program-level config
# ---------------------------------------------------------------------------

@dataclass
class ProgramConfig:
    """Top-level wrapper around the parsed YAML."""
    raw: dict
    config_path: Path

    _papers_by_id: dict = field(default_factory=dict, init=False)
    _papers_by_num: dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        for paper_def in self.raw.get('papers', []):
            paper = PaperConfig(
                id=_require(paper_def, 'id', 'paper'),
                paper_num=paper_def.get('paper_num'),
                tag=_require(paper_def, 'tag',
                             f"paper {paper_def.get('id', '?')}"),
                label=paper_def.get('label', ''),
                title=paper_def.get('title', ''),
                repo_name=paper_def.get('repo_name', ''),
                target_journal=paper_def.get('target_journal', ''),
                status=paper_def.get('status', 'draft'),
                raw=paper_def,
            )
            if paper.id in self._papers_by_id:
                raise ConfigError(f"Duplicate paper id: {paper.id}")
            self._papers_by_id[paper.id] = paper
            if paper.paper_num is not None:
                self._papers_by_num.setdefault(paper.paper_num, []).append(paper)

    # ---- Top-level config sections ----

    @property
    def program(self) -> dict:
        return self.raw.get('program', {})

    @property
    def name(self) -> str:
        return self.program.get('name', 'Research Programme')

    @property
    def org(self) -> str:
        return self.program.get('org', 'unknown')

    @property
    def default_max_parallel(self) -> int:
        return int(self.program.get('default_max_parallel', 2))

    @property
    def default_mpi_ranks(self) -> int:
        return int(self.program.get('default_mpi_ranks', 8))

    @property
    def runs_root(self) -> str:
        return self.program.get('paths', {}).get('runs_root', './runs')

    @property
    def shared_dir(self) -> Path:
        s = self.program.get('paths', {}).get('shared', './shared')
        return Path(s).expanduser()

    @property
    def program_root(self) -> Path:
        s = self.program.get('paths', {}).get('program_root', '.')
        return Path(s).expanduser()

    @property
    def drafting(self) -> dict:
        return self.program.get('drafting', {})

    @property
    def default_model(self) -> str:
        return self.drafting.get('default_model', 'claude-opus-4-6')

    @property
    def api_key_env_var(self) -> str:
        return self.drafting.get('api_key_env_var', 'ANTHROPIC_API_KEY')

    @property
    def preflight(self) -> dict:
        return self.drafting.get('preflight', {})

    @property
    def visibility(self) -> dict:
        return self.program.get('visibility', {})

    @property
    def citation_defaults(self) -> dict:
        return self.program.get('citation', {})

    # ---- Lookup tables ----

    @property
    def journal_profiles(self) -> dict:
        return self.raw.get('journal_profiles', {})

    def get_journal(self, name: str) -> dict:
        return self.journal_profiles.get(name, {})

    @property
    def figure_registry(self) -> dict:
        return self.raw.get('figure_registry', {})

    def get_figure_type(self, name: str) -> dict:
        return self.figure_registry.get(name, {})

    @property
    def deliverable_types(self) -> dict:
        return self.raw.get('deliverable_types', {})

    def get_deliverable_type(self, name: str) -> dict:
        return self.deliverable_types.get(name, {})

    # ---- Paper access ----

    @property
    def papers(self) -> list:
        return list(self._papers_by_id.values())

    def get_paper(self, paper_id: str) -> Optional[PaperConfig]:
        return self._papers_by_id.get(paper_id)

    def papers_by_num(self, paper_num: int) -> list:
        return list(self._papers_by_num.get(paper_num, []))

    def filter_papers(
        self,
        paper_ids: Optional[list] = None,
        paper_nums: Optional[list] = None,
    ) -> list:
        """Return papers matching either id list or num list (or all)."""
        if paper_ids:
            return [p for p in self.papers if p.id in paper_ids]
        if paper_nums:
            num_set = set(paper_nums)
            return [p for p in self.papers if p.paper_num in num_set]
        return self.papers


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path) -> ProgramConfig:
    """
    Load and validate program_config.yaml. Returns a ProgramConfig.

    Raises:
        FileNotFoundError if path does not exist
        ConfigError if structure invalid
        yaml.YAMLError if YAML malformed
    """
    cp = Path(path).expanduser().resolve()
    if not cp.exists():
        raise FileNotFoundError(f"Config file not found: {cp}")
    with open(cp) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ConfigError(f"Top-level YAML must be a mapping, got {type(raw).__name__}")
    if 'papers' not in raw:
        raise ConfigError("Missing 'papers' section in YAML")
    return ProgramConfig(raw=raw, config_path=cp)


# ---------------------------------------------------------------------------
# Self-test when run as a script
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Validate program_config.yaml")
    parser.add_argument('config', nargs='?', default='program_config.yaml')
    args = parser.parse_args()
    try:
        cfg = load_config(args.config)
        print(f"✓ Loaded: {cfg.config_path}")
        print(f"  Programme: {cfg.name}")
        print(f"  Org:       {cfg.org}")
        print(f"  Papers:    {len(cfg.papers)}")
        print(f"  Default model: {cfg.default_model}")
        print(f"  Deliverable types: {len(cfg.deliverable_types)}")
        print(f"  Journal profiles:  {len(cfg.journal_profiles)}")
        print(f"  Figure registry:   {len(cfg.figure_registry)}")
        for p in cfg.papers:
            n_deliv = len(p.deliverables)
            n_sub = len(p.simulation.get('sub_jobs', []))
            print(f"    {p.id:5s} {p.tag:35s}  deliverables={n_deliv:2d}  sub_jobs={n_sub:2d}")
    except (ConfigError, FileNotFoundError) as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)
