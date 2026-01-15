"""Data ingestion connectors for ADS.

Available connectors:
- OnetConnector: O*NET occupation/job role data
- OcwConnector: MIT OpenCourseWare course data
- MitCatalogConnector: MIT Course Catalog data
- MitOcwConnector: Enhanced OCW connector with manifest support
- UcbOutcomesConnector: UC Berkeley career outcomes (FDS) data
- AsuFdsConnector: ASU First Destination Survey schema
- AsuOutcomesConnector: ASU career outcomes data
- build_toy_artifacts: Built-in toy dataset for testing
"""
from ads_core.ingest.base import BaseConnector, ProvenanceInfo, create_provenance
from ads_core.ingest.toy_dataset import build_toy_artifacts
from ads_core.ingest.onet import OnetConnector, create_onet_fixture
from ads_core.ingest.ocw import OcwConnector, create_ocw_fixture
from ads_core.ingest.mit_catalog import MitCatalogConnector, create_catalog_fixture
from ads_core.ingest.mit_ocw import MitOcwConnector, create_mit_fixtures
from ads_core.ingest.ucb_outcomes import UcbOutcomesConnector, create_ucb_fixture
from ads_core.ingest.asu_fds import AsuFdsConnector, create_fds_fixture
from ads_core.ingest.asu_outcomes import AsuOutcomesConnector, create_outcomes_fixture

__all__ = [
    "BaseConnector",
    "ProvenanceInfo",
    "create_provenance",
    "build_toy_artifacts",
    "OnetConnector",
    "create_onet_fixture",
    "OcwConnector",
    "create_ocw_fixture",
    "MitCatalogConnector",
    "create_catalog_fixture",
    "MitOcwConnector",
    "create_mit_fixtures",
    "UcbOutcomesConnector",
    "create_ucb_fixture",
    "AsuFdsConnector",
    "create_fds_fixture",
    "AsuOutcomesConnector",
    "create_outcomes_fixture",
]
