"""Fixed paper-reproduction inventory, scientific hashes and citation policy.

Changing a manifest cannot authorize different scientific bytes. This policy is
reviewed alongside source changes; it does not generate or modify study data.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
from typing import Callable

PAPER_HASHES = {'reproduction/LICENSE': '1cd92ec7fce29aeef14ef7a12c71c70042cc2f87c71c8f9d68b4aa8bbfe0e0bb',
 'reproduction/LICENSE-RESULTS': '39c82e79f0e67f5034b00b62225d836d7293780aceda281139ab82f69e21b716',
 'reproduction/README.md': '204b2831d6a7b51fe5a5ad57e543c46f3d1bdaf0280f3f91fe2af63bb1f342cf',
 'reproduction/SHA256SUMS': '2b56a10887aebe950169587a844a646206e9cf72f8458a826ec6cb34ae9f392b',
 'reproduction/THIRD_PARTY_DATA.md': '96231dc69cce7d65bd40fa776feaf6a372f5b0b0a6335b5447824cf6636f5450',
 'reproduction/aggregate-data/README.md': 'b2f62d2312c3f8b2cd069f6aa25607b67b062a84322fe96335c996c83299c2cc',
 'reproduction/aggregate-data/focused-all-decoder-metrics.csv': 'ab65cfef0547900f0424f19caaac38cd732362efb9d77a439a7b9524c15fbcf0',
 'reproduction/aggregate-data/focused-all-primary-individual.csv': '72263792ba3d0ee4afe7760eb7610d75125f6cf40e25773cedf4802876eb16ea',
 'reproduction/aggregate-data/focused-cohort-descriptive.csv': '5bfcc8a83038543f44cd0c03ec7a9360f43d971fb1e48d62c677af47230f5431',
 'reproduction/aggregate-data/focused-jpeg-full-curves.csv': '699f121b9f8e8ce56d5df52039b713f50b32a17100674335f6b07ee968c83080',
 'reproduction/aggregate-data/focused-jpeg-selected-points.csv': '9486402bdc289aad7b9cd75b7fdc225af9e0ae9f0ae9a99ddd4106fbc559b095',
 'reproduction/aggregate-data/focused-matched-paired-descriptive.csv': '7ba078f5ff864c916aba1d42e2b301047d66dbf6bd89c761c5ab9e93a5a3cfba',
 'reproduction/aggregate-data/focused-matched-paired-individual.csv': 'b23e480cb38f40f38224e0bc64cc6ae5e4739ae5c09e60b7245ec32accd09088',
 'reproduction/aggregate-data/followup-decoder-contrast.csv': '94cd522364ef5ebdf97ef5ba150b640cffbbfde9b52eb941ead4335e2fa6d740',
 'reproduction/aggregate-data/followup-disco-candidates.csv': 'c0e51a30d5731eafb0a46e7047f7568a9d967dfe44bb57e4cb1f83d3b4a9ca60',
 'reproduction/aggregate-data/followup-test-utility.csv': '34bca7c1580bb64177ef6d2c5e47c7b42b2d5eeda4975db9159f49df30c676ab',
 'reproduction/aggregate-data/followup-training-history.csv': 'f85a80876b4e9089628aa17b4bc6b0992dfb4a42247712f5a42ca78089877ec7',
 'reproduction/aggregate-data/historical-secondary-directions.csv': '333cb3cec9727375c00797cdcdf86efde709d017b168c41c2b34b13a60be6ee6',
 'reproduction/aggregate-data/historical-secondary-group.csv': '7603abb89ca35b171d01962b453d928e4a95e0d022ba2eb380fddf87b2d95f6e',
 'reproduction/aggregate-data/historical-secondary-individual.csv': '49af23cb02831950256d010af2ada3cc1ec40b96443eb30b7d60bdfe12b34048',
 'reproduction/aggregate-data/historical-secondary-paired-individual.csv': '2ab7d5192193a9cab8aa767717e1d08b9a8e54e3f62ff64147dfe4377b1efafe',
 'reproduction/aggregate-data/historical-secondary-paired.csv': 'd4a2bd59bb191d96acbe758c2e790f1a84733b2569b042786d1c8d7bd005e48e',
 'reproduction/environment/focused-recorded.json': '0796462ae4fb6dd55dc5748bd5ac6c6f73f5164088b642ea92c76b38e57b4a77',
 'reproduction/environment/historical-reconstruction.json': '13647ecccbec166491affc6c606cc27b614722c2277e09d27015f15b4bd56978',
 'reproduction/environment/historical-workstation.json': 'd23012ef7339d38a54ed34e4bd72c230e372bddf4bd0a5ec63273d0d8b915325',
 'reproduction/migration-map.json': '5a5bcfcd0dad82ea7f6d9817e8a76aba2b8571d13244b4b1ecf599baf3dbe379',
 'reproduction/provenance.json': '5bdc66aed7d42a34b019981aea3290be69082e2e09c29138706154ee67706267',
 'reproduction/replay_aggregates.py': '57d0a042b52a0112e5d4ede6d8f4ce56888c3d263596a449f87dcafb4a119fdb',
 'reproduction/requirements-scientific.txt': '4bd08a973bce469f2457ecba2d674c961471904da296b1bee039e2ea68318af0',
 'reproduction/requirements-supporting.txt': 'ab36390dd4e93c9c4bd895fe5740e995b0f0da68f2d07eb9a64739216919d37f',
 'reproduction/supporting/DATA_DICTIONARY.md': 'c6ab93a52c632fcfbc8d6183f9ce87c9c63fc344d2ca9960d79d3adef8b25cc3',
 'reproduction/supporting/README.md': 'a5dec0fcf137edcdf965bdefd4a5e4f6df772e5ec4056b427d7e90a52b35fe27',
 'reproduction/supporting/SHA256SUMS': '157f5e5c54e392cc9099dadafc8bf02ccd26f7c4fc7d59cac58596de2e3d971b',
 'reproduction/supporting/analysis/compute_analysis.json': '262741aee3a21052f7c3d7e901f4c5943e0ac26b09ad41d39679bb9e9a4a67f1',
 'reproduction/supporting/analysis/compute_claim_audit.json': '001a6e433d6d3a1749224bfe8533ee1313690e2e78c31f0f2a7c62f36fcd3bd6',
 'reproduction/supporting/analysis_contract.json': '45124d8e98488df044c8eb672dd9e19a42e04a7320106b4319de9ec47ad1d277',
 'reproduction/supporting/diagnostics/known-offset-diagnostic.csv': '70862589f7e788891ad8d810cbef8d54757108b6ac9169eb0aee447f997a937b',
 'reproduction/supporting/diagnostics/known-offset-diagnostic.json': '3679eb943b4fb948eb64f0941b1125c2609e4ffd2db9f70958ec1a95b3096089',
 'reproduction/supporting/evidence/benchmark_budget_attacker_rows.csv': '0523fb0265b21f7850742645d728441c21c63f847c941f63cfb41a469b79e49a',
 'reproduction/supporting/expected/attacker_restart_summaries.csv': '89cd56dbb25d57afac00e44012e1c87ecfd32e3c065313a24355c0379e890618',
 'reproduction/supporting/expected/benchmark_budget_paired_effects.csv': '8312b09c91612e77b1c35fd713e47ffecf85dfc5a122461ff0aa4ac26c7ce0dc',
 'reproduction/supporting/expected/budget_method_summaries.csv': 'd2c3fbb4ea925d063e04c5131aa571d8f603298af298e06d0bfe7b3788a37d34',
 'reproduction/supporting/expected/compute_group_summary.csv': '402f17a7ff585944f39f13011e0fecb9b79853cd45af0b04208edbbcf2c90681',
 'reproduction/supporting/expected/compute_interface_effects.csv': '3f77831cb296e218030e4c46bbc02cbd7ac4249e2b834e4e43b7773ab5b79164',
 'reproduction/supporting/expected/compute_paired_effects.csv': '1ab4edfab53bfefc40beb0022b25a31249a164cc4ca2f62e2950be3240b8c525',
 'reproduction/supporting/expected/figure_signatures.json': '9a85f375f44a2490ffe1bdc2fde892d476d15c12e723e926097682a1fd700ed9',
 'reproduction/supporting/expected/figure_sources.json': '06fb5ca3aff79413898d3a3daf9540e38ef8a0dbaa250a9add682c572eadd356',
 'reproduction/supporting/expected/laplace-single-release-statistics.csv': '33cc3084a7467577e9c541eb72d3107210fefb7afa44fde628a7316b4e28349c',
 'reproduction/supporting/expected/supplementary-statistics.csv': 'a387a651be20f087faf91f2f7b6d5a89552e211604b8741559243229e18908b7',
 'reproduction/supporting/inputs/attacker_restart_rows.csv': 'b1da933c28665b186b27f081401c9a5445489acb4f6224d8c620c16b07670465',
 'reproduction/supporting/inputs/benchmark_budget_conditions.csv': '6d5961a3bae709f1d98e1630bd1e1f90736ae9c06f93199c58685c35f6a8e9c7',
 'reproduction/supporting/inputs/compute_condition_metrics.csv': 'beab01892dd8505e889b4f387ebd395efb6495345b15c171fbb6da9184c74f25',
 'reproduction/supporting/inputs/laplace_draw_metrics.csv': 'b60fc38b2cf5566c3f78975e1c81d85100ce8e054b3fbf60b3390b48e246b023',
 'reproduction/supporting/inputs/laplace_run_metrics.csv': 'df9c810a8c476f5920c6615ccdc4a92e0706b28c933db98f4be417a1f103b395',
 'reproduction/supporting/provenance.json': 'cc457078303264c5f2ec8934b32dd5624281d19a5adecc4a7f628e94382f2e75',
 'reproduction/supporting/schemas/attacker_restart_rows.schema.json': '9a36bdd36307a32610a4980dfba9213d0e75ded5338771c1dda2fa6e12ee6dce',
 'reproduction/supporting/schemas/attacker_restart_summaries.schema.json': 'ce6985e37b110c0b9265bc34ce815bad8f4edd71b85d61a543455248dd7d81bf',
 'reproduction/supporting/schemas/benchmark_budget_attacker_rows.schema.json': '32b7203b5f3d606095c0a0f689ad971c86533aefcf835900c4a4bd06583babc6',
 'reproduction/supporting/schemas/benchmark_budget_conditions.schema.json': '2ab0feb1e3d5c3a6dc02fd2f8146671ba51b5ef655d365edb52eb703e0bb1471',
 'reproduction/supporting/schemas/benchmark_budget_paired_effects.schema.json': '8e6a8338fcbdd887b9935ce4ceb282c05a29dd50eeb21803e8ae2c576d74c715',
 'reproduction/supporting/schemas/benchmark_v1_tables.schema.json': '900ce37be4ff8df523ffa4a97888a5accd7098a297492dbb8b09e1ad360c08f3',
 'reproduction/supporting/schemas/eng_compute_v1.schema.json': '5e2603047b488be03c800de0c897ce0ac0134c44d8e546d684e42b2639f498a1',
 'reproduction/upstream/LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
 'reproduction/upstream/disco.json': '055b112bf23876b949e670ebe99ed04505835e8e1ae20716926fd13561e12c58',
 'reproduction/upstream/disco.py': '1a4f32e9d6ed438acb04652b230ba5c8877ef28955e8ff499967ae36f8a79153'}

PAPER_LICENSE_PATHS = ('reproduction/aggregate-data/focused-all-decoder-metrics.csv',
 'reproduction/aggregate-data/focused-all-primary-individual.csv',
 'reproduction/aggregate-data/focused-cohort-descriptive.csv',
 'reproduction/aggregate-data/focused-jpeg-full-curves.csv',
 'reproduction/aggregate-data/focused-jpeg-selected-points.csv',
 'reproduction/aggregate-data/focused-matched-paired-descriptive.csv',
 'reproduction/aggregate-data/focused-matched-paired-individual.csv',
 'reproduction/aggregate-data/followup-decoder-contrast.csv',
 'reproduction/aggregate-data/followup-disco-candidates.csv',
 'reproduction/aggregate-data/followup-test-utility.csv',
 'reproduction/aggregate-data/followup-training-history.csv',
 'reproduction/aggregate-data/historical-secondary-directions.csv',
 'reproduction/aggregate-data/historical-secondary-group.csv',
 'reproduction/aggregate-data/historical-secondary-individual.csv',
 'reproduction/aggregate-data/historical-secondary-paired-individual.csv',
 'reproduction/aggregate-data/historical-secondary-paired.csv')

PAPER_METADATA = {'aggregate_replay': {'absolute_tolerance': 1e-12,
                      'descriptive_rows': 1140,
                      'numerical_comparisons': 8227,
                      'python': '>=3.10',
                      'third_party_dependencies': []},
 'file_count': 65,
 'licenses': {'authored_code_and_documentation': 'MIT',
              'contributor_owned_numeric_tables': 'CC-BY-4.0',
              'pinned_DISCO_source': 'Apache-2.0'},
 'manuscript_status': 'under_review',
 'path': 'reproduction',
 'source_archive_sha256': '88f4516febd32092c443d08068c97861b01802eff69399109ea15e016aa88865',
 'table_rows': {'reproduction/aggregate-data/focused-all-decoder-metrics.csv': 240,
                'reproduction/aggregate-data/focused-all-primary-individual.csv': 120,
                'reproduction/aggregate-data/focused-cohort-descriptive.csv': 520,
                'reproduction/aggregate-data/focused-jpeg-full-curves.csv': 48,
                'reproduction/aggregate-data/focused-jpeg-selected-points.csv': 4,
                'reproduction/aggregate-data/focused-matched-paired-descriptive.csv': 620,
                'reproduction/aggregate-data/focused-matched-paired-individual.csv': 900,
                'reproduction/aggregate-data/followup-decoder-contrast.csv': 4,
                'reproduction/aggregate-data/followup-disco-candidates.csv': 12,
                'reproduction/aggregate-data/followup-test-utility.csv': 90,
                'reproduction/aggregate-data/followup-training-history.csv': 360,
                'reproduction/aggregate-data/historical-secondary-directions.csv': 96,
                'reproduction/aggregate-data/historical-secondary-group.csv': 128,
                'reproduction/aggregate-data/historical-secondary-individual.csv': 160,
                'reproduction/aggregate-data/historical-secondary-paired-individual.csv': 120,
                'reproduction/aggregate-data/historical-secondary-paired.csv': 96},
 'title': 'Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker '
          'Configuration'}

PUBLIC_ADDITIONS = frozenset({'config/benchmark_extension_v1.json',
           'config/focused-recipe.json',
           'docs/MIGRATION.md',
           'docs/PAPER_MAP.md',
           'docs/REPRODUCIBILITY.md',
           'reproduction/LICENSE',
           'reproduction/LICENSE-RESULTS',
           'reproduction/README.md',
           'reproduction/SHA256SUMS',
           'reproduction/THIRD_PARTY_DATA.md',
           'reproduction/aggregate-data/README.md',
           'reproduction/aggregate-data/focused-all-decoder-metrics.csv',
           'reproduction/aggregate-data/focused-all-primary-individual.csv',
           'reproduction/aggregate-data/focused-cohort-descriptive.csv',
           'reproduction/aggregate-data/focused-jpeg-full-curves.csv',
           'reproduction/aggregate-data/focused-jpeg-selected-points.csv',
           'reproduction/aggregate-data/focused-matched-paired-descriptive.csv',
           'reproduction/aggregate-data/focused-matched-paired-individual.csv',
           'reproduction/aggregate-data/followup-decoder-contrast.csv',
           'reproduction/aggregate-data/followup-disco-candidates.csv',
           'reproduction/aggregate-data/followup-test-utility.csv',
           'reproduction/aggregate-data/followup-training-history.csv',
           'reproduction/aggregate-data/historical-secondary-directions.csv',
           'reproduction/aggregate-data/historical-secondary-group.csv',
           'reproduction/aggregate-data/historical-secondary-individual.csv',
           'reproduction/aggregate-data/historical-secondary-paired-individual.csv',
           'reproduction/aggregate-data/historical-secondary-paired.csv',
           'reproduction/environment/focused-recorded.json',
           'reproduction/environment/historical-reconstruction.json',
           'reproduction/environment/historical-workstation.json',
           'reproduction/migration-map.json',
           'reproduction/provenance.json',
           'reproduction/replay_aggregates.py',
           'reproduction/requirements-scientific.txt',
           'reproduction/requirements-supporting.txt',
           'reproduction/supporting/DATA_DICTIONARY.md',
           'reproduction/supporting/README.md',
           'reproduction/supporting/SHA256SUMS',
           'reproduction/supporting/analysis/compute_analysis.json',
           'reproduction/supporting/analysis/compute_claim_audit.json',
           'reproduction/supporting/analysis_contract.json',
           'reproduction/supporting/diagnostics/known-offset-diagnostic.csv',
           'reproduction/supporting/diagnostics/known-offset-diagnostic.json',
           'reproduction/supporting/evidence/benchmark_budget_attacker_rows.csv',
           'reproduction/supporting/expected/attacker_restart_summaries.csv',
           'reproduction/supporting/expected/benchmark_budget_paired_effects.csv',
           'reproduction/supporting/expected/budget_method_summaries.csv',
           'reproduction/supporting/expected/compute_group_summary.csv',
           'reproduction/supporting/expected/compute_interface_effects.csv',
           'reproduction/supporting/expected/compute_paired_effects.csv',
           'reproduction/supporting/expected/figure_signatures.json',
           'reproduction/supporting/expected/figure_sources.json',
           'reproduction/supporting/expected/laplace-single-release-statistics.csv',
           'reproduction/supporting/expected/supplementary-statistics.csv',
           'reproduction/supporting/inputs/attacker_restart_rows.csv',
           'reproduction/supporting/inputs/benchmark_budget_conditions.csv',
           'reproduction/supporting/inputs/compute_condition_metrics.csv',
           'reproduction/supporting/inputs/laplace_draw_metrics.csv',
           'reproduction/supporting/inputs/laplace_run_metrics.csv',
           'reproduction/supporting/provenance.json',
           'reproduction/supporting/schemas/attacker_restart_rows.schema.json',
           'reproduction/supporting/schemas/attacker_restart_summaries.schema.json',
           'reproduction/supporting/schemas/benchmark_budget_attacker_rows.schema.json',
           'reproduction/supporting/schemas/benchmark_budget_conditions.schema.json',
           'reproduction/supporting/schemas/benchmark_budget_paired_effects.schema.json',
           'reproduction/supporting/schemas/benchmark_v1_tables.schema.json',
           'reproduction/supporting/schemas/eng_compute_v1.schema.json',
           'reproduction/upstream/LICENSE',
           'reproduction/upstream/disco.json',
           'reproduction/upstream/disco.py',
           'scripts/paper_repository_policy.py',
           'scripts/reproduce.py',
           'scripts/run_study.py',
           'tests/test_paper_reproduction.py',
           'tests/test_reproduction_cli.py'})

EXPECTED_CITATION_CFF = 'cff-version: 1.2.0\nmessage: "If you use this work, please cite the paper under review and the software."\ntype: software\ntitle: "Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker Configuration"\nauthors:\n  - family-names: "Jang"\n    given-names: "Sooyoung"\nrepository-code: "https://github.com/pz1004/adaptive-reconstruction-split-computing"\nlicense: MIT\npreferred-citation:\n  type: unpublished\n  title: "Reconstruction Exposure and Task Utility in Split Computing: Sensitivity to Attacker Configuration"\n  authors:\n    - family-names: "Jang"\n      given-names: "Sooyoung"\n    - family-names: "Woo"\n      given-names: "Sungpil"\n    - family-names: "Lee"\n      given-names: "Ahyun"\n  year: 2026\n  notes: "Manuscript under review."\n'

def verify_paper_reproduction(read_bytes: Callable[[str], bytes]) -> dict:
    """Check fixed payloads, checksum membership, CSV rows and source bindings."""
    for relative, expected in PAPER_HASHES.items():
        if hashlib.sha256(read_bytes(relative)).hexdigest() != expected:
            raise RuntimeError(f"Paper reproduction hash mismatch: {relative}")
    rows = {}
    for line in read_bytes("reproduction/SHA256SUMS").decode().splitlines():
        expected, relative = line.split("  ", 1)
        if relative in rows:
            raise RuntimeError("Duplicate paper checksum entry")
        rows[relative] = expected
    expected_rows = {
        name.removeprefix("reproduction/"): digest for name, digest in PAPER_HASHES.items()
        if name != "reproduction/SHA256SUMS"
    }
    if rows != expected_rows:
        raise RuntimeError("Paper checksum inventory differs from fixed policy")
    for name, count in PAPER_METADATA["table_rows"].items():
        parsed = list(csv.DictReader(io.StringIO(read_bytes(name).decode())))
        if len(parsed) != count or any(None in row or None in row.values() for row in parsed):
            raise RuntimeError(f"Paper CSV shape mismatch: {name}")
    provenance = json.loads(read_bytes("reproduction/provenance.json"))
    for name, record in provenance["source_files"].items():
        # Sanitized records bind the source input separately; exported bytes are
        # independently fixed above, including the two records without sha256.
        if "sha256" in record and hashlib.sha256(read_bytes(name)).hexdigest() != record["sha256"]:
            raise RuntimeError(f"Paper source binding differs: {name}")
    result_license = read_bytes("LICENSE-RESULTS").decode()
    for name in PAPER_LICENSE_PATHS:
        if result_license.splitlines().count(name) != 1:
            raise RuntimeError(f"Paper table missing from exact license scope: {name}")
    local_scope = read_bytes("reproduction/LICENSE-RESULTS").decode().splitlines()
    if {line for line in local_scope if line.startswith("aggregate-data/")} != {
        name.removeprefix("reproduction/") for name in PAPER_LICENSE_PATHS
    }:
        raise RuntimeError("Paper-local license scope differs from fixed policy")
    return {"passed": True, "files_verified": len(PAPER_HASHES),
            "tables_verified": len(PAPER_METADATA["table_rows"]),
            "table_rows": sum(PAPER_METADATA["table_rows"].values())}
