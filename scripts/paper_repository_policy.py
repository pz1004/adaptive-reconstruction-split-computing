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
 'reproduction/README.md': '55a323ae78afa5ae4e2dd31e48f465ce942d4fe4bfbe40248d674bf4728f8689',
 'reproduction/SHA256SUMS': '1ad71ff21f65ad8c483c07f9dac94b6865bf16e10db29837d1098c0f3de0f95a',
 'reproduction/THIRD_PARTY_DATA.md': 'cde87fe3e8513ed685d40828f59e425fa5aec0f8fc7f6ff358af346b7278b364',
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
 'reproduction/environment/historical-protocol.json': 'c04be5a960f57c7c62cd4434cad94208f28b37aa425e945de9f564a2424186af',
 'reproduction/environment/historical-reconstruction.json': '13647ecccbec166491affc6c606cc27b614722c2277e09d27015f15b4bd56978',
 'reproduction/environment/historical-workstation.json': 'd23012ef7339d38a54ed34e4bd72c230e372bddf4bd0a5ec63273d0d8b915325',
 'reproduction/provenance.json': '2a58ff25ec8505264f217efd2a5c4e2c5d3361bff7f459d0b9363853b0cc14b4',
 'reproduction/replay_aggregates.py': 'a0a62ead366a614040fd84a36303622a8b262172a3cae3248876109653ad4c39',
 'reproduction/requirements-figures.txt': 'afe6f28dd495d8b5c68e828885b7e2f927234caa139be8809d9851e63786c470',
 'reproduction/requirements-scientific.txt': 'b18a3697789fbc7af4a1639ec9f662a8ccf2c4fcaf14b213b139c7f4cfe7578e',
 'reproduction/requirements.txt': '23a62d0fd89f8e506bddc7f67281040360c24d15457c5b6e685c45a5aa7f8c5a',
 'reproduction/scientific/config/benchmark_extension_v1.json': '76b5f794fec992f7da74577639789bddc3036401c21fc49e3a363bbabd837c55',
 'reproduction/scientific/config/focused-recipe.json': 'cd81753c05d4b6786d4b8e99cba5b1b74644024608fd7df2decd7f4ab8da5be8',
 'reproduction/scientific/config/implementation_protocol_v2.json': '4d80ecc39d0a9f09edad47e67912477623bc642d69c1948f8631f0e3a1d513ab',
 'reproduction/scientific/config/study_contract.json': 'e43780aa31661ba797d1832151cc5363a3933f90edc8f7e3375ac39e11bb9ef0',
 'reproduction/scientific/implementation/admission.py': '570bf557f407309d368bb83779c8d20dddcb3ddbef26e1fec524acf3b0a679fa',
 'reproduction/scientific/implementation/analyze_focused_results.py': 'c5d07a8885d768429aa24abf2f8cabc53f5c4d48dea66d78abb3f4fdf00e1268',
 'reproduction/scientific/implementation/components.py': '6d5202f3628e07f917234d17aa417eccc30b54ba71dc684d377270538b83eb6f',
 'reproduction/scientific/implementation/run_matrix.py': '5864566d1c89f8c818042af9f55fb0f624b8dec35984b274e16504068a49410f',
 'reproduction/scientific/implementation/science.py': '34090cd1b930168888ef11b83c2b743952c7456b85134395737f834a22e46099',
 'reproduction/scientific/implementation/supervise.py': '71956884699d957e20052054c8cfef1d26f6039710a0ace5c650a0a478c3a72b',
 'reproduction/scientific/implementation/training.py': 'cfc728c782667263f727b99c89927c56bc6f428d2106654829409d21b7ba0157',
 'reproduction/scientific/src/analysis.py': '48a62474afc81d7872d6a078bd3778501a9b39c9c77db44b78cfa9ced40ba617',
 'reproduction/scientific/src/benchmark_v1.py': '4fb9d158863a8b885296436ac6119f6db523ea6b695016cdabb2b215002d83e5',
 'reproduction/scientific/src/budget_result_audit.py': '14203c65b475de4bbbc97ed4a2e487d1e927c24dd7496bd4884472d9a21a48d6',
 'reproduction/scientific/src/checkpointing.py': '98559812bb3be3c84a494837a47301c4c5f07b5555aeb642cafebfcfc06ff289',
 'reproduction/scientific/src/dataset.py': 'a83e7da3b0d03ba7d8c807e9a6409bc9d59430ec3b990aab3b35ede53257f18b',
 'reproduction/scientific/src/defenses.py': 'a1ef0ad294375ba54c72c873336ba9919b9f922d166cdccef6433bf59b0c2b6d',
 'reproduction/scientific/src/eng_compute.py': 'cfe630d050055766b997c6f73e4bbf45f696ee85672d06957dce7bda405eace0',
 'reproduction/scientific/src/eval.py': 'cd6d308057b5ed7fc1a4a9cd67ff9ca83f4f0a11e35c78c03cb9f1a4528896a8',
 'reproduction/scientific/src/experiment.py': '0c4484c11f2c50794153d4c05017480dd3394733858759d3e713ec60c3c41dd0',
 'reproduction/scientific/src/implementation_provenance.py': 'a2f4e993fbcc22d41f176aaaf59446cd1b6d4bcbf7028486736ae3e08f9787ae',
 'reproduction/scientific/src/manifest.py': 'f3faba99216cf65c5fa499b21667b5ea7669273b16f351c4573ad0e3cea27e97',
 'reproduction/scientific/src/metrics.py': '2e675abf1702f0440b18594532975a8fee0762d9f29d6022c3902065b42b587d',
 'reproduction/scientific/src/models.py': '2e4e4897dcb5fcc847d78dc0cabe695bf87617d8e502cb4b3618f95c82373a4f',
 'reproduction/scientific/src/pipeline.py': '034408a9467e74565ba6dc5c117c2f1d5a4904578026ed45e3f02ea98f9f5f9a',
 'reproduction/scientific/src/posthoc_attacker_eval.py': '240cc622bd6f36f18e671824634039f71bdc028e65ae07fee7b37a347898f0b2',
 'reproduction/scientific/src/protocol.py': '72988ab1e6b60f77b9aeeb73371d2d4ee8b1007c594014fc45fbe17d37f266d5',
 'reproduction/scientific/src/result_audit.py': '943f55c296eeba099c25d75862fc78ee127d94550f247a835588f405c4980b6d',
 'reproduction/scientific/src/selection_provenance.py': 'db1834a03d08318bfd2dd2d221c719fe61225ec4b460def11da1256a87ae6467',
 'reproduction/scientific/src/semantic_analysis.py': '02114e67fd87669c7a1e0237748cf2197b21b14f7aea6eb516d94627927c8124',
 'reproduction/scientific/src/semantic_result_audit.py': 'e57e4dc8b49f1dba3066caa2695bdff0efd1df2ede1725b66eddef6f0418b7d7',
 'reproduction/scientific/src/train.py': '865f4448f539f1908fba3f4ceb0f36a012a617a88a36761b7a011c88f021b195',
 'reproduction/scientific/src/watcher_state.py': 'f4103fe1ae96dc7f298e8469173e29ef0dbc2a859a13c99ae17fb8a74e1759b3',
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
 'file_count': 67,
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

PUBLIC_ADDITIONS = frozenset(PAPER_HASHES) | {'docs/REPRODUCIBILITY.md', 'scripts/paper_repository_policy.py', 'scripts/run_study.py', 'tests/test_paper_reproduction.py'}

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
        if "sha256" in record and hashlib.sha256(read_bytes("reproduction/" + name)).hexdigest() != record["sha256"]:
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
