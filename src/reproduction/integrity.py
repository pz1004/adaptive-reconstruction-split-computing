"""Scientific digests retained from the original public commit, not manifests."""
from __future__ import annotations
import hashlib
from pathlib import Path

SCIENTIFIC_HASHES = {'config/benchmark_extension_v1.json': '76b5f794fec992f7da74577639789bddc3036401c21fc49e3a363bbabd837c55',
 'config/focused-recipe.json': 'cd81753c05d4b6786d4b8e99cba5b1b74644024608fd7df2decd7f4ab8da5be8',
 'config/implementation_protocol_v2.json': '4d80ecc39d0a9f09edad47e67912477623bc642d69c1948f8631f0e3a1d513ab',
 'config/study_contract.json': 'e43780aa31661ba797d1832151cc5363a3933f90edc8f7e3375ac39e11bb9ef0',
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
 'reproduction/supporting/schemas/attacker_restart_rows.schema.json': '9a36bdd36307a32610a4980dfba9213d0e75ded5338771c1dda2fa6e12ee6dce',
 'reproduction/supporting/schemas/attacker_restart_summaries.schema.json': 'ce6985e37b110c0b9265bc34ce815bad8f4edd71b85d61a543455248dd7d81bf',
 'reproduction/supporting/schemas/benchmark_budget_attacker_rows.schema.json': '32b7203b5f3d606095c0a0f689ad971c86533aefcf835900c4a4bd06583babc6',
 'reproduction/supporting/schemas/benchmark_budget_conditions.schema.json': '2ab0feb1e3d5c3a6dc02fd2f8146671ba51b5ef655d365edb52eb703e0bb1471',
 'reproduction/supporting/schemas/benchmark_budget_paired_effects.schema.json': '8e6a8338fcbdd887b9935ce4ceb282c05a29dd50eeb21803e8ae2c576d74c715',
 'reproduction/supporting/schemas/benchmark_v1_tables.schema.json': '900ce37be4ff8df523ffa4a97888a5accd7098a297492dbb8b09e1ad360c08f3',
 'reproduction/supporting/schemas/eng_compute_v1.schema.json': '5e2603047b488be03c800de0c897ce0ac0134c44d8e546d684e42b2639f498a1',
 'reproduction/upstream/LICENSE': 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4',
 'reproduction/upstream/disco.json': '055b112bf23876b949e670ebe99ed04505835e8e1ae20716926fd13561e12c58',
 'reproduction/upstream/disco.py': '1a4f32e9d6ed438acb04652b230ba5c8877ef28955e8ff499967ae36f8a79153',
 'results/protocol/environment.json': 'c04be5a960f57c7c62cd4434cad94208f28b37aa425e945de9f564a2424186af',
 'results/protocol/frozen_selection_lock.json': '593c4bd0d4307fe6c108bc32556bb8ae349ce4689d8d443fca391af87f96fa47',
 'results/raw/budget_attacker_results.csv': '95471d76b0726095a71014185ef6bcf71ba8e11825ae1c05be23ad7da6b310ad',
 'results/raw/budget_cell_results.csv': '7ca472e026f1740ddf76627a831777c9ebec56625ab01b23ab5eeb47247a12ab',
 'results/raw/primary_seed_results.csv': '4a1cec03db1db89a9e2b010ebeeff530a6f88fe3d23e3601db19553ce96b266d',
 'results/raw/semantic_attribute_results.csv': '240e6dcd9702838b2d2f597bbebe79e3abc9535e28221499582bb065f68e8896',
 'results/raw/semantic_seed_results.csv': 'd2c3bd11ebbc9277ee8fac7b66ce2b5e3edd3959be3aa0b6a1fc3ef2971eca90'}


def verify_files(root: Path, names: list[str]) -> None:
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required aggregate input missing: {name}")
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != SCIENTIFIC_HASHES[name]:
            raise ValueError(f"Scientific checksum mismatch: {name}")
