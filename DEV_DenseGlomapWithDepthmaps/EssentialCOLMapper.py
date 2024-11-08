import subprocess
from tqdm import tqdm
import time
__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
Research Domain: Computer Vision, Machine Learning
Email: xuanli(dot)chen(at)icloud.com
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""
class COLMapper3r:
    # TODO: add the gin config file
    def __init__(self, dp_blob, dp_output):
        self.colmap_cmd = 'colmap'
        self.dp_blob = dp_blob
        self.dp_output = dp_output
        self.fps_imgs = list(dp_blob.glob("*.jpg")) + list(dp_blob.glob("*.png"))
        self._model_report = None
        start_mapper = time.time()
        self._run_mapper_()
        end_mapper = time.time()
        print(f"Mapper took {end_mapper - start_mapper:.2f} seconds")

    def _run_mapper_(self):
        # 1. According to stem, selecting the images and write into the tmp folder
        dp_cwd_cam = self.dp_output
        dp_cwd_cam.mkdir(parents=True, exist_ok=True)

        fps_current = self.fps_imgs
        fp_db = dp_cwd_cam / f'database-{self.dp_blob.name}.db'
        dp_recon = dp_cwd_cam / f'recon-{self.dp_blob.name}'
        # Check whether the Mapping is already done
        if dp_recon.exists():
            print(f"Reconstruction {dp_recon.name} already exists.")
            return

        dp_recon.mkdir(parents=True, exist_ok=True)

        # 2. Run the COLMAP Feature Extractor
        cmd_feat = (
            f"{self.colmap_cmd}"
            f" feature_extractor "
            f"--database_path {fp_db.as_posix()} "
            f"--image_path {self.dp_blob.as_posix()} "
            f"--SiftExtraction.estimate_affine_shape 1 "
            f"--SiftExtraction.domain_size_pooling 1 "
            f"--SiftExtraction.max_num_orientations 8 "
        )
        result_feat = subprocess.run(cmd_feat, shell=True, check=True, capture_output=True, text=True)
        print("Feature Extractor Output:", result_feat.stdout)
        print("Feature Extractor Errors:", result_feat.stderr)

        # 3. Run the COLMAP Exhaustive Matcher
        cmd_match = (
            f"{self.colmap_cmd} exhaustive_matcher "
            f"--database_path {fp_db.as_posix()} "
            f"--SiftMatching.use_gpu 0 "
        )
        result_match = subprocess.run(cmd_match, shell=True, check=True, capture_output=True, text=True)
        print("Exhaustive Matcher Output:", result_match.stdout)
        print("Exhaustive Matcher Errors:", result_match.stderr)

        # 4. Run the COLMAP Mapper
        # TODO: set the ba numbers as a parameter
        cmd_mapper = (
            f"{self.colmap_cmd} mapper "
            f"--database_path {fp_db.as_posix()} "
            f"--image_path {self.dp_blob.as_posix()} "
            f"--output_path {dp_recon.as_posix()} "
            f"--Mapper.multiple_models 1 "
            # f"--Mapper.ba_local_max_num_iterations 100 "
            # f"--Mapper.ba_global_max_num_iterations 100 "
        )
        result_mapper = subprocess.run(cmd_mapper, shell=True, check=True, capture_output=True, text=True)
        print("Mapper Output:", result_mapper.stdout)
        print("Mapper Errors:", result_mapper.stderr)

        # 6. Analyze the reconstructed model
        dp_models = list(dp for dp in dp_recon.glob("*") if dp.is_dir())
        for i, dpm in enumerate(dp_models):
            assert dpm.exists(), f"Model {i:02d} does not exist: {dpm.as_posix()}"
            cmd_analyze = (
                f"{self.colmap_cmd} model_analyzer "
                f"--path {dpm.as_posix()} "
            )
            result_analyze = subprocess.run(cmd_analyze, shell=True, check=True, capture_output=True, text=True)
            print("Model %i Analyzer Output:" % i, result_analyze.stdout)
            print("Model Errors:", result_analyze.stderr)
            self._model_report = result_analyze.stdout