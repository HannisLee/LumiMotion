# Stage1 V5: camera-back ellipse light initialization

V5 keeps the V3 per-frame directional light table and Lambertian render path, but replaces the world-space circular initialization and multistart with a one-time camera-relative initialization. Training after initialization uses the existing RGB reconstruction loss and does not keep an ellipse, camera-side, or light-position constraint.

## Light convention

The renderer uses `max(0, normal dot light_dir)`, so `light_dir` points from a surface toward the light source. For the fixed camera in `data/LH-data/danamic/cloth_dynamic_shape/camera.json`, the physical light is behind the camera and its rays travel toward the object. V5 therefore initializes the model direction opposite the camera center ray.

At the start of S1C only, V5 obtains camera right, up, and forward vectors from the first training camera's center rays and fills the per-frame table with:

```text
raw_light_dir[t] = normalize(
    a * cos(theta[t]) * camera_right
  + b * sin(theta[t]) * camera_up
  - c * camera_forward
)
```

`a`, `b`, and `c` are the horizontal radius, vertical radius, and camera-back offset. The default trajectory covers one ellipse. This is initialization only: after the first optimizer step, every per-frame direction is freely optimized.

## V5 stages

| Stage | Iteration range | Render mode | Trainable parameters | Densify/prune |
| --- | ---: | --- | --- | --- |
| S1A `s1a_geometry_warmup` | 1-16000 | `original_sh` | Original Gaussian and deform parameters | enabled |
| S1B `s1b_geometry_settle` | 16001-20000 | `original_sh` | Original Gaussian and deform parameters | disabled |
| S1C `s1c_light_calib` | 20001-24000 | `photometric_lambertian` | Per-frame light table only | disabled |
| S1D `s1d_albedo_decompose` | 24001-30000 | `photometric_lambertian` | Light table and photometric albedo | disabled |
| S1E `s1e_joint_refine` | 30001-35000, optional | `photometric_lambertian` | Light, photometric albedo, rotation, scale | disabled |

S1D is the intended V5 endpoint. Run S1E only when S1D has stable light/albedo decomposition but consistent normal or scale artifacts remain. Position, opacity, and deformation stay frozen throughout S1C-S1E.

## Parameters

V5 removes `photometric_num_ctrl_points`, the V3 circular initialization parameters, and all multistart parameters. New ellipse parameters are:

```text
photometric_camera_ellipse_horizontal = 0.7
photometric_camera_ellipse_vertical   = 0.35
photometric_camera_ellipse_back       = 1.0
photometric_camera_ellipse_phase      = 0.0
photometric_camera_ellipse_direction_sign = 1
photometric_camera_ellipse_span       = 2*pi
```

For the RGB-only baseline, keep these regularizers at zero in S1C-S1D:

```text
lambda_photometric_light_smooth1 = 0.0
lambda_photometric_light_smooth2 = 0.0
lambda_photometric_hemi          = 0.0
lambda_photometric_albedo_prior  = 0.0
```

`lights.json`, normal EXR, and albedo images are not training supervision. They may be used only after training for diagnostics.

## Command template

Set `SOURCE` to the prepared Blender-format dataset directory containing `transforms_train.json`, `transforms_test.json`, and the training images. `MODEL_BASE` does not include the automatic `_static` suffix.

```bash
ENV=lumimotion-cu126
SOURCE=data/LH-data/transfer-dynamic/<scene>
MODEL_BASE=output/stage1-v5/<scene>_v5
MODEL=${MODEL_BASE}_static
COMMON="--source_path ${SOURCE} --train_light_folder images --is_blender --eval --deform-type static --resolution 2"
```

S1A:

```bash
conda run --no-capture-output -n ${ENV} python -m scripts.train_stage1 \
  ${COMMON} --model_path ${MODEL_BASE} \
  --render_mode original_sh --photometric_stage s1a_geometry_warmup \
  --iterations 16000 --densify_until_iter 14000 \
  --save_iterations 16000 --test_iterations 16000
```

S1B:

```bash
conda run --no-capture-output -n ${ENV} python -m scripts.train_stage1 \
  ${COMMON} --model_path ${MODEL_BASE} --load_iter 16000 \
  --render_mode original_sh --photometric_stage s1b_geometry_settle \
  --iterations 20000 --save_iterations 20000 --test_iterations 20000
```

S1C:

```bash
conda run --no-capture-output -n ${ENV} python -m scripts.train_stage1 \
  ${COMMON} --model_path ${MODEL_BASE} --load_iter 20000 \
  --render_mode photometric_lambertian --photometric_stage s1c_light_calib \
  --photometric_camera_ellipse_horizontal 0.7 \
  --photometric_camera_ellipse_vertical 0.35 \
  --photometric_camera_ellipse_back 1.0 \
  --photometric_camera_ellipse_span 6.283185307179586 \
  --photometric_s1c_light_lr 0.001 --photometric_s1c_albedo_lr 0.0 \
  --iterations 24000 --save_iterations 24000 --test_iterations 24000
```

S1D:

```bash
conda run --no-capture-output -n ${ENV} python -m scripts.train_stage1 \
  ${COMMON} --model_path ${MODEL_BASE} --load_iter 24000 \
  --render_mode photometric_lambertian --photometric_stage s1d_albedo_decompose \
  --photometric_s1d_light_lr 0.0001 --photometric_s1d_albedo_lr 0.001 \
  --iterations 30000 --save_iterations 30000 --test_iterations 30000
```

Optional S1E:

```bash
conda run --no-capture-output -n ${ENV} python -m scripts.train_stage1 \
  ${COMMON} --model_path ${MODEL_BASE} --load_iter 30000 \
  --render_mode photometric_lambertian --photometric_stage s1e_joint_refine \
  --photometric_s1e_light_lr 0.00005 \
  --photometric_s1e_albedo_lr 0.0005 \
  --photometric_s1e_rotation_lr 0.00001 \
  --photometric_s1e_scaling_lr 0.00001 \
  --iterations 35000 --save_iterations 35000 --test_iterations 35000
```

Each photometric stage saves its initial table at `photometric/iteration_<load_iter + 1>/photometric.pth`. S1C's initial checkpoint contains the camera-back ellipse metadata; S1D and S1E load the preceding photometric checkpoint without reinitializing it.
