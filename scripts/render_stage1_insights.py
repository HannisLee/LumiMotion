
#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import os
from os import makedirs
import json
import torch.nn.functional as F
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene, GaussianModel, DeformModel
from scene.photometric_lambertian import PhotometricLambertianRenderer
import imageio
import cv2
import re
import tqdm
import numpy as np
import torchvision


def _tensor_to_uint8(image):
    image = image.detach().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    return np.ascontiguousarray((image * 255.0).round().astype(np.uint8))


def _save_contact_sheet(images, output_path):
    if not images:
        return
    indices = [0, (len(images) - 1) // 2, len(images) - 1]
    selected = []
    for index in indices:
        image = images[index]
        if image.ndim == 2:
            image = image[:, :, None]
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        selected.append(image)
    imageio.imwrite(output_path, np.concatenate(selected, axis=1))


def _write_alpha_stats(alpha_values, output_path):
    stack = np.stack(alpha_values)
    temporal_diff = (
        float(np.abs(stack[1:] - stack[:-1]).mean())
        if stack.shape[0] > 1
        else 0.0
    )
    indices = [0, (stack.shape[0] - 1) // 2, stack.shape[0] - 1]
    payload = {
        "frames": int(stack.shape[0]),
        "alpha_mean": {
            "mean": float(stack.mean(axis=(1, 2)).mean()),
            "min": float(stack.mean(axis=(1, 2)).min()),
            "max": float(stack.mean(axis=(1, 2)).max()),
        },
        "coverage_gt_0.01": {
            "mean": float((stack > 0.01).mean(axis=(1, 2)).mean()),
            "min": float((stack > 0.01).mean(axis=(1, 2)).min()),
            "max": float((stack > 0.01).mean(axis=(1, 2)).max()),
        },
        "coverage_gt_0.25": {
            "mean": float((stack > 0.25).mean(axis=(1, 2)).mean()),
            "min": float((stack > 0.25).mean(axis=(1, 2)).min()),
            "max": float((stack > 0.25).mean(axis=(1, 2)).max()),
        },
        "temporal_abs_diff_mean": temporal_diff,
        "representative": {
            str(index): {
                "alpha_mean": float(stack[index].mean()),
                "coverage_gt_0.25": float((stack[index] > 0.25).mean()),
            }
            for index in indices
        },
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _write_normal_depth_metrics(
    values,
    valid_pixels,
    iteration,
    camera_name,
    output_path,
):
    indices = [0, (len(values) - 1) // 2, len(values) - 1]
    payload = {
        "iteration": int(iteration),
        "camera": camera_name,
        "alpha_threshold": 0.25,
        "frame_mean_angle_deg": values,
        "all_frames_mean_of_means_deg": float(np.mean(values)),
        "representative": {
            str(index): float(values[index])
            for index in indices
        },
        "representative_mean_deg": float(
            np.mean([values[index] for index in indices])
        ),
        "valid_pixels": valid_pixels,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def render_set(dataset: ModelParams, pipeline: PipelineParams, load_iter):
    with torch.no_grad():
    
        dataset.eval = True
        deform = DeformModel(deform_type=dataset.deform_type, is_blender=dataset.is_blender,
                             hyper_dim=dataset.hyper_dim,
                             pred_color=dataset.pred_color)
        deform_loaded = deform.load_weights(dataset.model_path, iteration=load_iter)

        gs_fea_dim = dataset.hyper_dim
        gaussians = GaussianModel(dataset.sh_degree, no_binary_separation=dataset.no_binary_separation,
                                 fea_dim=gs_fea_dim)

        scene = Scene(dataset, gaussians, load_iteration=load_iter)
        render_mode = getattr(pipeline, "render_mode", "original_sh")
        if render_mode == "original":
            render_mode = "original_sh"
        if render_mode not in {"original_sh", "photometric_lambertian"}:
            raise ValueError(f"Unsupported render mode: {render_mode}")
        pipeline.render_mode = render_mode
        photometric_renderer = None
        if render_mode == "photometric_lambertian":
            if not gaussians.use_photometric_albedo:
                gaussians.enable_photometric_albedo()
            photometric_renderer = PhotometricLambertianRenderer(
                scene.all_timesteps, device="cuda"
            )
            photometric_renderer.load_weights(dataset.model_path, scene.loaded_iter)
            photometric_renderer.eval()

        original_train_cameras = scene.getTrainCameras()
        all_timesteps = scene.all_timesteps

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # for synthetic scenes 5:6 train cameras have nice view
        # cameras = scene.getTrainCameras()[5:6]
        cameras = scene.getTestCameras()[:1]

        #for enerf actor 1 in paper we use cam 15.jpg :
        # cameras = [c for c in scene.getTrainCameras() if "15.jpg" in c.image_path_train_light][:1]

        assert len(cameras) > 0

        render_path_parent = os.path.join(dataset.model_path, "renders_stage1_insights",
                                            "ours_{}".format(scene.loaded_iter))
        render_path = os.path.join(render_path_parent)
        makedirs(render_path, exist_ok=True)
        experiment_path = os.path.dirname(os.path.normpath(dataset.model_path))

        for render_idx, view in enumerate(cameras):

            if dataset.load2gpu_on_the_fly:
                view.load2device()

            full_render_imgs = []
            alpha_imgs = []
            albedo_imgs = []
            small_gaussians_imgs = []
            separate_gaussians_imgs = []
            separate_gaussians_large_imgs = []
            normals_imgs = []
            photometric_normals_imgs = []
            shading_imgs = []
            ndotl_imgs = []
            alpha_values = []
            normal_depth_angles = []
            normal_depth_valid_pixels = []

            render_name = view.image_name_train_light

            
            for interp_fid, timestep_idx in tqdm.tqdm(list(all_timesteps.items())[:]):

                N = gaussians.get_xyz.shape[0]

                time_input = interp_fid.unsqueeze(0).expand(N, -1)
                d_values = deform.step(gaussians.get_xyz.detach(), time_input, feature=gaussians.get_binary_feature(),
                                        camera_center=view.camera_center)
                d_xyz, d_rotation, d_scaling, d_opacity, d_color = d_values['d_xyz'], d_values['d_rotation'], \
                                                                    d_values['d_scaling'], d_values['d_opacity'], \
                                                                    d_values['d_color']
                

                #full render
                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=d_color, \
                                    photometric_renderer=photometric_renderer)
                rendering = render_pkg["render"]

                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'full_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                full_render_imgs.append(_tensor_to_uint8(rendering))

                normal_a = F.normalize(render_pkg["rend_normal"], dim=0)
                normal_b = F.normalize(render_pkg["surf_normal"], dim=0)
                normal_dot = (normal_a * normal_b).sum(dim=0).clamp(-1.0, 1.0)
                normal_valid = (
                    (render_pkg["rend_alpha"][0] > 0.25)
                    & (render_pkg["rend_normal"].norm(dim=0) > 1e-6)
                    & (render_pkg["surf_normal"].norm(dim=0) > 1e-6)
                )
                normal_depth_valid_pixels.append(int(normal_valid.sum().item()))
                if normal_valid.any():
                    normal_angle = torch.rad2deg(
                        torch.acos(normal_dot[normal_valid])
                    ).mean()
                    normal_depth_angles.append(float(normal_angle.item()))
                else:
                    normal_depth_angles.append(float("nan"))

                if photometric_renderer is not None:
                    alpha_for_vis = render_pkg["rend_alpha"]
                    photometric_normal_vis = (
                        render_pkg["photometric_normal_map"] * 0.5 + 0.5
                    ) * alpha_for_vis
                    shading_vis = (
                        render_pkg["photometric_shading_map"]
                        .expand(3, -1, -1)
                        * alpha_for_vis
                    )
                    ndotl_vis = (
                        render_pkg["photometric_ndotl_map"] * 0.5 + 0.5
                    ).expand(3, -1, -1) * alpha_for_vis
                    photometric_normals_imgs.append(
                        _tensor_to_uint8(photometric_normal_vis)
                    )
                    shading_imgs.append(_tensor_to_uint8(shading_vis))
                    ndotl_imgs.append(_tensor_to_uint8(ndotl_vis))
                

                ## alpha render
                alpha_rend = render_pkg["rend_alpha"]
                alpha_values.append(alpha_rend[0].detach().cpu().numpy())
                torchvision.utils.save_image(
                    alpha_rend.clamp(0.0, 1.0),
                    os.path.join(render_path, 'alpha_t{}_cam{}'.format(timestep_idx, render_name) + ".png"),
                )
                # torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'full_{}'.format(timestep_idx) + ".png"))
                img_np = alpha_rend.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                alpha_imgs.append(img_np)

                ## normals render
                norm_rend = render_pkg["rend_normal_view"]* 0.5 + 0.5
                torchvision.utils.save_image(norm_rend.clamp(0.0,1.0), os.path.join(render_path, 'normals_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = norm_rend.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                normals_imgs.append(img_np)

                ## show small gaussians
                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, 0, d_opacity=d_opacity, d_color=d_color, \
                                    clamp_scale_for_vis=True,
                                    photometric_renderer=photometric_renderer)
                rendering = render_pkg["render"]
                # torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'gaussians_small_{}'.format(timestep_idx) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                small_gaussians_imgs.append(img_np)


                ##albedo- show color with no mlp-modifications and no sh1-3
                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, d_scaling, d_opacity=d_opacity, d_color=None, \
                                    override_color=(gaussians.get_photometric_albedo
                                                    if photometric_renderer is not None
                                                    else gaussians.get_albedo),
                                    photometric_renderer=photometric_renderer)
                rendering = render_pkg["render"]
                # torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'albedo_{}'.format(timestep_idx) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                albedo_imgs.append(img_np)


                ## show small gaussians - separation
                sep_color = torch.zeros_like(gaussians.get_xyz)
                sep_color[:, 0:1] = gaussians.get_binary_feature()
                sep_color[:, 1:2] = 1-gaussians.get_binary_feature()

                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, 0, d_opacity=d_opacity, d_color=d_color, \
                                    clamp_scale_for_vis=True, override_color = sep_color,
                                    photometric_renderer=photometric_renderer)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'separation_small_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                separate_gaussians_imgs.append(img_np)

                ## show large gaussians - separation
                sep_color = torch.zeros_like(gaussians.get_xyz)
                sep_color[:, 0:1] = gaussians.get_binary_feature()
                sep_color[:, 1:2] = 1-gaussians.get_binary_feature()

                render_pkg = render(view, gaussians, pipeline, background, \
                                    d_xyz, d_rotation, 0, d_opacity=d_opacity, d_color=d_color, \
                                    clamp_scale_for_vis=False, override_color = sep_color,
                                    photometric_renderer=photometric_renderer)
                rendering = render_pkg["render"]
                torchvision.utils.save_image(rendering.clamp(0.0,1.0), os.path.join(render_path, 'separation_large_t{}_cam{}'.format(timestep_idx, render_name) + ".png"))
                img_np = rendering.permute(1, 2, 0).cpu().numpy()
                img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
                img_np = np.ascontiguousarray(img_np)
                separate_gaussians_large_imgs.append(img_np)
                
            if dataset.load2gpu_on_the_fly:
                    view.load2device('cpu')

            _save_contact_sheet(
                full_render_imgs,
                os.path.join(experiment_path, "eval_rgb_contact_sheet.png"),
            )
            _save_contact_sheet(
                alpha_imgs,
                os.path.join(experiment_path, "alpha_render_contact_sheet.png"),
            )
            _save_contact_sheet(
                normals_imgs,
                os.path.join(experiment_path, "eval_normals_contact_sheet.png"),
            )
            _save_contact_sheet(
                albedo_imgs,
                os.path.join(experiment_path, "eval_albedo_contact_sheet.png"),
            )
            _save_contact_sheet(
                separate_gaussians_imgs,
                os.path.join(
                    experiment_path,
                    "eval_separation_small_contact_sheet.png",
                ),
            )
            _save_contact_sheet(
                separate_gaussians_large_imgs,
                os.path.join(
                    experiment_path,
                    "eval_separation_large_contact_sheet.png",
                ),
            )
            _save_contact_sheet(
                photometric_normals_imgs,
                os.path.join(
                    experiment_path,
                    "eval_photometric_normal_contact_sheet.png",
                ),
            )
            _save_contact_sheet(
                shading_imgs,
                os.path.join(experiment_path, "eval_shading_contact_sheet.png"),
            )
            _save_contact_sheet(
                ndotl_imgs,
                os.path.join(experiment_path, "eval_ndotl_contact_sheet.png"),
            )
            _write_alpha_stats(
                alpha_values,
                os.path.join(experiment_path, "alpha_render_stats.json"),
            )
            _write_normal_depth_metrics(
                normal_depth_angles,
                normal_depth_valid_pixels,
                scene.loaded_iter,
                render_name,
                os.path.join(
                    experiment_path,
                    f"normal_depth_metrics_{scene.loaded_iter}.json",
                ),
            )

            # Save videos
            vid_name = f"full_render_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in full_render_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"alpha_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in alpha_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"small_gaussians_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in small_gaussians_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Albedo_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in albedo_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Separation_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in separate_gaussians_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Separation_large_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in separate_gaussians_large_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()

            vid_name = f"Normals_cam{render_name}"
            output_video_path = os.path.join(render_path, f"{vid_name}.mp4")
            writer = imageio.get_writer(output_video_path, fps=15)
            for img in normals_imgs:
                cv2.putText(img, vid_name, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                writer.append_data(img)
            writer.close()


            
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)

    parser.add_argument('--load_iter', type=int, default=-1, help="Iteration to load.")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")


    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_set(model.extract(args), pipeline.extract(args), args.load_iter)
