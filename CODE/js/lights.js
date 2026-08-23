// 光线数据集解析与三维对象管理。
// 支持三种 JSON：
//   A. perlight / light_dirs.json：含 frames[].direction（light_to_surface 约定）
//   B. LH 数据集 lights.json：{"0001": {light_pos_world, light_rgb, ...}}

import * as THREE from "three";

export const LIGHT_PALETTE = [
  0x4da3ff, 0xffb454, 0x7bd88f, 0xff6b9d, 0xc792ea,
  0x4de1c2, 0xf2e35f, 0xff8a65,
];

function normalize3(vector) {
  const length = Math.hypot(vector[0], vector[1], vector[2]);
  if (!Number.isFinite(length) || length < 1e-12) return null;
  return [vector[0] / length, vector[1] / length, vector[2] / length];
}

// 解析 perlight / light_dirs.json（Schema A）。
function parsePerlightJSON(data, name) {
  const framesRaw = Array.isArray(data.frames) ? data.frames : null;
  if (!framesRaw || framesRaw.length === 0) {
    throw new Error("JSON 中缺少 frames 数组。");
  }
  const frames = [];
  for (const [index, frame] of framesRaw.entries()) {
    const dirSource = frame.direction ?? frame.ray_direction_light_to_surface ?? frame.raw;
    if (!Array.isArray(dirSource) || dirSource.length !== 3) continue;
    const direction = normalize3(dirSource.map(Number));
    if (!direction) continue;
    frames.push({
      index: frame.index ?? index,
      fid: Number(frame.fid ?? index),
      direction,
      position: Array.isArray(frame.light_position_world) && frame.light_position_world.length === 3
        ? frame.light_position_world.map(Number)
        : null,
      exposure: frame.exposure_log_delta ?? null,
    });
  }
  if (frames.length === 0) throw new Error("frames 中没有可用的方向数据。");

  const initialization = data.initialization ?? {};
  let referenceCenter = null;
  if (Array.isArray(initialization.reference_center) && initialization.reference_center.length === 3) {
    referenceCenter = initialization.reference_center.map(Number);
  } else if (Array.isArray(data.reference_center) && data.reference_center.length === 3) {
    referenceCenter = data.reference_center.map(Number);
  }

  const meta = {
    名称: name,
    版本: data.photometric_version ?? "-",
    光照模式: data.light_mode ?? "-",
    方向约定: data.direction_convention ?? "light_to_surface",
    帧数: frames.length,
    全局强度: data.global_intensity ?? "-",
    角半径: data.angular_radius_degrees != null ? `${Number(data.angular_radius_degrees).toFixed(3)}°` : "-",
    光源颜色: Array.isArray(data.light_color) ? data.light_color.map((value) => Number(value).toFixed(3)).join(", ") : "-",
    参考中心: referenceCenter ? referenceCenter.map((value) => value.toFixed(4)).join(", ") : "无",
    初始化: initialization.type ?? "-",
    光源文件: initialization.lights_path ?? "-",
  };

  return {
    kind: "direction",
    mode: data.light_mode ?? "unknown",
    convention: data.direction_convention ?? "light_to_surface",
    frames,
    referenceCenter,
    lightColor: Array.isArray(data.light_color) && data.light_color.length === 3
      ? data.light_color.map(Number)
      : null,
    meta,
  };
}

// 解析 LH 数据集 lights.json（Schema B）。
function parseLightsJSON(data, name) {
  const keys = Object.keys(data).sort();
  const frames = [];
  for (const key of keys) {
    const entry = data[key];
    if (!entry || typeof entry !== "object") continue;
    if (!Array.isArray(entry.light_pos_world) || entry.light_pos_world.length !== 3) continue;
    frames.push({
      index: frames.length,
      fid: Number(key),
      direction: null,
      position: entry.light_pos_world.map(Number),
      exposure: entry.intensity ?? null,
      rgb: Array.isArray(entry.light_rgb) ? entry.light_rgb.map(Number) : null,
    });
  }
  if (frames.length === 0) throw new Error("lights.json 中没有 light_pos_world 条目。");

  return {
    kind: "position",
    mode: "gt_point(lights.json)",
    convention: "light_to_surface",
    frames,
    referenceCenter: null,
    lightColor: null,
    meta: {
      名称: name,
      格式: "LH lights.json",
      帧数: frames.length,
      说明: "世界坐标点光源位置，方向需相对参考中心计算",
    },
  };
}

export function parseLightJSON(data, name) {
  if (data && typeof data === "object" && !Array.isArray(data)) {
    if (Array.isArray(data.frames)) return parsePerlightJSON(data, name);
    const firstValue = Object.values(data)[0];
    if (firstValue && typeof firstValue === "object" && Array.isArray(firstValue.light_pos_world)) {
      return parseLightsJSON(data, name);
    }
  }
  throw new Error("无法识别的 JSON 格式（需为 perlight/light_dirs.json 或 lights.json）。");
}

// 每条光线数据集对应一组三维对象，随显示模式/参考中心重建。
export class LightLayer {
  constructor(scene, dataset, colorHex) {
    this.scene = scene;
    this.dataset = dataset;
    this.color = new THREE.Color(colorHex);
    this.group = new THREE.Group();
    this.visible = true;
    this.trailLength = 60;
    this.sphereRadius = 1;
    this.displayMode = "sphere";
    this.worldCenter = new THREE.Vector3(0, 0, 0);
    scene.add(this.group);

    this.trajectoryLine = null;
    this.trailLine = null;
    this.arrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, colorHex, 0.12, 0.06
    );
    this.lightMarker = new THREE.Mesh(
      new THREE.SphereGeometry(0.03, 16, 12),
      new THREE.MeshBasicMaterial({ color: colorHex })
    );
    this.group.add(this.arrow, this.lightMarker);
  }

  hasPositions() {
    return this.dataset.frames.some((frame) => frame.position);
  }

  referenceCenterVector() {
    const center = this.dataset.referenceCenter;
    if (center) return new THREE.Vector3(center[0], center[1], center[2]);
    return this.worldCenter.clone();
  }

  // 计算某一帧箭头尾部（光源侧）的显示坐标。
  displayPoint(frame) {
    if (this.displayMode === "world" && frame.position) {
      return new THREE.Vector3(frame.position[0], frame.position[1], frame.position[2]);
    }
    if (!frame.direction) return null;
    const center = this.referenceCenterVector();
    return new THREE.Vector3(
      center.x + frame.direction[0] * this.sphereRadius,
      center.y + frame.direction[1] * this.sphereRadius,
      center.z + frame.direction[2] * this.sphereRadius
    );
  }

  rebuild() {
    if (this.trajectoryLine) {
      this.group.remove(this.trajectoryLine);
      this.trajectoryLine.geometry.dispose();
      this.trajectoryLine.material.dispose();
      this.trajectoryLine = null;
    }
    const points = this.dataset.frames
      .map((frame) => this.displayPoint(frame))
      .filter((point) => point !== null);
    if (points.length >= 2) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      this.trajectoryLine = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: this.color, transparent: true, opacity: 0.55 })
      );
      this.group.add(this.trajectoryLine);
    }
    this.group.visible = this.visible;
  }

  updateFrame(frameIndex) {
    const frames = this.dataset.frames;
    if (frames.length === 0) return;
    const clamped = Math.min(Math.max(frameIndex, 0), frames.length - 1);
    const frame = frames[clamped];
    const anchor = this.displayPoint(frame);
    if (!anchor) {
      this.arrow.visible = false;
      this.lightMarker.visible = false;
      return;
    }

    // 与 scripts/visualize_stage1_light_trajectory.py 的约定保持一致：
    // 存储方向为 light_to_surface（从光源指向表面/参考中心）。
    // 球面模式：箭头从参考中心沿 +dir 指向球面标记（表示向量本身）；
    // 世界模式：箭头从光源位置沿光线指向参考中心。
    const center = this.referenceCenterVector();
    let direction;
    let arrowLength;
    if (this.displayMode === "world" && frame.position) {
      direction = center.clone().sub(anchor);
      const distance = direction.length();
      if (distance < 1e-9) direction.set(0, 0, -1);
      else direction.divideScalar(distance);
      arrowLength = Math.max(0.15, distance * 0.95);
      this.arrow.position.copy(anchor);
    } else {
      direction = new THREE.Vector3(frame.direction[0], frame.direction[1], frame.direction[2]);
      arrowLength = Math.max(0.2, this.sphereRadius * 0.94);
      this.arrow.position.copy(center);
    }
    this.arrow.setDirection(direction);
    const headLength = Math.min(arrowLength * 0.25, Math.max(0.08, this.sphereRadius * 0.18));
    this.arrow.setLength(arrowLength, headLength, headLength * 0.5);
    this.arrow.visible = this.visible;

    this.lightMarker.position.copy(anchor);
    this.lightMarker.visible = this.visible;

    this.updateTrail(clamped);
  }

  updateTrail(currentIndex) {
    if (this.trailLine) {
      this.group.remove(this.trailLine);
      this.trailLine.geometry.dispose();
      this.trailLine.material.dispose();
      this.trailLine = null;
    }
    if (this.trailLength <= 1) return;
    const start = Math.max(0, currentIndex - this.trailLength + 1);
    const points = [];
    for (let i = start; i <= currentIndex; i++) {
      const point = this.displayPoint(this.dataset.frames[i]);
      if (point) points.push(point);
    }
    if (points.length >= 2) {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      this.trailLine = new THREE.Line(
        geometry,
        new THREE.LineBasicMaterial({ color: this.color, transparent: true, opacity: 0.95, linewidth: 2 })
      );
      this.group.add(this.trailLine);
    }
  }

  setVisible(visible) {
    this.visible = visible;
    this.group.visible = visible;
  }

  dispose() {
    this.scene.remove(this.group);
    this.group.traverse((object) => {
      if (object.geometry) object.geometry.dispose();
      if (object.material) object.material.dispose();
    });
  }
}
