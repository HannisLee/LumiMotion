import * as THREE from "three";
import { OrbitControls } from "../vendor/three/OrbitControls.js";
import { parsePLY } from "./ply_loader.js";
import { parseLightJSON, LightLayer, LIGHT_PALETTE } from "./lights.js";

// ---------- 全局状态 ----------
const state = {
  lightLayers: [],       // {id, name, layer, dataset}
  pointClouds: [],       // {id, name, points, bbox, isDataset}
  datasetCloudId: null,  // 下拉框加载的数据集点云 id
  datasetIndex: null,
  nextId: 1,
  currentFrame: 0,
  playing: false,
  playSpeed: 1,
  loop: true,
  selectedId: null,
  displayMode: "sphere",
  sphereRadius: 1,
  trailLength: 60,
};

// ---------- 场景搭建 ----------
const viewport = document.getElementById("viewport");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewport.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x14161a);

const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 500);
camera.up.set(0, 0, 1); // 数据为 Blender Z-up 坐标系
camera.position.set(3.2, -3.6, 2.4);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// 根节点：切换 Y-up 时整体旋转。
const rootGroup = new THREE.Group();
scene.add(rootGroup);

const gridHelper = new THREE.GridHelper(10, 20, 0x3a4150, 0x262b34);
gridHelper.rotation.x = Math.PI / 2; // 放到 XY 平面（Z-up）
const axesHelper = new THREE.AxesHelper(1.2);
rootGroup.add(gridHelper, axesHelper);

const referenceSphere = new THREE.Mesh(
  new THREE.SphereGeometry(1, 36, 18),
  new THREE.MeshBasicMaterial({ color: 0x4da3ff, wireframe: true, transparent: true, opacity: 0.12 })
);
rootGroup.add(referenceSphere);

function resize() {
  const width = viewport.clientWidth;
  const height = viewport.clientHeight;
  renderer.setSize(width, height);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

// ---------- 工具函数 ----------
const toastElement = document.getElementById("toast");
let toastTimer = null;
function toast(message, isError = false) {
  toastElement.textContent = message;
  toastElement.className = isError ? "show error" : "show";
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { toastElement.className = ""; }, isError ? 8000 : 3500);
}

function sceneCenter() {
  const selected = state.lightLayers.find((entry) => entry.id === state.selectedId);
  if (selected?.layer.dataset.referenceCenter) {
    const [x, y, z] = selected.layer.dataset.referenceCenter;
    return new THREE.Vector3(x, y, z);
  }
  for (const entry of state.lightLayers) {
    if (entry.layer.dataset.referenceCenter) {
      const [x, y, z] = entry.layer.dataset.referenceCenter;
      return new THREE.Vector3(x, y, z);
    }
  }
  if (state.pointClouds.length > 0) {
    const box = new THREE.Box3();
    for (const cloud of state.pointClouds) box.union(cloud.bbox);
    return box.getCenter(new THREE.Vector3());
  }
  return new THREE.Vector3(0, 0, 0);
}

function refreshLightDisplay() {
  const center = sceneCenter();
  for (const entry of state.lightLayers) {
    entry.layer.worldCenter = center.clone();
    entry.layer.displayMode = state.displayMode === "world" && entry.layer.hasPositions()
      ? "world"
      : "sphere";
    entry.layer.sphereRadius = state.sphereRadius;
    entry.layer.trailLength = state.trailLength;
    entry.layer.rebuild();
  }
  referenceSphere.position.copy(center);
  referenceSphere.scale.setScalar(state.sphereRadius);
  updateFrame(state.currentFrame);
}

function updateFrame(frameIndex) {
  state.currentFrame = frameIndex;
  for (const entry of state.lightLayers) entry.layer.updateFrame(frameIndex);
  updateFrameInfo();
  const slider = document.getElementById("frame-slider");
  slider.value = frameIndex;
  document.getElementById("frame-label").textContent = `${frameIndex} / ${slider.max}`;
}

function maxFrames() {
  return state.lightLayers.reduce((max, entry) => Math.max(max, entry.layer.dataset.frames.length), 0);
}

function updateTimelineRange() {
  const slider = document.getElementById("frame-slider");
  const total = maxFrames();
  slider.max = Math.max(0, total - 1);
  if (state.currentFrame > Number(slider.max)) updateFrame(Number(slider.max));
  else updateFrame(state.currentFrame);
}

// ---------- 帧信息 / 元信息面板 ----------
function formatVector(values, digits = 4) {
  return values.map((value) => Number(value).toFixed(digits)).join(", ");
}

function updateFrameInfo() {
  const panel = document.getElementById("frame-info");
  if (state.lightLayers.length === 0) {
    panel.textContent = "未加载光线数据";
    return;
  }
  const lines = [];
  for (const entry of state.lightLayers) {
    const frames = entry.layer.dataset.frames;
    if (frames.length === 0) continue;
    const frame = frames[Math.min(state.currentFrame, frames.length - 1)];
    const parts = [`#${frame.index} fid=${frame.fid.toFixed(3)}`];
    if (frame.direction) parts.push(`方向 ${formatVector(frame.direction, 3)}`);
    if (frame.position) parts.push(`位置 ${formatVector(frame.position, 2)}`);
    if (frame.exposure != null) parts.push(`强度/曝光 ${Number(frame.exposure).toFixed(3)}`);
    lines.push(`● ${entry.name}: ${parts.join("  ")}`);
  }
  panel.textContent = lines.join("\n");
}

function updateMetaInfo() {
  const panel = document.getElementById("meta-info");
  const selected = state.lightLayers.find((entry) => entry.id === state.selectedId);
  if (!selected) {
    panel.textContent = "点击左侧列表项查看。";
    return;
  }
  panel.textContent = Object.entries(selected.layer.dataset.meta)
    .map(([key, value]) => `${key}: ${value}`)
    .join("\n");
}

// ---------- 列表 UI ----------
function renderLists() {
  const lightList = document.getElementById("light-list");
  lightList.innerHTML = "";
  for (const entry of state.lightLayers) {
    const item = document.createElement("li");
    item.className = entry.id === state.selectedId ? "selected" : "";
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = `#${entry.layer.color.getHexString()}`;
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = entry.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${entry.layer.dataset.frames.length}帧`;
    const eye = document.createElement("button");
    eye.textContent = entry.layer.visible ? "隐藏" : "显示";
    eye.addEventListener("click", (event) => {
      event.stopPropagation();
      entry.layer.setVisible(!entry.layer.visible);
      renderLists();
    });
    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "删除";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      entry.layer.dispose();
      state.lightLayers = state.lightLayers.filter((other) => other !== entry);
      if (state.selectedId === entry.id) {
        state.selectedId = state.lightLayers[0]?.id ?? null;
      }
      refreshLightDisplay();
      updateTimelineRange();
      renderLists();
      updateMetaInfo();
    });
    item.addEventListener("click", () => {
      state.selectedId = entry.id;
      renderLists();
      updateMetaInfo();
      refreshLightDisplay();
    });
    item.append(swatch, name, count, eye, remove);
    lightList.appendChild(item);
  }

  const plyList = document.getElementById("ply-list");
  plyList.innerHTML = "";
  for (const cloud of state.pointClouds) {
    const item = document.createElement("li");
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = "#7bd88f";
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = cloud.name;
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = `${cloud.count.toLocaleString()}点`;
    const eye = document.createElement("button");
    eye.textContent = cloud.points.visible ? "隐藏" : "显示";
    eye.addEventListener("click", (event) => {
      event.stopPropagation();
      cloud.points.visible = !cloud.points.visible;
      renderLists();
    });
    const remove = document.createElement("button");
    remove.className = "remove";
    remove.textContent = "删除";
    remove.addEventListener("click", (event) => {
      event.stopPropagation();
      rootGroup.remove(cloud.points);
      cloud.points.geometry.dispose();
      cloud.points.material.dispose();
      state.pointClouds = state.pointClouds.filter((other) => other !== cloud);
      if (state.datasetCloudId === cloud.id) {
        state.datasetCloudId = null;
        document.getElementById("dataset-select").value = "";
      }
      refreshLightDisplay();
      renderLists();
    });
    item.append(swatch, name, count, eye, remove);
    plyList.appendChild(item);
  }
}

// ---------- 加载逻辑 ----------
function addLightDataset(data, name) {
  const dataset = parseLightJSON(data, name);
  const colorHex = LIGHT_PALETTE[state.lightLayers.length % LIGHT_PALETTE.length];
  const layer = new LightLayer(rootGroup, dataset, colorHex);
  const entry = { id: state.nextId++, name, layer, dataset };
  state.lightLayers.push(entry);
  state.selectedId = entry.id;
  refreshLightDisplay();
  updateTimelineRange();
  renderLists();
  updateMetaInfo();
  toast(`已加载光线数据「${name}」（${dataset.frames.length} 帧，${dataset.mode}）。`);
  return entry;
}

function addPointCloud(parsed, name, isDataset = false) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(parsed.positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(parsed.colors, 3));
  const material = new THREE.PointsMaterial({
    size: Number(document.getElementById("point-size").value),
    vertexColors: true,
    sizeAttenuation: true,
  });
  const points = new THREE.Points(geometry, material);
  const bbox = new THREE.Box3().setFromBufferAttribute(geometry.getAttribute("position"));
  rootGroup.add(points);
  const cloudId = state.nextId++;
  state.pointClouds.push({ id: cloudId, name, points, bbox, count: parsed.count, isDataset });
  if (isDataset) state.datasetCloudId = cloudId;
  refreshLightDisplay();
  renderLists();
  toast(`已加载点云「${name}」（${parsed.count.toLocaleString()} 点，颜色来源: ${parsed.colorMode}）。`);
}

async function loadFiles(files) {
  for (const file of files) {
    const lower = file.name.toLowerCase();
    try {
      if (lower.endsWith(".json")) {
        const text = await file.text();
        addLightDataset(JSON.parse(text), file.name);
      } else if (lower.endsWith(".ply")) {
        const buffer = await file.arrayBuffer();
        addPointCloud(parsePLY(buffer), file.name);
      } else if (lower.endsWith(".pth")) {
        toast(
          `.pth 为 PyTorch 检查点，浏览器无法直接解析。\n请先运行：python tools/export_photometric_pth.py ${file.name}\n再导入生成的 JSON。`,
          true
        );
      } else {
        toast(`不支持的文件类型: ${file.name}`, true);
      }
    } catch (error) {
      toast(`加载 ${file.name} 失败: ${error.message}`, true);
    }
  }
  if (state.pointClouds.length > 0 || state.lightLayers.length > 0) fitView();
}

// ---------- 视角 ----------
function fitView() {
  const box = new THREE.Box3().setFromObject(rootGroup);
  if (box.isEmpty()) return;
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length() || 1;
  const direction = camera.position.clone().sub(controls.target);
  if (direction.lengthSq() < 1e-9) direction.set(1, -1, 0.8);
  direction.normalize();
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(direction, size * 0.9);
  camera.near = Math.max(0.001, size * 0.001);
  camera.far = size * 50;
  camera.updateProjectionMatrix();
  controls.update();
}

// ---------- 事件绑定 ----------
document.getElementById("btn-load-json").addEventListener("click", () => {
  document.getElementById("file-json").click();
});
document.getElementById("btn-load-ply").addEventListener("click", () => {
  document.getElementById("file-ply").click();
});
document.getElementById("file-json").addEventListener("change", (event) => {
  loadFiles([...event.target.files]);
  event.target.value = "";
});
document.getElementById("file-ply").addEventListener("change", (event) => {
  loadFiles([...event.target.files]);
  event.target.value = "";
});

viewport.addEventListener("dragover", (event) => {
  event.preventDefault();
  viewport.classList.add("dragover");
});
viewport.addEventListener("dragleave", () => viewport.classList.remove("dragover"));
viewport.addEventListener("drop", (event) => {
  event.preventDefault();
  viewport.classList.remove("dragover");
  loadFiles([...event.dataTransfer.files]);
});

document.querySelectorAll('input[name="up-axis"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    if (!radio.checked) return;
    if (radio.value === "z") {
      rootGroup.rotation.x = 0;
      camera.up.set(0, 0, 1);
      gridHelper.rotation.x = Math.PI / 2;
    } else {
      rootGroup.rotation.x = -Math.PI / 2;
      camera.up.set(0, 1, 0);
      gridHelper.rotation.x = 0;
    }
    controls.update();
  });
});

document.querySelectorAll('input[name="display-mode"]').forEach((radio) => {
  radio.addEventListener("change", () => {
    if (radio.checked) {
      state.displayMode = radio.value;
      refreshLightDisplay();
    }
  });
});

document.getElementById("sphere-radius").addEventListener("input", (event) => {
  state.sphereRadius = Number(event.target.value);
  refreshLightDisplay();
});
document.getElementById("trail-length").addEventListener("input", (event) => {
  state.trailLength = Number(event.target.value);
  refreshLightDisplay();
});
document.getElementById("show-sphere").addEventListener("change", (event) => {
  referenceSphere.visible = event.target.checked;
});
document.getElementById("show-grid").addEventListener("change", (event) => {
  gridHelper.visible = event.target.checked;
  axesHelper.visible = event.target.checked;
});
document.getElementById("point-size").addEventListener("input", (event) => {
  const size = Number(event.target.value);
  for (const cloud of state.pointClouds) cloud.points.material.size = size;
});
document.getElementById("btn-fit").addEventListener("click", fitView);
document.getElementById("btn-clear").addEventListener("click", () => {
  for (const entry of state.lightLayers) entry.layer.dispose();
  for (const cloud of state.pointClouds) {
    rootGroup.remove(cloud.points);
    cloud.points.geometry.dispose();
    cloud.points.material.dispose();
  }
  state.lightLayers = [];
  state.pointClouds = [];
  state.selectedId = null;
  refreshLightDisplay();
  updateTimelineRange();
  renderLists();
  updateMetaInfo();
});

// 时间轴
const playButton = document.getElementById("btn-play");
playButton.addEventListener("click", () => {
  state.playing = !state.playing;
  playButton.textContent = state.playing ? "暂停" : "播放";
});
document.getElementById("frame-slider").addEventListener("input", (event) => {
  updateFrame(Number(event.target.value));
});
document.getElementById("play-speed").addEventListener("change", (event) => {
  state.playSpeed = Number(event.target.value);
});
document.getElementById("loop-play").addEventListener("change", (event) => {
  state.loop = event.target.checked;
});

// ---------- 动画循环 ----------
const BASE_FPS = 15;
let lastTime = performance.now();
let frameAccumulator = 0;

function animate(now) {
  requestAnimationFrame(animate);
  const deltaSeconds = Math.min(0.1, (now - lastTime) / 1000);
  lastTime = now;

  if (state.playing && maxFrames() > 1) {
    frameAccumulator += deltaSeconds * BASE_FPS * state.playSpeed;
    if (frameAccumulator >= 1) {
      const advance = Math.floor(frameAccumulator);
      frameAccumulator -= advance;
      const total = maxFrames();
      let next = state.currentFrame + advance;
      if (next >= total) {
        if (state.loop) next = next % total;
        else {
          next = total - 1;
          state.playing = false;
          playButton.textContent = "播放";
        }
      }
      updateFrame(next);
    }
  }

  controls.update();
  renderer.render(scene, camera);
}
requestAnimationFrame(animate);

// ---------- 数据集下拉框 ----------
async function switchDataset(name) {
  const index = state.datasetIndex;
  if (!index) return;
  const dataset = index.datasets.find((entry) => entry.name === name);
  if (!dataset || !dataset.ply) {
    toast(`数据集「${name}」缺少 points3d.ply。`, true);
    return;
  }
  try {
    const response = await fetch(dataset.ply);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const parsed = parsePLY(await response.arrayBuffer());
    // 移除上一个数据集点云，保留手动导入的点云。
    const previous = state.pointClouds.find((cloud) => cloud.id === state.datasetCloudId);
    if (previous) {
      rootGroup.remove(previous.points);
      previous.points.geometry.dispose();
      previous.points.material.dispose();
      state.pointClouds = state.pointClouds.filter((cloud) => cloud !== previous);
    }
    addPointCloud(parsed, `${name}/points3d.ply`, true);
    fitView();
  } catch (error) {
    toast(`加载数据集「${name}」失败: ${error.message}`, true);
  }
}

async function loadDatasetIndex(preferred = "only_clothV3") {
  const select = document.getElementById("dataset-select");
  try {
    const response = await fetch("data/datasets.json");
    if (!response.ok) return;
    state.datasetIndex = await response.json();
  } catch {
    return;
  }
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "（未选择）";
  select.appendChild(placeholder);
  for (const dataset of state.datasetIndex.datasets) {
    const option = document.createElement("option");
    option.value = dataset.name;
    option.textContent = dataset.ply ? dataset.name : `${dataset.name}（缺少 PLY）`;
    option.disabled = !dataset.ply;
    select.appendChild(option);
  }
  const defaultDataset = state.datasetIndex.datasets.find((entry) => entry.name === preferred && entry.ply)
    ?? state.datasetIndex.datasets.find((entry) => entry.ply);
  if (defaultDataset) {
    select.value = defaultDataset.name;
    await switchDataset(defaultDataset.name);
  }
}

document.getElementById("dataset-select").addEventListener("change", (event) => {
  if (event.target.value) switchDataset(event.target.value);
});

// ---------- 默认数据 ----------
async function loadDefaults() {
  let gtLightsPath = "lhdata/danamic/only_clothV3/lights.json";
  try {
    const response = await fetch("data/datasets.json");
    if (response.ok) {
      const index = await response.json();
      if (index.gt_lights) gtLightsPath = index.gt_lights;
    }
  } catch { /* 使用默认路径 */ }
  try {
    const response = await fetch(gtLightsPath);
    if (response.ok) addLightDataset(await response.json(), "GT lights.json（初始光线）");
  } catch { /* 静默：允许离线缺失 */ }
  await loadDatasetIndex();
  if (state.lightLayers.length > 0 || state.pointClouds.length > 0) fitView();
}
loadDefaults();
