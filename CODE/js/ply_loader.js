// PLY 解析器：支持 ascii / binary_little_endian / binary_big_endian。
// 颜色来源优先级：red/green/blue -> albedo_dc_* -> f_dc_*（SH DC 转 RGB）-> 法线 -> 灰色。

const TYPE_SIZES = {
  char: 1, int8: 1,
  uchar: 1, uint8: 1,
  short: 2, int16: 2,
  ushort: 2, uint16: 2,
  int: 4, int32: 4,
  uint: 4, uint32: 4,
  float: 4, float32: 4,
  double: 8, float64: 8,
};

const SH_C0 = 0.28209479177387814;

function parseHeader(text) {
  const lines = text.split("\n").map((line) => line.trim());
  if (lines[0] !== "ply") {
    throw new Error("不是有效的 PLY 文件（缺少 ply 文件头）。");
  }
  let format = "ascii";
  const elements = [];
  let current = null;
  let headerEnd = Buffer.byteLength(lines[0] + "\n", "latin1");
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line === "end_header") {
      headerEnd += Buffer.byteLength(line + "\n", "latin1");
      break;
    }
    headerEnd += Buffer.byteLength(line + "\n", "latin1");
    const parts = line.split(/\s+/);
    if (parts[0] === "format") {
      format = parts[1];
    } else if (parts[0] === "element") {
      current = { name: parts[1], count: parseInt(parts[2], 10), properties: [] };
      elements.push(current);
    } else if (parts[0] === "property" && current) {
      if (parts[1] === "list") {
        current.properties.push({ list: true, countType: parts[2], type: parts[3], name: parts[4] });
      } else {
        current.properties.push({ type: parts[1], name: parts[2] });
      }
    }
  }
  return { format, elements, headerBytes: headerEnd };
}

function readScalar(view, offset, type, littleEndian) {
  switch (type) {
    case "char": case "int8": return [view.getInt8(offset), 1];
    case "uchar": case "uint8": return [view.getUint8(offset), 1];
    case "short": case "int16": return [view.getInt16(offset, littleEndian), 2];
    case "ushort": case "uint16": return [view.getUint16(offset, littleEndian), 2];
    case "int": case "int32": return [view.getInt32(offset, littleEndian), 4];
    case "uint": case "uint32": return [view.getUint32(offset, littleEndian), 4];
    case "float": case "float32": return [view.getFloat32(offset, littleEndian), 4];
    case "double": case "float64": return [view.getFloat64(offset, littleEndian), 8];
    default: throw new Error(`不支持的 PLY 属性类型: ${type}`);
  }
}

function shToColor(value) {
  return Math.min(1, Math.max(0, 0.5 + SH_C0 * value));
}

export function parsePLY(buffer) {
  const bytes = new Uint8Array(buffer);
  // 文件头按 latin1 解析，足够覆盖标准 PLY 头。
  const headText = new TextDecoder("latin1").decode(bytes.slice(0, Math.min(bytes.length, 65536)));
  const header = parseHeader(headText);
  const vertexElement = header.elements.find((element) => element.name === "vertex");
  if (!vertexElement) throw new Error("PLY 中没有 vertex 元素。");

  const count = vertexElement.count;
  const propNames = vertexElement.properties.map((property) => property.name);
  const indexOf = (name) => propNames.indexOf(name);
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  const has = (names) => names.every((name) => indexOf(name) >= 0);
  let colorMode = "gray";
  if (has(["red", "green", "blue"])) colorMode = "rgb";
  else if (has(["albedo_dc_0", "albedo_dc_1", "albedo_dc_2"])) colorMode = "albedo_dc";
  else if (has(["f_dc_0", "f_dc_1", "f_dc_2"])) colorMode = "f_dc";
  else if (has(["nx", "ny", "nz"])) colorMode = "normal";

  const xi = indexOf("x"), yi = indexOf("y"), zi = indexOf("z");
  if (xi < 0 || yi < 0 || zi < 0) throw new Error("PLY 缺少 x/y/z 属性。");

  const colorChannels =
    colorMode === "rgb" ? ["red", "green", "blue"] :
    colorMode === "albedo_dc" ? ["albedo_dc_0", "albedo_dc_1", "albedo_dc_2"] :
    colorMode === "f_dc" ? ["f_dc_0", "f_dc_1", "f_dc_2"] :
    colorMode === "normal" ? ["nx", "ny", "nz"] : null;
  const colorIsByte = colorMode === "rgb" &&
    ["red", "green", "blue"].every((name) => {
      const property = vertexElement.properties[indexOf(name)];
      return property.type === "uchar" || property.type === "uint8";
    });
  const colorIndices = colorChannels ? colorChannels.map((name) => indexOf(name)) : null;

  if (header.format === "ascii") {
    const bodyText = new TextDecoder("latin1").decode(bytes.subarray(header.headerBytes));
    const rows = bodyText.split(/\r?\n/).filter((line) => line.trim().length > 0);
    if (rows.length < count) throw new Error("PLY ascii 数据行数不足。");
    for (let i = 0; i < count; i++) {
      const values = rows[i].trim().split(/\s+/).map(Number);
      positions[i * 3] = values[xi];
      positions[i * 3 + 1] = values[yi];
      positions[i * 3 + 2] = values[zi];
      if (colorIndices) {
        const triple = colorIndices.map((propertyIndex) => values[propertyIndex] ?? 0);
        setColor(colors, i, triple, colorMode, colorIsByte);
      } else {
        colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.7;
      }
    }
  } else {
    const littleEndian = header.format === "binary_little_endian";
    const view = new DataView(buffer, header.headerBytes);
    let offset = 0;
    for (let i = 0; i < count; i++) {
      const rowStart = offset;
      const values = new Map();
      for (const property of vertexElement.properties) {
        if (property.list) {
          const [listCount, c1] = readScalar(view, offset, property.countType, littleEndian);
          offset += c1;
          const [, size] = readScalar(view, offset, property.type, littleEndian);
          offset += size * listCount;
          continue;
        }
        const [value, size] = readScalar(view, offset, property.type, littleEndian);
        values.set(property.name, value);
        offset += size;
      }
      positions[i * 3] = values.get("x");
      positions[i * 3 + 1] = values.get("y");
      positions[i * 3 + 2] = values.get("z");
      if (colorChannels) {
        const triple = colorChannels.map((name) => values.get(name));
        setColor(colors, i, triple, colorMode, colorIsByte);
      } else {
        colors[i * 3] = colors[i * 3 + 1] = colors[i * 3 + 2] = 0.7;
      }
      void rowStart;
    }
  }

  return { positions, colors, count, colorMode };
}

function setColor(colors, i, triple, colorMode, colorIsByte) {
  let [r, g, b] = triple;
  if (colorMode === "rgb") {
    if (colorIsByte) { r /= 255; g /= 255; b /= 255; }
    r = Math.min(1, Math.max(0, r)); g = Math.min(1, Math.max(0, g)); b = Math.min(1, Math.max(0, b));
  } else if (colorMode === "albedo_dc" || colorMode === "f_dc") {
    r = shToColor(r); g = shToColor(g); b = shToColor(b);
  } else if (colorMode === "normal") {
    r = Math.min(1, Math.max(0, 0.5 + 0.5 * r));
    g = Math.min(1, Math.max(0, 0.5 + 0.5 * g));
    b = Math.min(1, Math.max(0, 0.5 + 0.5 * b));
  }
  colors[i * 3] = r;
  colors[i * 3 + 1] = g;
  colors[i * 3 + 2] = b;
}
