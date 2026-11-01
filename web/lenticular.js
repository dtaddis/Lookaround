// Glasses-free 3D on SR (Simulated Reality) displays: Acer SpatialLabs, Samsung Odyssey 3D and
// other eye-tracked lenticular panels on the LeiaSR / Dimenco runtime.
//
// The panel's sub-pixels sit under a sheet of slanted lenses: through them each eye sees only
// some of the sub-pixels, which ones depending on where the eye is. The server's SR helper
// (lookaround/sr.py, at /api/sr) switches the lens on and reports, about a hundred times a
// second, where the eyes are and, for each eye, the lens phase of every sub-pixel as the
// runtime's own weaver works it out: a plane, origin + across * column + down * row, in lens
// periods, with an origin per colour. An eye sees the sub-pixels whose phase is near a whole
// number. So each frame renders one view per eye, with a frustum from that eye through the
// screen's physical rectangle (head-coupled perspective), and a shader gives each sub-pixel the
// colour of the eye that sees it best.
//
// The canvas must cover the SR screen pixel for pixel: fullscreen on it, at its native
// resolution, as the weave is worked out in the screen's own pixels.

import * as THREE from "three";

// The viewing distance the scene is scaled for: at this distance the screen is a window onto the
// view the 2D viewer shows (mm; the SR runtime's own default viewer position).
const DESIGN_DISTANCE_MM = 600;

// Reads the SR helper's stream.
export class SRStream {
  constructor(url = "/api/sr") {
    this.url = url;
    this.screen = null;  // {width_mm, height_mm, columns, rows}
    this.lens = null;
    this.eyes = null;    // [[x, y, z], [x, y, z]] mm from the screen's centre, x right, y up, z out
    this.weave = null;   // per eye: [across, down, red, green, blue]
    this.tracking = false;
    this.error = null;
    this.onchange = () => {};
  }

  open() {
    this.close();
    this.error = null;
    this.source = new EventSource(this.url);
    this.source.onmessage = (event) => {
      const message = JSON.parse(event.data);
      if (message.error) {
        this.error = message.error;
        this.close();
        this.onchange();
        return;
      }
      const first = !this.screen;
      const was = this.tracking;
      if (message.screen) this.screen = message.screen;
      if (message.lens) this.lens = message.lens;
      this.eyes = message.eyes;
      this.weave = message.weave;
      this.tracking = message.tracking;
      if (first || was !== this.tracking) this.onchange();
    };
    this.source.onerror = () => {
      if (this.source?.readyState === EventSource.CLOSED) {
        this.error = "Lost the connection to the server";
        this.onchange();
      }
    };
  }

  close() {
    this.source?.close();
    this.source = null;
  }

  get ready() {
    return !!(this.screen && this.eyes && this.weave);
  }
}

const vertexShader = /* glsl */ `
  out vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const fragmentShader = /* glsl */ `
  precision highp float;
  in vec2 vUv;
  out vec4 fragColor;
  uniform sampler2D left;
  uniform sampler2D right;
  uniform float rows;         // drawing buffer height, pixels
  uniform vec2 stepL;         // each eye's phase change a pixel across and a row down
  uniform vec2 stepR;
  uniform vec3 originL;       // each eye's phase at pixel (0, 0), per colour
  uniform vec3 originR;
  uniform float blend;        // width of the soft edge between the eyes' stripes (lens periods)
  uniform bool swapEyes;
  uniform int pattern;        // 0: the views; 1: left white, right black; 2: left red, right blue

  // How far a phase is from a whole number, 0..0.5.
  float off(float phase) { return abs(phase - floor(phase + 0.5)); }

  void main() {
    // The screen's pixel grid: column from the left, row from the top.
    vec2 pixel = vec2(floor(gl_FragCoord.x), rows - 1.0 - floor(gl_FragCoord.y));
    vec3 l = texture(left, vUv).rgb;
    vec3 r = texture(right, vUv).rgb;
    if (pattern == 1) { l = vec3(1.0); r = vec3(0.0); }
    if (pattern == 2) { l = vec3(1.0, 0.1, 0.1); r = vec3(0.1, 0.3, 1.0); }
    if (swapEyes) { vec3 t = l; l = r; r = t; }
    float baseL = dot(stepL, pixel);
    float baseR = dot(stepR, pixel);
    vec3 colour;
    for (int c = 0; c < 3; c++) {
      float dl = off(originL[c] + baseL);
      float dr = off(originR[c] + baseR);
      colour[c] = mix(l[c], r[c], smoothstep(-blend, blend, dl - dr));
    }
    fragColor = vec4(colour, 1.0);
  }
`;

export class LenticularRenderer {
  constructor(renderer) {
    this.renderer = renderer;
    this.pattern = 0;
    this.swapEyes = false;
    this.blend = 0.04;
    this.renderScale = 0.5;  // of the screen's resolution, for each eye's view
    this.screenDepth = 1;    // where the screen sits, as a multiple of the distance to the target
    this.targets = [0, 1].map(() => new THREE.WebGLRenderTarget(1, 1, { depthBuffer: true }));
    this.cameras = [new THREE.PerspectiveCamera(), new THREE.PerspectiveCamera()];
    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader,
      uniforms: {
        left: { value: this.targets[0].texture },
        right: { value: this.targets[1].texture },
        rows: { value: 1 },
        stepL: { value: new THREE.Vector2() },
        stepR: { value: new THREE.Vector2() },
        originL: { value: new THREE.Vector3() },
        originR: { value: new THREE.Vector3() },
        blend: { value: 0 },
        swapEyes: { value: false },
        pattern: { value: 0 },
      },
      depthTest: false,
      depthWrite: false,
    });
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.material);
    this.quad.frustumCulled = false;
    this.quadScene = new THREE.Scene();
    this.quadScene.add(this.quad);
    this.quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  }

  // Renders camera's view as seen through the SR screen by the eyes stream reports, and weaves.
  render(scene, camera, spark, targetDistance, stream) {
    const renderer = this.renderer;
    const size = renderer.getDrawingBufferSize(new THREE.Vector2());
    const width = Math.max(1, Math.round(size.x * this.renderScale));
    const height = Math.max(1, Math.round(size.y * this.renderScale));
    for (const target of this.targets) {
      if (target.width !== width || target.height !== height) target.setSize(width, height);
    }
    const widthMm = stream.screen.width_mm, heightMm = stream.screen.height_mm;

    // The screen is a window in front of the camera: its centre lies on the camera's axis at
    // screenDepth times the target distance, and a millimetre there is k world units. An eye at
    // the design distance, straight in front, is where the camera is.
    const screenDistance = targetDistance * this.screenDepth;
    const k = screenDistance / DESIGN_DISTANCE_MM;
    camera.updateMatrixWorld();
    const rotation = camera.getWorldQuaternion(new THREE.Quaternion());
    for (let i = 0; i < 2; i++) {
      const [x, y, z] = stream.eyes[i];
      const local = new THREE.Vector3(x * k, y * k, Math.max(z, 50) * k - screenDistance);
      const toScreen = Math.max(z, 50) * k;
      const near = Math.min(camera.near, toScreen * 0.5);
      const scale = near / toScreen;
      const cam = this.cameras[i];
      cam.position.copy(local).applyMatrix4(camera.matrixWorld);
      cam.quaternion.copy(rotation);
      cam.near = near;
      cam.far = camera.far;
      cam.updateMatrixWorld();
      cam.projectionMatrix.makePerspective(
        (-widthMm / 2 * k - local.x) * scale, (widthMm / 2 * k - local.x) * scale,
        (heightMm / 2 * k - local.y) * scale, (-heightMm / 2 * k - local.y) * scale,
        near, camera.far);
      cam.projectionMatrixInverse.copy(cam.projectionMatrix).invert();
      // Sort the splats for one eye only: the other is a few centimetres away.
      if (spark) spark.autoUpdate = i === 0;
      renderer.setRenderTarget(this.targets[i]);
      renderer.render(scene, cam);
    }
    if (spark) spark.autoUpdate = true;
    renderer.setRenderTarget(null);

    const u = this.material.uniforms;
    const [l, r] = stream.weave;
    u.rows.value = size.y;
    u.stepL.value.set(l[0], l[1]);
    u.stepR.value.set(r[0], r[1]);
    u.originL.value.set(l[2], l[3], l[4]);
    u.originR.value.set(r[2], r[3], r[4]);
    u.blend.value = this.blend;
    u.swapEyes.value = this.swapEyes;
    u.pattern.value = this.pattern;
    renderer.render(this.quadScene, this.quadCamera);
  }
}

// Fullscreen on the SR display if the browser can tell which screen that is (it asks first),
// else on the screen the window is on.
export async function enterFullscreen(element, stream) {
  let options;
  if (window.getScreenDetails && stream.screen) {
    try {
      const { screens } = await window.getScreenDetails();
      const native = (s) => Math.round(s.width * s.devicePixelRatio) === stream.screen.columns &&
                            Math.round(s.height * s.devicePixelRatio) === stream.screen.rows;
      const named = (s) => /spatial ?labs|odyssey ?3d|\bSR\b|PSV27|leia/i.test(s.label || "");
      const pick = screens.find(named) || (native(window.screen) ? null : screens.find(native));
      if (pick) options = { screen: pick };
    } catch {
      // permission refused: use the window's own screen
    }
  }
  await element.requestFullscreen(options);
}
