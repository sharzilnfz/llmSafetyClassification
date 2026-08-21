/**
 * AegisGuard AI - Minimalist Scroll-Driven 3D Visualization Engine
 * Style: shadcn/ui Minimal Geometric Architecture (Zero Blobs, Smooth Scroll Inertia)
 */

class Aegis3DScene {
  constructor(canvasId = 'threeBgCanvas') {
    this.canvasId = canvasId;
    this.container = document.getElementById(canvasId);
    if (!this.container) return;

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.animationFrameId = null;

    // Scroll & Mouse Inertia State
    this.scrollProgress = 0;
    this.targetScrollProgress = 0;
    this.mouse = { x: 0, y: 0, targetX: 0, targetY: 0 };
    this.isDragging = false;
    this.previousMousePosition = { x: 0, y: 0 };

    // Scene Meshes
    this.gridGroup = null;
    this.constellationGroup = null;
    this.probeRay = null;
    this.probeTarget = null;
    this.targetCoord = new THREE.Vector3(-3.2, 1.8, 1.2);

    // Active state
    this.currentClassId = 0;
    this.targetColor = new THREE.Color(0x22c55e);

    this.init();
  }

  init() {
    const width = window.innerWidth;
    const height = window.innerHeight;

    // 1. Scene & Camera
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x09090b, 0.045);

    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000);
    this.camera.position.set(0, 8, 22);
    this.camera.lookAt(0, 0, 0);

    // 2. Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.container,
      antialias: true,
      alpha: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    // 3. Minimal Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(10, 20, 15);
    this.scene.add(dirLight);

    // 4. Build Minimalist 3D Geometry
    this.createMinimalGrid();
    this.createLatentConstellation();
    this.createProbeRay();

    // 5. Setup Listeners
    this.setupEvents();

    // 6. Start Loop
    this.animate();
  }

  createMinimalGrid() {
    this.gridGroup = new THREE.Group();

    // Sleek flat coordinate plane
    const gridHelper = new THREE.GridHelper(30, 30, 0x27272a, 0x18181b);
    gridHelper.position.y = -4;
    this.gridGroup.add(gridHelper);

    // Subtle boundary wireframe box
    const boxGeo = new THREE.BoxGeometry(16, 10, 16);
    const boxEdges = new THREE.EdgesGeometry(boxGeo);
    const boxMat = new THREE.LineBasicMaterial({ color: 0x27272a, transparent: true, opacity: 0.4 });
    const wireBox = new THREE.LineSegments(boxEdges, boxMat);
    wireBox.position.y = 1;
    this.gridGroup.add(wireBox);

    this.scene.add(this.gridGroup);
  }

  createLatentConstellation() {
    this.constellationGroup = new THREE.Group();

    // 4 Taxonomy Cluster Centers in 3D Embedding Space
    const clusters = [
      { id: 0, center: [-4, 2, 2], color: 0x22c55e, count: 35 },  // Benign Vanilla
      { id: 1, center: [-3, -2, -2], color: 0xf59e0b, count: 35 }, // Benign Adversarial
      { id: 2, center: [4, 2, -2], color: 0xf97316, count: 35 },   // Harmful Vanilla
      { id: 3, center: [4, -2, 3], color: 0xef4444, count: 35 }    // Harmful Adversarial
    ];

    const pointGeo = new THREE.BufferGeometry();
    const positions = [];
    const colors = [];

    clusters.forEach(c => {
      const clusterColor = new THREE.Color(c.color);
      for (let i = 0; i < c.count; i++) {
        const x = c.center[0] + (Math.random() - 0.5) * 2.8;
        const y = c.center[1] + (Math.random() - 0.5) * 2.2;
        const z = c.center[2] + (Math.random() - 0.5) * 2.8;

        positions.push(x, y, z);
        colors.push(clusterColor.r, clusterColor.g, clusterColor.b);
      }
    });

    pointGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    pointGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const pointMat = new THREE.PointsMaterial({
      size: 0.16,
      vertexColors: true,
      transparent: true,
      opacity: 0.65
    });

    const points = new THREE.Points(pointGeo, pointMat);
    this.constellationGroup.add(points);

    // Add minimal cluster focal nodes
    clusters.forEach(c => {
      const nodeGeo = new THREE.SphereGeometry(0.22, 16, 16);
      const nodeMat = new THREE.MeshBasicMaterial({ color: c.color, wireframe: true });
      const nodeMesh = new THREE.Mesh(nodeGeo, nodeMat);
      nodeMesh.position.set(...c.center);
      this.constellationGroup.add(nodeMesh);
    });

    this.scene.add(this.constellationGroup);
  }

  createProbeRay() {
    // Real-time animated laser vector from ground to inspected prompt node
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x22c55e,
      transparent: true,
      opacity: 0.8,
      linewidth: 1.5
    });

    const points = [
      new THREE.Vector3(0, -4, 0),
      this.targetCoord.clone()
    ];
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    this.probeRay = new THREE.Line(lineGeo, lineMat);
    this.scene.add(this.probeRay);

    // Target pointer sphere
    const targetGeo = new THREE.SphereGeometry(0.28, 16, 16);
    const targetMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
    this.probeTarget = new THREE.Mesh(targetGeo, targetMat);
    this.probeTarget.position.copy(this.targetCoord);
    this.scene.add(this.probeTarget);
  }

  updateState(classId, colorHex, coords) {
    this.currentClassId = classId;
    this.targetColor = new THREE.Color(colorHex);

    if (coords && coords.length === 3) {
      this.targetCoord.set(coords[0], coords[1], coords[2]);
    }

    if (this.probeTarget) {
      this.probeTarget.material.color.copy(this.targetColor);
    }
    if (this.probeRay) {
      this.probeRay.material.color.copy(this.targetColor);
    }
  }

  setupEvents() {
    // 1. Scroll-driven camera and perspective animation
    window.addEventListener('scroll', () => {
      const maxScroll = Math.max(
        document.body.scrollHeight - window.innerHeight,
        1
      );
      this.targetScrollProgress = Math.min(Math.max(window.scrollY / maxScroll, 0), 1);
    }, { passive: true });

    // 2. Mouse Parallax
    window.addEventListener('mousemove', (e) => {
      this.mouse.targetX = (e.clientX / window.innerWidth - 0.5) * 2;
      this.mouse.targetY = -(e.clientY / window.innerHeight - 0.5) * 2;
    }, { passive: true });

    // 3. Responsive Resize
    window.addEventListener('resize', () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      this.camera.aspect = width / height;
      this.camera.updateProjectionMatrix();
      this.renderer.setSize(width, height);
    });
  }

  animate() {
    this.animationFrameId = requestAnimationFrame(() => this.animate());

    // Smooth Scroll Progress Interpolation (Inertia)
    this.scrollProgress += (this.targetScrollProgress - this.scrollProgress) * 0.06;

    // Smooth Mouse Interpolation
    this.mouse.x += (this.mouse.targetX - this.mouse.x) * 0.05;
    this.mouse.y += (this.mouse.targetY - this.mouse.y) * 0.05;

    // Scroll-Driven 3D Camera Trajectory:
    // Scroll progress 0.0 -> Top (Hero & Sandbox)
    // Scroll progress 0.5 -> Middle (Leaderboard)
    // Scroll progress 1.0 -> Bottom (Taxonomy & Stress test)
    const angle = this.scrollProgress * Math.PI * 0.85 + this.mouse.x * 0.15;
    const radius = 20 - this.scrollProgress * 4;
    const camY = 7 + this.scrollProgress * 6 + this.mouse.y * 1.5;

    this.camera.position.x = Math.sin(angle) * radius;
    this.camera.position.z = Math.cos(angle) * radius;
    this.camera.position.y = camY;
    this.camera.lookAt(0, -0.5, 0);

    // Subtle idle rotation of geometric structures
    if (this.gridGroup) {
      this.gridGroup.rotation.y = this.scrollProgress * 0.2;
    }
    if (this.constellationGroup) {
      this.constellationGroup.rotation.y += 0.0015;
    }

    // Smooth lerp of probe target sphere & line ray
    if (this.probeTarget && this.probeRay) {
      this.probeTarget.position.lerp(this.targetCoord, 0.08);

      const positions = this.probeRay.geometry.attributes.position.array;
      positions[3] = this.probeTarget.position.x;
      positions[4] = this.probeTarget.position.y;
      positions[5] = this.probeTarget.position.z;
      this.probeRay.geometry.attributes.position.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);
  }
}

// Global Export
window.Aegis3DScene = Aegis3DScene;
