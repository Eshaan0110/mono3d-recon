/**
 * Three.js 3D Scene Viewer
 * Handles rendering, camera controls, and model loading.
 */

class SceneViewer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        // Scene objects
        this.meshObject = null;
        this.pointsObject = null;
        this.wireframeObject = null;
        this.trajectoryLine = null;
        this.gridHelper = null;

        // State
        this.displayMode = 'mesh'; // 'mesh', 'points', 'wireframe'
        this.pointSize = 2.0;
        this.showTrajectory = false;
        this.modelLoaded = false;
        this.modelUrl = null;

        this.init();
    }

    init() {
        // Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x0a0e17);

        // Camera
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(60, aspect, 0.01, 1000);
        this.camera.position.set(2, 2, 2);

        // Renderer
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            preserveDrawingBuffer: true // for screenshots
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.outputEncoding = THREE.sRGBEncoding;

        // Controls
        this.controls = new THREE.OrbitControls(this.camera, this.canvas);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.08;
        this.controls.minDistance = 0.1;
        this.controls.maxDistance = 100;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
        this.scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
        dirLight.position.set(5, 10, 7);
        this.scene.add(dirLight);

        const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.3);
        dirLight2.position.set(-5, -2, -7);
        this.scene.add(dirLight2);

        // Grid
        this.gridHelper = new THREE.GridHelper(10, 20, 0x1e293b, 0x1e293b);
        this.scene.add(this.gridHelper);

        // Resize handler
        window.addEventListener('resize', () => this.onResize());

        // Animation loop
        this.animate();
    }

    animate() {
        requestAnimationFrame(() => this.animate());
        this.controls.update();
        this.renderer.render(this.scene, this.camera);
    }

    onResize() {
        const container = this.canvas.parentElement;
        const w = container.clientWidth;
        const h = container.clientHeight;

        this.camera.aspect = w / h;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(w, h);
    }

    /**
     * Load a GLB/GLTF model from a URL.
     */
    loadGLB(url) {
        const loader = new THREE.GLTFLoader();
        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (gltf) => {
                    this.clearScene();

                    const model = gltf.scene;

                    // Center and scale model
                    const box = new THREE.Box3().setFromObject(model);
                    const center = box.getCenter(new THREE.Vector3());
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 3 / maxDim;

                    model.position.sub(center);
                    model.scale.multiplyScalar(scale);

                    // Store as mesh object
                    this.meshObject = model;
                    this.scene.add(model);

                    // Create wireframe version
                    this.createWireframe(model);

                    // Fit camera
                    this.fitCamera(box, scale, center);

                    this.modelLoaded = true;
                    this.modelUrl = url;
                    this.setDisplayMode('mesh');
                    resolve(model);
                },
                undefined,
                (error) => {
                    console.error('Error loading GLB:', error);
                    reject(error);
                }
            );
        });
    }

    /**
     * Load a PLY point cloud from a URL.
     */
    loadPLY(url) {
        const loader = new THREE.PLYLoader();
        return new Promise((resolve, reject) => {
            loader.load(
                url,
                (geometry) => {
                    this.clearScene();

                    // Center geometry
                    geometry.computeBoundingBox();
                    const center = geometry.boundingBox.getCenter(new THREE.Vector3());
                    geometry.translate(-center.x, -center.y, -center.z);

                    const size = geometry.boundingBox.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    const scale = 3 / maxDim;

                    // Point cloud material
                    const material = new THREE.PointsMaterial({
                        size: this.pointSize * 0.01,
                        vertexColors: geometry.hasAttribute('color'),
                        sizeAttenuation: true,
                    });

                    if (!geometry.hasAttribute('color')) {
                        material.color = new THREE.Color(0x3b82f6);
                    }

                    const points = new THREE.Points(geometry, material);
                    points.scale.multiplyScalar(scale);

                    this.pointsObject = points;
                    this.scene.add(points);

                    this.fitCamera(geometry.boundingBox, scale, center);

                    this.modelLoaded = true;
                    this.setDisplayMode('points');
                    resolve(points);
                },
                undefined,
                (error) => {
                    console.error('Error loading PLY:', error);
                    reject(error);
                }
            );
        });
    }

    /**
     * Load a model file (auto-detect format from File object).
     */
    loadFile(file) {
        const url = URL.createObjectURL(file);
        const name = file.name.toLowerCase();

        if (name.endsWith('.glb') || name.endsWith('.gltf')) {
            return this.loadGLB(url);
        } else if (name.endsWith('.ply')) {
            return this.loadPLY(url);
        } else {
            return Promise.reject(new Error('Unsupported format. Use .glb or .ply'));
        }
    }

    createWireframe(model) {
        this.wireframeObject = model.clone();
        this.wireframeObject.traverse((child) => {
            if (child.isMesh) {
                child.material = new THREE.MeshBasicMaterial({
                    color: 0x3b82f6,
                    wireframe: true,
                    transparent: true,
                    opacity: 0.6,
                });
            }
        });
        this.wireframeObject.visible = false;
        this.scene.add(this.wireframeObject);
    }

    /**
     * Set the camera trajectory visualization.
     */
    setTrajectory(positions) {
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
        }

        if (!positions || positions.length < 2) return;

        const points = positions.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
            color: 0xf59e0b,
            linewidth: 2,
        });

        this.trajectoryLine = new THREE.Line(geometry, material);
        this.trajectoryLine.visible = this.showTrajectory;
        this.scene.add(this.trajectoryLine);

        // Add camera frustum markers
        const frustumGeo = new THREE.ConeGeometry(0.02, 0.05, 4);
        const frustumMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });

        for (let i = 0; i < points.length; i += Math.max(1, Math.floor(points.length / 20))) {
            const marker = new THREE.Mesh(frustumGeo, frustumMat);
            marker.position.copy(points[i]);
            this.trajectoryLine.add(marker);
        }
    }

    setDisplayMode(mode) {
        this.displayMode = mode;

        if (this.meshObject) {
            this.meshObject.visible = (mode === 'mesh');
        }
        if (this.pointsObject) {
            this.pointsObject.visible = (mode === 'points');
        }
        if (this.wireframeObject) {
            this.wireframeObject.visible = (mode === 'wireframe');
        }

        document.getElementById('point-size-group').style.display =
            mode === 'points' ? 'block' : 'none';
    }

    setPointSize(size) {
        this.pointSize = size;
        if (this.pointsObject) {
            this.pointsObject.material.size = size * 0.01;
        }
    }

    setBackground(type) {
        switch (type) {
            case 'dark':
                this.scene.background = new THREE.Color(0x0a0e17);
                this.gridHelper.visible = true;
                break;
            case 'light':
                this.scene.background = new THREE.Color(0xf1f5f9);
                this.gridHelper.visible = true;
                break;
            case 'gradient':
                this.scene.background = new THREE.Color(0x111827);
                this.gridHelper.visible = false;
                break;
        }
    }

    toggleTrajectory(show) {
        this.showTrajectory = show;
        if (this.trajectoryLine) {
            this.trajectoryLine.visible = show;
        }
    }

    fitCamera(bbox, scale, center) {
        const size = bbox.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) * scale;
        const dist = maxDim * 1.5;

        this.camera.position.set(dist, dist * 0.8, dist);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    resetCamera() {
        this.camera.position.set(2, 2, 2);
        this.camera.lookAt(0, 0, 0);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    takeScreenshot() {
        this.renderer.render(this.scene, this.camera);
        const dataUrl = this.renderer.domElement.toDataURL('image/png');

        const link = document.createElement('a');
        link.download = 'mono3d-screenshot.png';
        link.href = dataUrl;
        link.click();
    }

    clearScene() {
        if (this.meshObject) {
            this.scene.remove(this.meshObject);
            this.meshObject = null;
        }
        if (this.pointsObject) {
            this.scene.remove(this.pointsObject);
            this.pointsObject = null;
        }
        if (this.wireframeObject) {
            this.scene.remove(this.wireframeObject);
            this.wireframeObject = null;
        }
        if (this.trajectoryLine) {
            this.scene.remove(this.trajectoryLine);
            this.trajectoryLine = null;
        }
        this.modelLoaded = false;
    }
}
