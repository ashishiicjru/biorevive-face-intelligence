/* 3D scenes (Three.js r149, vendored): Holo.head / Holo.cloud / Holo.relief */
'use strict';
const Holo = (() => {
  const T = () => window.THREE;
  const gauss = (x, y, sx, sy) => Math.exp(-((x / sx) ** 2 + (y / sy) ** 2));
  const smooth = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

  function setup(canvas, { fov = 38, z = 4.2, alpha = true } = {}) {
    const THREE = T();
    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(fov, 1, .1, 100); camera.position.set(0, 0, z);
    const fit = () => {
      const w = canvas.clientWidth || 300, h = canvas.clientHeight || 300;
      if (canvas._w !== w || canvas._h !== h) { renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix(); canvas._w = w; canvas._h = h; }
    };
    return { THREE, renderer, scene, camera, fit };
  }

  function loop(fn, canvas) {
    let t0 = performance.now(), on = true;
    const tick = () => {
      if (!on) return;
      const visible = canvas.offsetParent !== null && !document.hidden;
      if (visible) { const t = performance.now(); fn((t - t0) / 1000, (t - t0)); }
      requestAnimationFrame(tick);
    };
    tick(); return () => { on = false; };
  }

  /* ---- procedural head grid ---- */
  function headGrid(nLat = 60, nLon = 88) {
    const P = [], idx = [];
    for (let i = 0; i <= nLat; i++) {
      const phi = i / nLat * Math.PI, y = Math.cos(phi), r = Math.sin(phi);
      for (let j = 0; j < nLon; j++) {
        const th = j / nLon * 2 * Math.PI, x = r * Math.sin(th), z = r * Math.cos(th);
        let X = x * .78, Y = y * 1.06, Z = z * .9;
        if (Y < 0) { const t = smooth(0, 1.0, -Y); X *= 1 - .24 * t; Z *= 1 - .10 * t; }
        const fw = smooth(.05, .55, z);                         // front-of-face weight
        const ax = Math.abs(X);
        let dz = 0;
        dz += .40 * gauss(X, Y + .02, .085, .27) * fw + .10 * gauss(X, Y + .17, .17, .07) * fw;   // nose (bridge + tip/wings)
        dz += .10 * gauss(Y - .31, 0, .07, 1) * gauss(X, 0, .55, 1) * fw; // brow ridge
        dz -= .12 * gauss(ax - .30, Y - .15, .13, .08) * fw;      // eye sockets
        dz += .05 * gauss(ax - .48, Y + .12, .16, .17) * fw;       // cheekbones
        dz += .11 * gauss(X, Y + .47, .22, .055) * fw;             // lips
        dz -= .035 * gauss(X, Y + .525, .25, .015) * fw;           // mouth line
        dz += .12 * gauss(X, Y + .90, .20, .10) * fw;              // chin
        Z += dz;
        const side = 1 - fw, ear = gauss(Y - .02, 0, .22, 1) * gauss(Math.abs(z), 0, .35, 1) * side; // ears
        X += Math.sign(X || 1) * .09 * ear * (ax > .55 ? 1 : 0);
        P.push(X, Y, Z);
      }
    }
    for (let i = 0; i <= nLat; i++) for (let j = 0; j < nLon; j++) {
      const a = i * nLon + j, b = i * nLon + (j + 1) % nLon;
      idx.push(a, b); if (i < nLat) idx.push(a, (i + 1) * nLon + j);
    }
    const tri = [];
    for (let i = 0; i < nLat; i++) for (let j = 0; j < nLon; j++) {
      const a = i * nLon + j, b = i * nLon + (j + 1) % nLon, c = (i + 1) * nLon + j, d = (i + 1) * nLon + (j + 1) % nLon;
      tri.push(a, c, b, b, c, d);
    }
    return { P: new Float32Array(P), idx, tri, nLon };
  }

  function head(canvas, opts = {}) {
    const { THREE, renderer, scene, camera, fit } = setup(canvas, { z: opts.z || 4.6 });
    camera.position.y = .15;
    const { P, idx, tri } = headGrid();
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(P, 3)); g.setIndex(idx);
    const col = new THREE.Color(opts.color || 0x22d3ee), col2 = new THREE.Color(opts.color2 || 0x8b5cf6);
    const lines = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: .34, blending: THREE.AdditiveBlending, depthWrite: false }));
    lines.material.opacity = .42;
    const uni = { uScan: { value: -1 }, uC1: { value: col }, uC2: { value: col2 } };
    const pm = new THREE.ShaderMaterial({
      uniforms: uni, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: 'uniform float uScan;varying float vG;void main(){vec4 mv=modelViewMatrix*vec4(position,1.);float d=abs(position.y-uScan);vG=exp(-d*d*38.);gl_PointSize=(2.2+vG*3.2)*(3.6/-mv.z);gl_Position=projectionMatrix*mv;}',
      fragmentShader: 'uniform vec3 uC1;uniform vec3 uC2;varying float vG;void main(){float r=length(gl_PointCoord-.5)*2.;if(r>1.)discard;float a=(1.-r*r);vec3 c=mix(uC1,uC2,vG);gl_FragColor=vec4(c*(.5+vG*.9),a*(.5+vG*.5));}'
    });
    const pts = new THREE.Points(g, pm);
    const sg = new THREE.BufferGeometry(); sg.setAttribute('position', new THREE.BufferAttribute(P, 3)); sg.setIndex(tri); sg.computeVertexNormals();
    const solid = new THREE.Mesh(sg, new THREE.ShaderMaterial({
      uniforms: { uC1: { value: col }, uC2: { value: col2 } }, side: THREE.DoubleSide, transparent: true, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
      vertexShader: 'varying vec3 vN;varying vec3 vV;varying float vY;void main(){vec4 mv=modelViewMatrix*vec4(position,1.);vN=normalize(normalMatrix*normal);vV=normalize(-mv.xyz);vY=position.y;gl_Position=projectionMatrix*mv;}',
      fragmentShader: 'uniform vec3 uC1;uniform vec3 uC2;varying vec3 vN;varying vec3 vV;varying float vY;void main(){float d=abs(dot(normalize(vN),normalize(vV)));float rim=pow(1.-d,2.1);vec3 base=vec3(.012,.05,.085)+uC2*.05;vec3 c=base+mix(uC1,uC2,.5+.5*sin(vY*3.))*rim*.85+uC1*pow(d,6.)*.07;gl_FragColor=vec4(c,.93);}'
    }));
    lines.material.depthTest = true; pm.depthTest = true;
    solid.scale.setScalar(.982); solid.renderOrder = 0; lines.renderOrder = 1; pts.renderOrder = 2;
    const headG = new THREE.Group(); headG.add(solid, lines, pts);
    // eyes
    const eyeM = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: .75, blending: THREE.AdditiveBlending });
    const nLonG = 88;
    const surf = (tx, ty) => { let best = 1e9, bi = 0; for (let i = 0; i < P.length / 3; i++) { if (P[i * 3 + 2] < .3) continue; const d = (P[i * 3] - tx) ** 2 + (P[i * 3 + 1] - ty) ** 2; if (d < best) { best = d; bi = i; } } return [P[bi * 3], P[bi * 3 + 1], P[bi * 3 + 2]]; };
    for (const s of [-1, 1]) { const [ex, ey, ez] = surf(s * .30, .15); const e = new THREE.Mesh(new THREE.SphereGeometry(.03, 12, 12), eyeM); e.position.set(ex, ey, ez + .02); headG.add(e);
      const r = new THREE.Mesh(new THREE.RingGeometry(.055, .062, 24), new THREE.MeshBasicMaterial({ color: col, side: THREE.DoubleSide, transparent: true, opacity: .8, blending: THREE.AdditiveBlending })); r.position.set(ex, ey, ez + .025); headG.add(r); }
    // neck + shoulders hint
    const neck = new THREE.Mesh(new THREE.CylinderGeometry(.33, .40, .85, 28, 9, true), new THREE.MeshBasicMaterial({ color: col, wireframe: true, transparent: true, opacity: .16, blending: THREE.AdditiveBlending, depthWrite: false }));
    neck.position.y = -1.32; headG.add(neck);
    const sh = new THREE.Mesh(new THREE.SphereGeometry(1.25, 30, 12, 0, 6.2832, 1.2, .55), new THREE.MeshBasicMaterial({ color: col2, wireframe: true, transparent: true, opacity: .13, blending: THREE.AdditiveBlending, depthWrite: false }));
    sh.scale.set(1.5, .55, .8); sh.position.y = -2.15; headG.add(sh);
    // scan ring + orbit rings + particles
    const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 96 }, (_, i) => new THREE.Vector3(Math.cos(i / 96 * 6.2832) * .98, 0, Math.sin(i / 96 * 6.2832) * .98))),
      new THREE.LineBasicMaterial({ color: 0x9ef3ff, transparent: true, opacity: .9, blending: THREE.AdditiveBlending }));
    headG.add(ring);
    const orbit = new THREE.Group();
    [[1.55, .0, .35], [1.78, 1.1, -.5], [2.0, .5, .9]].forEach(([r, rx, rz], k) => {
      const m = new THREE.Mesh(new THREE.TorusGeometry(r, .004, 6, 160), new THREE.MeshBasicMaterial({ color: k % 2 ? col2 : col, transparent: true, opacity: .35, blending: THREE.AdditiveBlending }));
      m.rotation.set(rx, 0, rz); m.userData.sp = .1 + k * .07; orbit.add(m);
    });
    scene.add(orbit); scene.add(headG);
    const N = 260, pp = new Float32Array(N * 3);
    for (let i = 0; i < N; i++) { const r = 1.6 + Math.random() * 1.6, a = Math.random() * 6.28, b = (Math.random() - .5) * 3.2; pp[i * 3] = Math.cos(a) * r; pp[i * 3 + 1] = b; pp[i * 3 + 2] = Math.sin(a) * r; }
    const pg = new THREE.BufferGeometry(); pg.setAttribute('position', new THREE.BufferAttribute(pp, 3));
    const dust = new THREE.Points(pg, new THREE.PointsMaterial({ size: .022, color: 0x9bc4ff, transparent: true, opacity: .55, blending: THREE.AdditiveBlending, depthWrite: false })); scene.add(dust);
    let px = 0, py = 0, dragging = false, ry = 0, rx = 0, lx = 0, ly = 0;
    const rect = () => canvas.getBoundingClientRect();
    window.addEventListener('pointermove', e => { const r = rect(); px = ((e.clientX - r.left) / r.width - .5) * 2; py = ((e.clientY - r.top) / r.height - .5) * 2; if (dragging) { ry += (e.clientX - lx) * .008; rx += (e.clientY - ly) * .006; lx = e.clientX; ly = e.clientY; } });
    canvas.addEventListener('pointerdown', e => { dragging = true; lx = e.clientX; ly = e.clientY; canvas.setPointerCapture?.(e.pointerId); });
    window.addEventListener('pointerup', () => dragging = false);
    return loop((t) => {
      fit(); uni.uScan.value = opts.scan ?? Math.sin(t * .9) * .95;
      ring.position.y = uni.uScan.value; ring.scale.setScalar(Math.sqrt(Math.max(.05, 1 - (uni.uScan.value / 1.06) ** 2)));
      const base = opts.angle != null ? opts.angle : (dragging ? ry : Math.sin(t * .45) * .55 + px * .25 + ry * .6);
      headG.rotation.y += (base - headG.rotation.y) * .08;
      headG.rotation.x += ((opts.pitch != null ? opts.pitch : (dragging ? rx : py * .12 + rx * .6)) - headG.rotation.x) * .08;
      headG.position.y = Math.sin(t * 1.1) * .03;
      orbit.children.forEach(m => m.rotation.z += m.userData.sp * .01); orbit.rotation.y = t * .08; dust.rotation.y = -t * .03;
      renderer.render(scene, camera);
    }, canvas);
  }

  /* ---- live landmark cloud / head avatar ---- */
  function cloud(canvas, conn, opts = {}) {
    const { THREE, renderer, scene, camera, fit } = setup(canvas, { fov: 34, z: opts.z || 3.6 });
    const N = 478, pos = new Float32Array(N * 3);
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    const pts = new THREE.Points(g, new THREE.PointsMaterial({ size: .022, color: 0x7ee8ff, transparent: true, opacity: .95, blending: THREE.AdditiveBlending, depthWrite: false }));
    const edges = conn.FACE_LANDMARKS_TESSELATION, ei = []; edges.forEach(([a, b]) => ei.push(a, b));
    const lg = new THREE.BufferGeometry(); lg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); lg.setIndex(ei);
    const mesh = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: 0x4f8cff, transparent: true, opacity: .28, blending: THREE.AdditiveBlending, depthWrite: false }));
    const cont = [];['FACE_LANDMARKS_FACE_OVAL', 'FACE_LANDMARKS_LIPS', 'FACE_LANDMARKS_LEFT_EYE', 'FACE_LANDMARKS_RIGHT_EYE', 'FACE_LANDMARKS_LEFT_EYEBROW', 'FACE_LANDMARKS_RIGHT_EYEBROW', 'FACE_LANDMARKS_NOSE'].forEach(k => conn[k].forEach(([a, b]) => cont.push(a, b)));
    const cg = new THREE.BufferGeometry(); cg.setAttribute('position', new THREE.BufferAttribute(pos, 3)); cg.setIndex(cont);
    const contour = new THREE.LineSegments(cg, new THREE.LineBasicMaterial({ color: 0xc4a3ff, transparent: true, opacity: .95 }));
    const grp = new THREE.Group(); grp.add(mesh, contour, pts); scene.add(grp);
    let has = false, drag = false, ay = 0, ax = 0, lx = 0, ly = 0, auto = opts.auto !== false;
    canvas.addEventListener('pointerdown', e => { drag = true; lx = e.clientX; ly = e.clientY; auto = false; });
    window.addEventListener('pointerup', () => drag = false);
    window.addEventListener('pointermove', e => { if (drag) { ay += (e.clientX - lx) * .01; ax += (e.clientY - ly) * .01; lx = e.clientX; ly = e.clientY; } });
    const stop = loop((t) => {
      fit(); if (auto) ay = Math.sin(t * .6) * .6;
      grp.rotation.y += (ay - grp.rotation.y) * .1; grp.rotation.x += (ax - grp.rotation.x) * .1;
      grp.visible = has; renderer.render(scene, camera);
    }, canvas);
    return {
      stop, setAuto: v => { auto = v; if (v) { ax = 0; } },
      update(lm, vw, vh) {
        if (!lm) { has = false; return; }
        has = true; const a = vw / vh;
        let cx = 0, cy = 0, cz = 0; for (const p of lm) { cx += p.x; cy += p.y; cz += p.z; } cx /= lm.length; cy /= lm.length; cz /= lm.length;
        for (let i = 0; i < N && i < lm.length; i++) { const p = lm[i]; pos[i * 3] = -(p.x - cx) * a * 3.4; pos[i * 3 + 1] = -(p.y - cy) * 3.4; pos[i * 3 + 2] = -(p.z - cz) * a * 3.4; }
        g.attributes.position.needsUpdate = lg.attributes.position.needsUpdate = cg.attributes.position.needsUpdate = true;
      }
    };
  }

  /* ---- heightmap relief of an image (3D image) ---- */
  function relief(canvas, src, opts = {}) {
    const { THREE, renderer, scene, camera, fit } = setup(canvas, { fov: 35, z: 4.5 });
    const img = new Image(); let mesh = null, wire = null, grp = new THREE.Group(); scene.add(grp);
    scene.add(new THREE.AmbientLight(0xffffff, .55)); const dl = new THREE.DirectionalLight(0xbfe6ff, 1.1); dl.position.set(-1.5, 2, 3); scene.add(dl);
    let lastSrc = null; const api = { stop: null, setSrc, setAmp: v => { opts.amp = v; if (lastSrc) setSrc(lastSrc); } };
    function setSrc(s) {
      lastSrc = s;
      img.onload = () => {
        const S = 96, c = document.createElement('canvas'); c.width = c.height = S; const x = c.getContext('2d'); x.drawImage(img, 0, 0, S, S);
        const d = x.getImageData(0, 0, S, S).data;
        if (mesh) { grp.remove(mesh); mesh.geometry.dispose(); } if (wire) grp.remove(wire);
        const geo = new THREE.PlaneGeometry(2.1, 2.1, S - 1, S - 1), p = geo.attributes.position, amp = opts.amp || .38;
        for (let i = 0; i < p.count; i++) { const u = i % S, v = Math.floor(i / S), o = (v * S + u) * 4; const L = (d[o] * .3 + d[o + 1] * .59 + d[o + 2] * .11) / 255; p.setZ(i, (L - .45) * amp * (opts.invert ? -1 : 1)); }
        geo.computeVertexNormals();
        const tex = new THREE.Texture(img); tex.needsUpdate = true;
        mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ map: tex, roughness: .78, metalness: .05, side: THREE.DoubleSide }));
        wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: 0x22d3ee, transparent: true, opacity: .06, blending: THREE.AdditiveBlending }));
        grp.add(mesh, wire);
      }; img.src = s;
    }
    if (src) setSrc(src);
    let ry = .35, rx = -.15, drag = false, lx = 0, ly = 0, auto = true;
    canvas.addEventListener('pointerdown', e => { drag = true; auto = false; lx = e.clientX; ly = e.clientY; });
    window.addEventListener('pointerup', () => drag = false);
    window.addEventListener('pointermove', e => { if (drag) { ry += (e.clientX - lx) * .01; rx += (e.clientY - ly) * .01; lx = e.clientX; ly = e.clientY; } });
    api.stop = loop(t => { fit(); if (auto) { ry = Math.sin(t * .7) * .55; } grp.rotation.y += (ry - grp.rotation.y) * .1; grp.rotation.x += (rx - grp.rotation.x) * .1; renderer.render(scene, camera); }, canvas);
    return api;
  }
  return { head, cloud, relief };
})();
