'use strict';
/* Poly Studio editor — leCore is the authority; this file is the viewport, the tools, and the wire.
   Sections: state · scene graph · server io · selection · gizmo · pointer routing · sculpt ·
   materials · uv · renders · keyboard · boot. */

const $ = id => document.getElementById(id);
// Any uncaught JS error surfaces in the status bar -- a silently dead UI is undebuggable from a screenshot.
window.addEventListener('error', e=>{ try{ $('status').textContent = 'JS error: '+e.message+' @'+(e.filename||'').split('/').pop()+':'+e.lineno; }catch(_){} });
window.addEventListener('unhandledrejection', e=>{ try{ $('status').textContent = 'JS promise error: '+(e.reason && e.reason.message || e.reason); }catch(_){} });
const status = t => { $('status').textContent = t || ''; };

/* ================================ viewport ================================ */
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({canvas, antialias:true});
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x10131a);
const camera = new THREE.PerspectiveCamera(45, 1, 0.05, 200);
let camTheta = 0.9, camPhi = 1.05, camDist = 4.6, camTarget = new THREE.Vector3(0, 0.1, 0);
function applyCam(){
  const sp = Math.sin(camPhi), cp = Math.cos(camPhi);
  camera.position.set(camTarget.x + camDist*sp*Math.cos(camTheta),
                      camTarget.y + camDist*cp,
                      camTarget.z + camDist*sp*Math.sin(camTheta));
  camera.lookAt(camTarget);
}
scene.add(new THREE.HemisphereLight(0xcfd8ee, 0x3a3126, 1.0));
const key = new THREE.DirectionalLight(0xffffff, 0.8); key.position.set(-2.2, 3.5, -1.4); scene.add(key);
const grid = new THREE.GridHelper(8, 32, 0x2a3140, 0x1b2029); grid.position.y = -0.9; scene.add(grid);

/* ================================ state ================================ */
let MODE = 'object';            // object | vertex | face | sculpt
let TOOL = 'move';              // move | rotate | scale
let BRUSH = 'inflate';
const OBJS = new Map();         // id -> {d, group, mesh, wire, points}
let ACTIVE = null;              // active object id (component + sculpt target)
let selObjs = new Set();
let selVerts = new Set(), selFaces = new Set();   // on ACTIVE
let softW = null;               // Float per-vertex soft-selection weights (ACTIVE)
let lastHit = null;
let MATS = null, MATCLASS = null, MATNAME = null;

function activeObj(){ return OBJS.get(ACTIVE); }
function selectionVerts(){
  const o = activeObj(); if(!o) return [];
  if(MODE === 'vertex') return [...selVerts];
  const s = new Set();
  for(const f of selFaces) for(const v of o.d.faces[f]) s.add(v);
  return [...s];
}

/* ================================ scene graph ================================ */
const meshMat = new THREE.MeshStandardMaterial({vertexColors:true, metalness:0.1, roughness:0.7,
  polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1, side:THREE.DoubleSide});
// display-mode materials (textured/lit is meshMat above). Flat = single default-lit grey, no vertex colours.
// SELECTED variants carry an accent emissive so the current selection reads instantly in flat/textured views.
const meshMatSel = new THREE.MeshStandardMaterial({vertexColors:true, metalness:0.1, roughness:0.7,
  emissive:0x2f5db3, emissiveIntensity:0.33, polygonOffset:true, polygonOffsetFactor:1,
  polygonOffsetUnits:1, side:THREE.DoubleSide});
const flatMat = new THREE.MeshStandardMaterial({color:0x9aa3b4, metalness:0.0, roughness:0.85,
  polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1, side:THREE.DoubleSide});
const flatMatSel = flatMat.clone(); flatMatSel.emissive = new THREE.Color(0x2f5db3); flatMatSel.emissiveIntensity = 0.4;
// flat-shaded twins for hard-surface objects (auto-shade picks per object)
const mkFlat = m => { const f=m.clone(); f.flatShading=true; return f; };
const meshMatF=mkFlat(meshMat), meshMatSelF=mkFlat(meshMatSel), flatMatF=mkFlat(flatMat), flatMatSelF=mkFlat(flatMatSel);
let RMB_DOWN = null;             // right-button press point: distinguishes orbit-drag from context-click
let DISPLAY = 'flat';            // flat (default) | textured | wireframe | vertex | bbox
let BOXUNFOCUSED = false;        // optional speed mode: draw non-selected objects as bounding boxes only
const boxMat = new THREE.LineBasicMaterial({color:0x6b7a95, transparent:true, opacity:0.7});
const boxMatSel = new THREE.LineBasicMaterial({color:0x7fa4dd, transparent:true, opacity:0.9});
const selFaceMat = new THREE.MeshBasicMaterial({color:0x4d8dff, transparent:true, opacity:0.4, side:THREE.DoubleSide});

/* AUTO-SHADE: decide per object whether the viewport should shade SMOOTH (soft normals, matching the SDF
   renderer, right for spheres/organic surfaces) or FLAT (crisp facets, right for boxes/hard-surface).
   Test: median dihedral between adjacent triangles < 35deg => the surface is curvature, not corners.
   (A crease-normal middle road was measured and rejected: with INDEXED geometry a cube corner holds ONE
   normal for three walls, which rounds the cube off — so the choice is per-object, not per-edge.) */
function isSmoothMesh(d){
  const idx=d.indices, pos=d.positions;
  const nf=idx.length/3;
  if(nf<2) return false;
  const fN=new Float32Array(nf*3);
  const A=new THREE.Vector3(),B=new THREE.Vector3(),C=new THREE.Vector3(),N=new THREE.Vector3();
  const e2f=new Map();
  for(let f=0;f<nf;f++){
    const a=idx[f*3],b=idx[f*3+1],c=idx[f*3+2];
    A.set(pos[a*3],pos[a*3+1],pos[a*3+2]); B.set(pos[b*3],pos[b*3+1],pos[b*3+2]); C.set(pos[c*3],pos[c*3+1],pos[c*3+2]);
    N.subVectors(B,A).cross(C.clone().sub(A)).normalize();
    fN[f*3]=N.x; fN[f*3+1]=N.y; fN[f*3+2]=N.z;
    for(const [u,w] of [[a,b],[b,c],[c,a]]){
      const k=u<w?u+'_'+w:w+'_'+u;
      if(e2f.has(k)){ e2f.set(k, [e2f.get(k), f]); } else e2f.set(k, f);
    }
  }
  const cosangs=[];
  for(const v of e2f.values()){
    if(Array.isArray(v)){
      const [f1,f2]=v;
      cosangs.push(fN[f1*3]*fN[f2*3]+fN[f1*3+1]*fN[f2*3+1]+fN[f1*3+2]*fN[f2*3+2]);
    }
  }
  if(!cosangs.length) return false;
  cosangs.sort((x,y)=>y-x);
  const med=cosangs[Math.floor(cosangs.length/2)];
  return med > Math.cos(35*Math.PI/180);
}

function polyWireGeometry(d){
  const seen=new Set(); const segs=[];
  for(const f of d.faces){
    for(let k=0;k<f.length;k++){
      const a=f[k], b=f[(k+1)%f.length];
      const key=a<b?a+'_'+b:b+'_'+a;
      if(seen.has(key)) continue; seen.add(key);
      segs.push(a,b);
    }
  }
  const g=new THREE.BufferGeometry();
  const arr=new Float32Array(segs.length*3);
  for(let i=0;i<segs.length;i++){ const vi=segs[i];
    arr[i*3]=d.positions[vi*3]; arr[i*3+1]=d.positions[vi*3+1]; arr[i*3+2]=d.positions[vi*3+2]; }
  g.setAttribute('position', new THREE.BufferAttribute(arr,3));
  g.userData.segVerts = segs;                       // vertex ids per segment endpoint, for live drag updates
  return g;
}

function buildObject(d){
  const old = OBJS.get(d.id);
  if(old) scene.remove(old.group);
  const group = new THREE.Group();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(d.positions), 3));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(d.vertColor), 3));
  g.setIndex(new THREE.BufferAttribute(new Uint32Array(d.indices), 1));
  const smooth = isSmoothMesh(d);                   // sphere-like -> soft normals (matches the renderer); box-like -> flat
  g.computeVertexNormals();
  const mesh = new THREE.Mesh(g, meshMat); group.add(mesh);
  // VIEWPORT PARITY: a textured import carries per-vertex uv; fetch its real texture and build a
  // per-object material pair (normal + selected) used by the Textured display mode instead of the palette
  let texMat=null, texMatSel=null;
  if(d.hasTexture && d.uv && d.uv.length === d.positions.length/3*2){
    g.setAttribute('uv', new THREE.BufferAttribute(new Float32Array(d.uv), 2));
    const tx = new THREE.TextureLoader().load('api/object_texture?object='+d.id, ()=>{ if(typeof requestRender==='function') requestRender(); });
    tx.flipY = false; tx.colorSpace = THREE.SRGBColorSpace;
    texMat = new THREE.MeshStandardMaterial({map:tx, metalness:0.05, roughness:0.75,
      flatShading:false, side:THREE.DoubleSide, polygonOffset:true, polygonOffsetFactor:1, polygonOffsetUnits:1});
    texMatSel = texMat.clone(); texMatSel.emissive = new THREE.Color(0x2f5db3); texMatSel.emissiveIntensity = 0.35;
  }
  const wire = new THREE.LineSegments(polyWireGeometry(d),
    new THREE.LineBasicMaterial({color:0x53617a, transparent:true, opacity:0.5}));
  group.add(wire);
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', g.attributes.position);
  pg.setAttribute('color', new THREE.BufferAttribute(new Float32Array(d.positions.length), 3));
  const points = new THREE.Points(pg, new THREE.PointsMaterial({size:0.04, vertexColors:true}));
  points.visible = false; group.add(points);
  scene.add(group);
  OBJS.set(d.id, {d, group, mesh, wire, points, texMat, texMatSel, smooth});
  applyDisplay(OBJS.get(d.id));
}

let APPLIED_REV = 0;          // highest server rev whose SCENE-level state we've applied
let OBJ_REV = new Map();      // per-object: highest rev applied for that object (single-object payloads)
let PENDING_DRAG = false;     // true while a local gizmo/vertex drag owns the geometry (server echoes are advisory)

function applyResp(r, opts){
  opts = opts || {};
  // TRANSACTION ORDERING: the server stamps every mutation with a monotonic rev and processes ops serially under
  // a lock, so rev reflects true commit order. Async responses can still LAND out of order, so we drop any that
  // are older than what we've already applied — this is what prevents the "jumping" where an earlier edit's reply
  // overwrites a later one. Scene-level payloads use a global rev; single-object payloads use a PER-OBJECT rev so
  // that edits to two different objects never cancel each other. force=true is the authoritative drag-commit path.
  const rev = (typeof r.rev === 'number') ? r.rev : null;
  if(r.objects){                               // scene-level payload
    if(rev !== null){
      if(rev < APPLIED_REV && !opts.force){ return; }
      APPLIED_REV = Math.max(APPLIED_REV, rev);
    }
    if(PENDING_DRAG && !opts.force){ return; }
    const ids = new Set(r.objects.map(o=>o.id));
    for(const [id,o] of OBJS) if(!ids.has(id)){ scene.remove(o.group); OBJS.delete(id); }
    for(const d of r.objects) buildObject(d);
    if(rev !== null) for(const d of r.objects) OBJ_REV.set(d.id, rev);
    if(!OBJS.has(ACTIVE)) ACTIVE = r.objects.length ? r.objects[r.objects.length-1].id : null;
    selObjs = new Set([...selObjs].filter(id=>OBJS.has(id)));
    if(ACTIVE && !selObjs.size) selObjs.add(ACTIVE);
  } else if(r.object){                         // single-object payload
    const oid = r.object.id;
    if(rev !== null){
      const prev = OBJ_REV.get(oid) || 0;
      if(rev < prev && !opts.force){ return; }  // stale echo for THIS object — ignore
      OBJ_REV.set(oid, Math.max(prev, rev));
      APPLIED_REV = Math.max(APPLIED_REV, rev);
    }
    if(PENDING_DRAG && !opts.force){ return; }
    buildObject(r.object);
    if(!ACTIVE) ACTIVE = r.object.id;
  } else if(PENDING_DRAG && !opts.force){ return; }
  selVerts.clear(); selFaces.clear(); softW = null;
  refreshObjList(); refreshSelectionVisuals(); refreshAttrs(); refreshModePanels();
  scheduleRender();
}

function refreshObjList(){
  const el = $('objlist'); el.innerHTML = '';
  for(const [id,o] of OBJS){
    const row = document.createElement('div');
    row.className = 'objrow' + (selObjs.has(id) ? ' sel' : '');
    row.dataset.id = id;
    row.innerHTML = `<span class="nm">${o.d.name}${o.d.sculpt?' ✦':''}</span>` +
                    `<span class="c">${o.d.counts.f}p</span>` +
                    `<span class="boxtog${o._forceBox?' on':''}" title="Show as bounding box">▢</span>`;
    row.querySelector('.boxtog').onclick = (e)=>{
      e.stopPropagation();
      o._forceBox = !o._forceBox; applyDisplay(o); refreshSelectionVisuals(); refreshObjList();
      status(`${o.d.name}: ${o._forceBox?'shown as box':'shown normally'}`);
    };
    row.onclick = (e)=>{ if(!e.shiftKey) selObjs.clear(); selObjs.add(id); ACTIVE=id;
                         selVerts.clear(); selFaces.clear(); softW=null;
                         refreshObjList(); refreshSelectionVisuals(); refreshAttrs(); };
    el.appendChild(row);
  }
}

function refreshAttrs(){
  const o = activeObj();
  $('a_name').textContent = o ? o.d.name : '—';
  $('a_counts').textContent = o ? `${o.d.counts.v} / ${o.d.counts.f}` : '—';
  $('a_sel').textContent = MODE==='object' ? `${selObjs.size} object(s)` :
    MODE==='vertex' ? `${selVerts.size} vertex(es)` : MODE==='face' ? `${selFaces.size} face(s)` : 'sculpting';
  $('a_mat').textContent = MATNAME || '—';
  const actName = (ACTIVE && OBJS.has(ACTIVE)) ? (OBJS.get(ACTIVE).d.name || 'object') : '—';
  const modeLabel = {object:'Object',vertex:'Vertex',face:'Face',sculpt:'Sculpt',paint:'Paint'}[MODE]||MODE;
  const toolLabel = MODE==='sculpt'?BRUSH:(MODE==='paint'?'brush':TOOL);
  $('hud').textContent = `${modeLabel} · ${toolLabel} · ${actName}`;
}

function refreshSelectionVisuals(){
  const anyForceBox = [...OBJS.values()].some(o=>o._forceBox);
  if(BOXUNFOCUSED || DISPLAY==='bbox' || anyForceBox){ for(const [,o] of OBJS) applyDisplay(o); }
  for(const [id,o] of OBJS){
    if(o._boxed){                    // object is drawn as a bounding box only — don't reveal solid/wire/points
      if(o.box){ o.box.material = (selObjs.has(id)||id===ACTIVE) ? boxMatSel : boxMat; }
      o.points.visible = false; o.wire.visible = false;
      if(o.selFaceObj){ o.group.remove(o.selFaceObj); o.selFaceObj.geometry.dispose(); o.selFaceObj=null; }
      continue;
    }
    const isSelHl = selObjs.has(id) || id===ACTIVE;
    if(DISPLAY==='textured'){ o.mesh.material = o.texMat ? (isSelHl ? o.texMatSel : o.texMat)
                                                          : (isSelHl ? (o.smooth?meshMatSel:meshMatSelF) : (o.smooth?meshMat:meshMatF)); }
    else if(DISPLAY==='flat'){ o.mesh.material = isSelHl ? (o.smooth?flatMatSel:flatMatSelF) : (o.smooth?flatMat:flatMatF); }
    const editPts = (id===ACTIVE && (MODE==='vertex'||MODE==='face'));
    o.points.visible = editPts || (DISPLAY==='vertex');
    // wire visible when: wireframe display, overlay on, or actively editing verts/faces
    o.wire.visible = (DISPLAY==='wireframe') || o._wireForced || editPts;
    o.wire.material.opacity = selObjs.has(id) ? 0.85 : (DISPLAY==='wireframe'?0.6:0.35);
    o.wire.material.color.set(selObjs.has(id) ? 0x7fa4dd : 0x53617a);
    if(o.selFaceObj){ o.group.remove(o.selFaceObj); o.selFaceObj.geometry.dispose(); o.selFaceObj=null; }
  }
  const o = activeObj(); if(!o) { placeGizmo(); return; }
  if(o.points.visible){
    const col = o.points.geometry.attributes.color;
    const vsel = new Set(selectionVerts());
    for(let i=0;i<col.count;i++){
      const s = vsel.has(i);
      const w = softW ? softW[i] : 0;
      col.setXYZ(i, s?0.35:0.55+w*0.4, s?0.65:0.60+w*0.2, s?1.0:0.70);
    }
    col.needsUpdate = true;
  }
  if(MODE==='face' && selFaces.size){
    const d = o.d, tris=[];
    d.triFace.forEach((f,t)=>{ if(selFaces.has(f)) tris.push(d.indices[t*3],d.indices[t*3+1],d.indices[t*3+2]); });
    const g=new THREE.BufferGeometry();
    g.setAttribute('position', o.mesh.geometry.attributes.position);
    g.setIndex(tris);
    o.selFaceObj = new THREE.Mesh(g, selFaceMat); o.group.add(o.selFaceObj);
  }
  placeGizmo();
  refreshAttrs();
}

/* ================================ server io ================================ */
async function api(path, body){
  const r = await fetch('api/'+path, body===undefined?{}:{method:'POST',
    headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const d = await r.json().catch(()=>({}));
  if(!r.ok){ status('Engine: '+(d.error||r.status)); throw new Error(d.error||r.status); }
  status('');
  return d;
}
async function runOp(op, args){
  status(op+'…');
  try{
    const r = await api('op', Object.assign({op}, args||{}));
    if(r && r.error){ status(op+' — '+r.error); return; }
    applyResp(r);
    status(op.replace(/_/g,' ')+' done.');
  }catch(e){ status(op+' failed.'); }
}

/* ================================ picking ================================ */
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
function viewUnderCursor(ev){
  if(SPLIT===1) return VIEWS[0];
  const r = canvas.getBoundingClientRect();
  const px = ev.clientX-r.left, py = ev.clientY-r.top;
  const rects = viewRects(r.width, r.height);
  const nv = SPLIT===2?2:4;
  for(let i=0;i<nv;i++){ const [x,y,rw,rh]=rects[i];
    // rects are GL coords (y up); convert cursor y
    const gy = r.height-py;
    if(px>=x && px<x+rw && gy>=y && gy<y+rh) return VIEWS[i];
  }
  return VIEWS[0];
}
function setNdc(ev){
  const r = canvas.getBoundingClientRect();
  const v = viewUnderCursor(ev);
  let nx, ny;
  if(SPLIT===1 || v===VIEWS[0] && SPLIT===1){
    nx = ((ev.clientX-r.left)/r.width)*2-1; ny = -((ev.clientY-r.top)/r.height)*2+1;
  } else {
    // map cursor into the local rect of its view
    const rects = viewRects(r.width, r.height);
    const idx = VIEWS.indexOf(v); const [x,y,rw,rh]=rects[Math.min(idx, rects.length-1)];
    const lx = (ev.clientX-r.left)-x; const gy = r.height-(ev.clientY-r.top); const ly = gy-y;
    nx = (lx/rw)*2-1; ny = (ly/rh)*2-1;
  }
  ndc.set(nx, ny);
  ray.setFromCamera(ndc, v.cam);
}
function pickScene(ev){
  setNdc(ev);
  const meshes = [...OBJS.values()].map(o=>o.mesh);
  const hit = ray.intersectObjects(meshes)[0];
  if(!hit) return null;
  const id = [...OBJS.entries()].find(([,o])=>o.mesh===hit.object)[0];
  const d = OBJS.get(id).d;
  const poly = d.triFace[hit.faceIndex];
  let vertex=-1, bd=1e9;
  for(const v of d.faces[poly]){
    const p = new THREE.Vector3(d.positions[v*3],d.positions[v*3+1],d.positions[v*3+2]);
    const dd = p.distanceTo(hit.point); if(dd<bd){bd=dd;vertex=v;}
  }
  const f=d.faces[poly]; let edge=null, bed=1e9;
  for(let k=0;k<f.length;k++){
    const a=f[k], b=f[(k+1)%f.length];
    const mid=new THREE.Vector3((d.positions[a*3]+d.positions[b*3])/2,
      (d.positions[a*3+1]+d.positions[b*3+1])/2,(d.positions[a*3+2]+d.positions[b*3+2])/2);
    const dd=mid.distanceTo(hit.point); if(dd<bed){bed=dd;edge=[a,b];}
  }
  return {id, poly, vertex, edge, point:hit.point.clone(), normal:hit.face.normal.clone()};
}

function selectIsland(h){
  const d = OBJS.get(h.id).d;
  if(MODE==='face' || MODE==='object'){
    const v2f = new Map();
    d.faces.forEach((f,i)=>f.forEach(v=>{ (v2f.get(v)||v2f.set(v,[]).get(v)).push(i); }));
    const seen=new Set([h.poly]), q=[h.poly];
    while(q.length){ const f=q.pop(); for(const v of d.faces[f]) for(const g of v2f.get(v)) if(!seen.has(g)){seen.add(g);q.push(g);} }
    if(MODE==='face'){ seen.forEach(f=>selFaces.add(f)); }
  } else {
    const adj=new Map();
    d.faces.forEach(f=>{ for(let k=0;k<f.length;k++){ const a=f[k],b=f[(k+1)%f.length];
      (adj.get(a)||adj.set(a,new Set()).get(a)).add(b); (adj.get(b)||adj.set(b,new Set()).get(b)).add(a);} });
    const seen=new Set([h.vertex]), q=[h.vertex];
    while(q.length){ const v=q.pop(); for(const w of (adj.get(v)||[])) if(!seen.has(w)){seen.add(w);q.push(w);} }
    seen.forEach(v=>selVerts.add(v));
  }
}

async function fetchSoftWeights(){
  softW = null;
  const o = activeObj();
  if(!o || !$('softsel').checked || MODE!=='vertex' || !selVerts.size) { refreshSelectionVisuals(); return; }
  try{
    const d = await api('softsel', {object:ACTIVE, verts:[...selVerts],
                                    radius:parseFloat($('softrad').value)||0.5,
                                    falloff:$('softfall').value});
    softW = d.weights;
  }catch(e){}
  refreshSelectionVisuals();
}

/* ================================ gizmo ================================ */
/* A compact C4D-style handle set built from primitives: translate arrows, rotate rings, scale cubes
   (+ center cube = uniform). Object mode drags the whole selected object(s) live and BAKES the final
   matrix server-side on release; component modes apply the matrix to the selected vertices (times the
   soft-selection weight) and stream positions at ~20 Hz. */
const AXCOL = [0xd06a5a, 0x7fc06a, 0x5a8ad0];
const gizmo = new THREE.Group(); gizmo.visible = false; scene.add(gizmo);
let gizmoParts = [];

function buildGizmo(){
  gizmo.clear(); gizmoParts = [];
  const mk = (geom, color, axis, kind, pos, rot)=>{
    const m = new THREE.Mesh(geom, new THREE.MeshBasicMaterial({color, depthTest:false, transparent:true, opacity:0.92}));
    if(pos) m.position.copy(pos);
    if(rot) m.rotation.set(rot.x,rot.y,rot.z);
    m.renderOrder = 999; m.userData = {axis, kind};
    gizmo.add(m); gizmoParts.push(m);
  };
  const L = 0.62;
  for(let a=0;a<3;a++){
    const dir = new THREE.Vector3(); dir.setComponent(a,1);
    if(TOOL==='move'){
      const rot = a===0?{x:0,y:0,z:-Math.PI/2}:a===2?{x:Math.PI/2,y:0,z:0}:{x:0,y:0,z:0};
      mk(new THREE.CylinderGeometry(0.014,0.014,L,8), AXCOL[a], a, 'axis', dir.clone().multiplyScalar(L/2), rot);
      mk(new THREE.ConeGeometry(0.05,0.14,10), AXCOL[a], a, 'axis', dir.clone().multiplyScalar(L+0.06), rot);
    } else if(TOOL==='rotate'){
      const rot = a===0?{x:0,y:Math.PI/2,z:0}:a===1?{x:Math.PI/2,y:0,z:0}:{x:0,y:0,z:0};
      mk(new THREE.TorusGeometry(0.55, 0.016, 8, 48), AXCOL[a], a, 'ring', null, rot);
    } else {
      const rot = a===0?{x:0,y:0,z:-Math.PI/2}:a===2?{x:Math.PI/2,y:0,z:0}:{x:0,y:0,z:0};
      mk(new THREE.CylinderGeometry(0.012,0.012,L,8), AXCOL[a], a, 'axis', dir.clone().multiplyScalar(L/2), rot);
      mk(new THREE.BoxGeometry(0.09,0.09,0.09), AXCOL[a], a, 'axis', dir.clone().multiplyScalar(L+0.05));
    }
  }
  if(TOOL==='scale') mk(new THREE.BoxGeometry(0.11,0.11,0.11), 0xd8dbe2, -1, 'uniform');
}

function selectionPivot(){
  if(MODE==='object'){
    if(!selObjs.size) return null;
    const c = new THREE.Vector3(); let n=0;
    for(const id of selObjs){
      const p = OBJS.get(id).mesh.geometry.attributes.position;
      for(let i=0;i<p.count;i++){ c.x+=p.getX(i); c.y+=p.getY(i); c.z+=p.getZ(i); }
      n += p.count;
    }
    return n? c.multiplyScalar(1/n) : null;
  }
  const o = activeObj(); const vs = selectionVerts();
  if(!o || !vs.length) return null;
  const p = o.mesh.geometry.attributes.position;
  const c = new THREE.Vector3();
  for(const v of vs){ c.x+=p.getX(v); c.y+=p.getY(v); c.z+=p.getZ(v); }
  return c.multiplyScalar(1/vs.length);
}

function placeGizmo(){
  const piv = (MODE!=='sculpt') ? selectionPivot() : null;
  if(!piv){ gizmo.visible=false; return; }
  buildGizmo();
  gizmo.position.copy(piv);
  const s = camera.position.distanceTo(piv) * 0.22;
  gizmo.scale.setScalar(s);
  gizmo.visible = true;
}

/* -------- gizmo drag math -------- */
let gdrag = null;   // {kind, axis, pivot, start(point), base:{objId->positions[] or matrices}, softVs, accumM}
function gizmoHit(ev){
  if(!gizmo.visible) return null;
  setNdc(ev);
  const hit = ray.intersectObjects(gizmoParts)[0];
  return hit ? hit.object.userData : null;
}
function axisDir(a){ const v=new THREE.Vector3(); v.setComponent(a,1); return v; }
function rayOnAxis(pivot, a){                 // closest point param of the mouse ray to the axis line
  const d = axisDir(a), o = ray.ray.origin, u = ray.ray.direction;
  const w0 = o.clone().sub(pivot);
  const b = u.dot(d), c = u.dot(w0), e = d.dot(w0);
  const den = 1 - b*b; if(Math.abs(den)<1e-6) return 0;
  // minimising |w0 + s*u - t*d|^2 over (s,t) gives t = (e - b*c)/(1 - b^2). The previous (b*c - e) was the
  // NEGATED parameter -- every axis-drag moved opposite the cursor (scale never showed it: t/t0 cancels a
  // global sign; move is t - t0, which doesn't). Found by a user's first real drag; verified numerically.
  return (e - b*c)/den;                        // param along axis
}
function rayOnPlane(pivot, normal){
  const p = new THREE.Vector3();
  return ray.ray.intersectPlane(new THREE.Plane().setFromNormalAndCoplanarPoint(normal, pivot), p) ? p : null;
}

function beginGizmoDrag(ud, ev){
  const pivot = gizmo.position.clone();
  setNdc(ev);
  PENDING_DRAG = true;                          // local geometry is authoritative until the drag commits
  gdrag = {kind:ud.kind, axis:ud.axis, pivot, accumM:new THREE.Matrix4()};
  gdrag.snap = ($('snapGrid') && $('snapGrid').checked) || ev.ctrlKey || ev.metaKey;
  gdrag.snapStep = parseFloat($('snapStep') && $('snapStep').value) || 0.25;
  if(TOOL==='move'){
    gdrag.t0 = rayOnAxis(pivot, ud.axis);
  } else if(TOOL==='rotate'){
    const n = axisDir(ud.axis);
    const p = rayOnPlane(pivot, n); if(!p){ gdrag=null; return false; }
    gdrag.v0 = p.sub(pivot).normalize(); gdrag.n = n;
  } else {
    if(ud.axis>=0) gdrag.t0 = Math.max(rayOnAxis(pivot, ud.axis), 1e-3);
    else { const p = rayOnPlane(pivot, camera.getWorldDirection(new THREE.Vector3()).negate());
           gdrag.d0 = p ? Math.max(p.distanceTo(pivot),1e-3) : 1; }
  }
  // capture base positions
  if(MODE==='object'){
    gdrag.base = new Map();
    for(const id of selObjs) gdrag.base.set(id, Float32Array.from(OBJS.get(id).mesh.geometry.attributes.position.array));
    api('op', {op:'begin_drag'});             // scene snapshot (multi-object transform)
  } else {
    const o = activeObj(); if(!o){ gdrag=null; return false; }
    gdrag.vs = selectionVerts(); if(!gdrag.vs.length){ gdrag=null; return false; }
    gdrag.softVs = softW ? Array.from({length:o.d.counts.v},(_,i)=>i).filter(i=>softW[i]>0 && !gdrag.vs.includes(i)) : [];
    gdrag.base = Float32Array.from(o.mesh.geometry.attributes.position.array);
    api('op', {op:'begin_drag', object:ACTIVE});
  }
  return true;
}

function gizmoMatrix(ev){
  setNdc(ev);
  const P = gdrag.pivot, M = new THREE.Matrix4();
  if(TOOL==='move'){
    let t = rayOnAxis(P, gdrag.axis) - gdrag.t0;
    if(gdrag.snap){ const step = gdrag.snapStep||0.25; t = Math.round(t/step)*step; }   // Ctrl / snap-toggle: quantise
    return M.makeTranslation(...axisDir(gdrag.axis).multiplyScalar(t).toArray());
  }
  if(TOOL==='rotate'){
    const p = rayOnPlane(P, gdrag.n); if(!p) return null;
    const v1 = p.sub(P).normalize();
    let ang = Math.acos(THREE.MathUtils.clamp(gdrag.v0.dot(v1), -1, 1));
    if(gdrag.n.dot(new THREE.Vector3().crossVectors(gdrag.v0, v1)) < 0) ang = -ang;
    const R = new THREE.Matrix4().makeRotationAxis(gdrag.n, ang);
    return M.makeTranslation(P.x,P.y,P.z).multiply(R).multiply(new THREE.Matrix4().makeTranslation(-P.x,-P.y,-P.z));
  }
  // scale
  let sx=1, sy=1, sz=1;
  if(gdrag.axis>=0){
    const s = Math.max(rayOnAxis(P, gdrag.axis) / gdrag.t0, 0.02);
    if(gdrag.axis===0) sx=s; else if(gdrag.axis===1) sy=s; else sz=s;
  } else {
    const p = rayOnPlane(P, camera.getWorldDirection(new THREE.Vector3()).negate());
    const s = p ? Math.max(p.distanceTo(P)/gdrag.d0, 0.02) : 1;
    sx=sy=sz=s;
  }
  const S = new THREE.Matrix4().makeScale(sx,sy,sz);
  return M.makeTranslation(P.x,P.y,P.z).multiply(S).multiply(new THREE.Matrix4().makeTranslation(-P.x,-P.y,-P.z));
}

let lastStream = 0, streamBusy=false, streamQueued=false;
let streamPending = null;    // latest vertex set awaiting a free channel (coalesces rapid drag frames)
async function streamActiveVerts(vs){
  streamPending = vs;                            // remember the most recent request
  if(streamBusy){ streamQueued = true; return; }
  streamBusy = true;
  while(true){
    const cur = streamPending; streamPending = null; streamQueued = false;
    const o = activeObj();
    if(!o){ break; }
    const P = o.mesh.geometry.attributes.position;
    const pos = []; for(const v of cur) pos.push(P.getX(v), P.getY(v), P.getZ(v));   // read LATEST positions
    try{ await api('verts', {object:ACTIVE, indices:cur, positions:pos}); }catch(e){}
    if(streamPending === null) break;            // nothing newer arrived while we were in flight
  }
  streamBusy = false;
}

function applyGizmoDrag(ev){
  const M = gizmoMatrix(ev); if(!M) return;
  gdrag.accumM.copy(M);
  const v = new THREE.Vector3();
  if(MODE==='object'){
    for(const id of selObjs){
      const o = OBJS.get(id);
      const P = o.mesh.geometry.attributes.position, B = gdrag.base.get(id);
      for(let i=0;i<P.count;i++){ v.set(B[i*3],B[i*3+1],B[i*3+2]).applyMatrix4(M); P.setXYZ(i,v.x,v.y,v.z); }
      P.needsUpdate = true; o.mesh.geometry.computeVertexNormals(); o.mesh.geometry.computeBoundingSphere();
      rebuildWire(o);
    }
  } else {
    const o = activeObj();
    const P = o.mesh.geometry.attributes.position, B = gdrag.base;
    const touched = [];
    for(const i of gdrag.vs){
      v.set(B[i*3],B[i*3+1],B[i*3+2]).applyMatrix4(M);
      P.setXYZ(i,v.x,v.y,v.z); touched.push(i);
      o.d.positions[i*3]=v.x; o.d.positions[i*3+1]=v.y; o.d.positions[i*3+2]=v.z;
    }
    for(const i of (gdrag.softVs||[])){       // soft selection: blend by geodesic weight
      const w = softW[i];
      const b = new THREE.Vector3(B[i*3],B[i*3+1],B[i*3+2]);
      v.copy(b).applyMatrix4(M).lerp(b, 1-w);
      P.setXYZ(i,v.x,v.y,v.z); touched.push(i);
      o.d.positions[i*3]=v.x; o.d.positions[i*3+1]=v.y; o.d.positions[i*3+2]=v.z;
    }
    P.needsUpdate = true; o.mesh.geometry.computeVertexNormals(); o.mesh.geometry.computeBoundingSphere();
    rebuildWire(o);
    const now = performance.now();
    if(now-lastStream > 50){ lastStream = now; streamActiveVerts(touched); }
    gdrag.touched = touched;
  }
  placeGizmo();
}

async function endGizmoDrag(){
  if(!gdrag) return;
  const wasObject = (MODE==='object');
  const touched = gdrag.touched;
  const commitM = gdrag.accumM.clone().transpose();  // three is column-major; server math is row-major
  const ids = [...selObjs];
  gdrag = null;                                  // stop applyGizmoDrag from mutating further
  PENDING_DRAG = false;                           // release local ownership; the commit below is authoritative
  try{
    if(wasObject){
      // commit each selected object with the SAME accumulated matrix. transform returns a single-object
      // payload, so apply each (force past the stale-rev guard, in commit order) — the rev advances monotonically
      // so the guard still rejects any *older* in-flight echo that lands afterwards.
      for(const id of ids){
        const resp = await api('op', {op:'transform', object:id, matrix:commitM.elements});
        applyResp(resp, {force:true});
      }
    } else if(touched){
      await streamActiveVerts(touched);          // final flush; server rev advances, no geometry echo to fight
      try{                                        // then re-sync from the server: one authoritative rebuild
        const fresh = await (await fetch('api/scene')).json();
        applyResp(fresh, {force:true});
      }catch(_){}
    }
  }catch(e){}
  scheduleRender();
}

function rebuildWire(o){
  // polygon wire tracks the mesh: rewrite each segment endpoint from the live positions
  const segs = o.wire.geometry.userData.segVerts;
  const P = o.mesh.geometry.attributes.position, W = o.wire.geometry.attributes.position;
  if(segs && W.count === segs.length){
    for(let i=0;i<segs.length;i++){ const vi=segs[i]; W.setXYZ(i, P.getX(vi), P.getY(vi), P.getZ(vi)); }
    W.needsUpdate = true; o.wire.geometry.computeBoundingSphere();
  }
}

/* ================================ pointer routing ================================ */
let orbiting=false, panning=false, dollying=false, px=0, py=0, downPos=null, marqueeStart=null;
let sculptStroke = null;   // {pts:[], timer}
canvas.addEventListener('contextmenu', e=>e.preventDefault());

canvas.addEventListener('pointerdown', async ev=>{
  if(ev.pointerType==='touch') return;                      // G5: the touch module owns these
  px=ev.clientX; py=ev.clientY; downPos={x:ev.clientX,y:ev.clientY};
  // C4D navigation: Alt+LMB orbit, Alt+MMB pan, Alt+RMB dolly -- plus plain RMB orbit / MMB pan
  // F10: capture on EVERY navigation button, not just button 0 -- otherwise dragging the cursor out
  // of the window mid-orbit silently stopped tracking and left the camera half-moved.
  const nav = ()=>{ try{ canvas.setPointerCapture(ev.pointerId); }catch(_){ } };
  if(ev.altKey && ev.button===0){ orbiting=true; nav(); return; }
  if(ev.altKey && ev.button===2){ dollying=true; RMB_DOWN={x:ev.clientX,y:ev.clientY}; nav(); ev.preventDefault(); return; }
  if(ev.button===2){ orbiting=true; RMB_DOWN={x:ev.clientX,y:ev.clientY}; nav(); return; }
  if(ev.button===1){ panning=true; nav(); ev.preventDefault(); return; }
  if(ev.button!==0) return;
  canvas.setPointerCapture(ev.pointerId);

  if(MODE==='sculpt'){ await beginSculptStroke(ev); return; }
  if(MODE==='paint'){
    paintStroke = {id:'p'+Math.random().toString(36).slice(2), painted:new Set(), pending:new Set(), timer:null};
    paintAt(ev); return;
  }

  const ud = gizmoHit(ev);
  if(ud){ if(beginGizmoDrag(ud, ev)) return; }

  selectAtPointer(ev, true);
});

/* G5: extracted so the touch path can select with a tap without duplicating the rules. */
function selectAtPointer(ev, allowMarquee){
  const h = pickScene(ev); lastHit = h;
  if(!h){ if(allowMarquee) marqueeStart = {x:ev.clientX,y:ev.clientY, add:ev.shiftKey}; return false; }
  if(MODE==='object'){
    if(!ev.shiftKey) selObjs.clear();
    if(ev.shiftKey && selObjs.has(h.id)) selObjs.delete(h.id); else selObjs.add(h.id);
    ACTIVE = h.id;
  } else {
    if(h.id!==ACTIVE){ ACTIVE=h.id; selVerts.clear(); selFaces.clear(); softW=null; }
    const set = MODE==='face' ? selFaces : selVerts;
    const key = MODE==='face' ? h.poly : h.vertex;
    if(!ev.shiftKey && !set.has(key)){ set.clear(); }
    if(ev.shiftKey && set.has(key)) set.delete(key); else set.add(key);
    if(MODE==='vertex') fetchSoftWeights();
  }
  refreshObjList(); refreshSelectionVisuals();
  return true;
}

canvas.addEventListener('dblclick', ev=>{
  if(MODE==='sculpt') return;
  if(ev.pointerType==='touch') return;                      // G5: a double-tap is not a double-click
  const h = pickScene(ev); if(!h) return;
  if(h.id!==ACTIVE){ ACTIVE=h.id; selVerts.clear(); selFaces.clear(); }
  if(MODE==='object'){ selObjs.add(h.id); } else selectIsland(h);
  refreshObjList(); refreshSelectionVisuals();
  if(MODE==='vertex') fetchSoftWeights();
});

canvas.addEventListener('pointermove', ev=>{
  if(ev.pointerType==='touch') return;                      // G5
  const dx=ev.clientX-px, dy=ev.clientY-py;
  if(orbiting){ camTheta+=dx*0.008; camPhi=Math.min(2.9,Math.max(0.15,camPhi-dy*0.008));
                px=ev.clientX; py=ev.clientY; applyCam(); placeGizmo(); camMoved(); return; }
  if(panning){
    const right=new THREE.Vector3().setFromMatrixColumn(camera.matrix,0);
    const up=new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);
    camTarget.addScaledVector(right,-dx*0.0022*camDist).addScaledVector(up,dy*0.0022*camDist);
    px=ev.clientX; py=ev.clientY; applyCam(); placeGizmo(); camMoved(); return;
  }
  if(dollying){
    camDist=Math.min(30,Math.max(0.8,camDist*(1+(dx+dy)*0.004)));
    px=ev.clientX; py=ev.clientY; applyCam(); placeGizmo(); camMoved(); return;
  }
  if(MODE==='sculpt'){ updateBrushCursor(ev); if(sculptStroke) addStrokePoint(ev); return; }
  if(MODE==='paint'){ updateBrushCursor(ev); if(paintStroke) paintAt(ev); return; }
  if(gdrag){ applyGizmoDrag(ev); return; }
  if(marqueeStart){ drawMarquee(ev); return; }
});

canvas.addEventListener('pointerup', async ev=>{
  if(ev.pointerType==='touch') return;                      // G5
  orbiting=panning=dollying=false;
  if(MODE==='paint'){ endPaintStroke(); }
  if(MODE==='sculpt'){ endSculptStroke(); return; }
  if(gdrag){ await endGizmoDrag(); return; }
  if(marqueeStart){ finishMarquee(ev); return; }
});

canvas.addEventListener('wheel', ev=>{
  ev.preventDefault();
  camDist=Math.min(30,Math.max(0.8,camDist*(1+ev.deltaY*0.0011)));
  applyCam(); placeGizmo(); camMoved();
}, {passive:false});

/* -------- marquee -------- */
function drawMarquee(ev){
  const m = $('marquee'), r = canvas.getBoundingClientRect();
  const x0=Math.min(marqueeStart.x,ev.clientX)-r.left, y0=Math.min(marqueeStart.y,ev.clientY)-r.top;
  m.style.display='block';
  m.style.left=x0+'px'; m.style.top=y0+'px';
  m.style.width=Math.abs(ev.clientX-marqueeStart.x)+'px';
  m.style.height=Math.abs(ev.clientY-marqueeStart.y)+'px';
}
function finishMarquee(ev){
  const m = $('marquee'); m.style.display='none';
  const r = canvas.getBoundingClientRect();
  const x0=Math.min(marqueeStart.x,ev.clientX), x1=Math.max(marqueeStart.x,ev.clientX);
  const y0=Math.min(marqueeStart.y,ev.clientY), y1=Math.max(marqueeStart.y,ev.clientY);
  const add = marqueeStart.add; marqueeStart=null;
  const tiny = (x1-x0)<4 && (y1-y0)<4;
  if(tiny){ if(!add){ selObjs.clear(); selVerts.clear(); selFaces.clear(); softW=null;
            refreshObjList(); refreshSelectionVisuals(); } return; }
  const v = new THREE.Vector3();
  const inRect = (wx,wy,wz)=>{
    v.set(wx,wy,wz).project(camera);
    const sx = (v.x+1)/2*r.width + r.left, sy = (-v.y+1)/2*r.height + r.top;
    return sx>=x0 && sx<=x1 && sy>=y0 && sy<=y1 && v.z<1;
  };
  if(MODE==='object'){
    if(!add) selObjs.clear();
    for(const [id,o] of OBJS){
      const P = o.mesh.geometry.attributes.position;
      let inside=false;
      for(let i=0;i<P.count && !inside;i+=Math.max(1,(P.count/200)|0))
        inside = inRect(P.getX(i),P.getY(i),P.getZ(i));
      if(inside){ selObjs.add(id); ACTIVE=id; }
    }
  } else {
    const o = activeObj(); if(!o) return;
    if(!add){ selVerts.clear(); selFaces.clear(); }
    const P = o.mesh.geometry.attributes.position;
    if(MODE==='vertex'){
      for(let i=0;i<P.count;i++) if(inRect(P.getX(i),P.getY(i),P.getZ(i))) selVerts.add(i);
      fetchSoftWeights();
    } else {
      o.d.faces.forEach((f,fi)=>{
        let cx=0,cy=0,cz=0; for(const vv of f){ cx+=P.getX(vv); cy+=P.getY(vv); cz+=P.getZ(vv); }
        if(inRect(cx/f.length,cy/f.length,cz/f.length)) selFaces.add(fi);
      });
    }
  }
  refreshObjList(); refreshSelectionVisuals();
}

/* ================================ material paint ================================ */
let paintStroke = null;                                   // {id, painted:Set, timer, pending:Set}
function paintAt(ev){
  const o = activeObj(); if(!o) return;
  const h = pickScene(ev);
  if(!h || h.id!==ACTIVE) return;
  const R = parseFloat($('pradius').value);
  const d = o.d, P = d.positions, hp = h.point;
  const hits = [];
  for(let fi=0; fi<d.faces.length; fi++){
    if(paintStroke.painted.has(fi)) continue;
    const f = d.faces[fi];
    let cx=0, cy=0, cz=0;
    for(const v of f){ cx+=P[v*3]; cy+=P[v*3+1]; cz+=P[v*3+2]; }
    const n=f.length, dx=cx/n-hp.x, dy=cy/n-hp.y, dz=cz/n-hp.z;
    if(dx*dx+dy*dy+dz*dz <= R*R) hits.push(fi);
  }
  if(!hits.length) return;
  for(const fi of hits){ paintStroke.painted.add(fi); paintStroke.pending.add(fi); }
  // immediate local feedback: recolor picked faces client-side by tinting selFaceObj-style overlay is heavy;
  // rely on the throttled server round-trip below (~80ms), which rebuilds this object's colors from truth.
  if(!paintStroke.timer) paintStroke.timer = setTimeout(flushPaint, 80);
}
async function flushPaint(){
  if(!paintStroke){ return; }
  paintStroke.timer = null;
  const faces = [...paintStroke.pending];
  if(!faces.length) return;
  paintStroke.pending.clear();
  try{
    const r = await api('assign', {object:ACTIVE, faces, material:MATNAME, stroke:paintStroke.id});
    if(r.object) buildObject(r.object);                   // light path: refresh this object only, mid-stroke
  }catch(e){}
  if(paintStroke && paintStroke.pending.size) paintStroke.timer = setTimeout(flushPaint, 80);
}
function endPaintStroke(){
  if(!paintStroke) return;
  clearTimeout(paintStroke.timer);
  const had = paintStroke.pending.size;
  const p = paintStroke; paintStroke = null;
  if(had){ (async()=>{ try{
    const r = await api('assign', {object:ACTIVE, faces:[...p.pending], material:MATNAME, stroke:p.id});
    if(r.object) buildObject(r.object);
  }catch(e){} refreshAttrs(); scheduleRender(); })(); }
  else { refreshAttrs(); scheduleRender(); }
}

/* ================================ sculpt ================================ */
function updateBrushCursor(ev){
  const o = activeObj();
  const c = $('brushcursor');
  const h = pickScene(ev);
  const paint = MODE==='paint';
  if(!o || (!paint && !o.d.sculpt) || !h || h.id!==ACTIVE){ c.style.display='none'; return; }
  const r = parseFloat(paint ? $('pradius').value : $('brad').value);
  // project brush radius to pixels at the hit depth
  const d = camera.position.distanceTo(h.point);
  const rect = canvas.getBoundingClientRect();
  const pix = r / (2*d*Math.tan(THREE.MathUtils.degToRad(45/2))) * rect.height;
  c.style.display='block';
  c.style.left=(ev.clientX-rect.left)+'px'; c.style.top=(ev.clientY-rect.top)+'px';
  c.style.width=(pix*2)+'px'; c.style.height=(pix*2)+'px';
}
async function beginSculptStroke(ev){
  const o = activeObj();
  if(!o || !o.d.sculpt) { status('Enter Sculpt on an object first (press 4).'); return; }
  const h = pickScene(ev); if(!h || h.id!==ACTIVE) return;
  try{ await api('sculpt/begin_stroke', {object:ACTIVE}); }catch(e){ return; }
  sculptStroke = {pts:[[h.point.x,h.point.y,h.point.z]], busy:false, pending:false};
  flushStroke();
}
function addStrokePoint(ev){
  const h = pickScene(ev);
  if(h && h.id===ACTIVE) sculptStroke.pts.push([h.point.x,h.point.y,h.point.z]);
  if(sculptStroke.pts.length) flushStroke();
}
async function flushStroke(){
  if(!sculptStroke || sculptStroke.busy || !sculptStroke.pts.length) return;
  sculptStroke.busy = true;
  const pts = sculptStroke.pts.splice(0);
  try{
    const d = await api('sculpt/stroke', {object:ACTIVE, brush:BRUSH, points:pts,
      r:parseFloat($('brad').value), s:parseFloat($('bstr').value)});
    buildObject(d.object); refreshSelectionVisuals(); refreshObjList(); refreshAttrs();
  }catch(e){}
  if(sculptStroke){ sculptStroke.busy=false; if(sculptStroke.pts.length) flushStroke(); }
}
function endSculptStroke(){
  if(sculptStroke && sculptStroke.pts.length) flushStroke();
  sculptStroke = null;
  scheduleRender();
}
async function setMode(m){
  const o = activeObj();
  if(m==='sculpt'){
    if(!o){ status('Select an object first.'); return; }
    if(!o.d.sculpt){
      status('Baking the sculpt field…');
      try{ applyResp(await api('sculpt/enter', {object:ACTIVE})); }catch(e){ return; }
    }
  } else if(MODE==='sculpt' && o && o.d.sculpt){
    status('Rebuilding polygons…');
    try{ applyResp(await api('sculpt/exit', {object:ACTIVE, target_faces:parseInt($('btgt').value)})); }
    catch(e){ return; }
  }
  MODE = m;
  selVerts.clear(); selFaces.clear(); softW=null;
  for(const b of ['modeObj','modeVert','modeFace','modeSculpt','modePaint']) $(b).classList.remove('on');
  $({object:'modeObj',vertex:'modeVert',face:'modeFace',sculpt:'modeSculpt',paint:'modePaint'}[m]).classList.add('on');
  refreshModePanels(); refreshSelectionVisuals(); refreshAttrs();
  if(typeof updateHint==='function') updateHint();
}
function refreshModePanels(){
  $('polytools').style.display = (MODE==='vertex'||MODE==='face') ? '' : 'none';   // soft-select only in Vertex/Face
  $('sculpttools').style.display = MODE==='sculpt' ? '' : 'none';
  $('painttools').style.display = MODE==='paint' ? '' : 'none';
  $('toolseg').style.display = (MODE==='sculpt'||MODE==='paint') ? 'none' : 'flex';
  $('brushcursor').style.display = 'none';
}

/* ================================ materials (browser strip) ================================ */
let MATBROWSER = null;
function matItems(){
  // flatten every class into one searchable list; group = class name (folder)
  const items = [];
  for(const cls of Object.keys(MATS.classes)){
    for(const m of MATS.classes[cls]){
      items.push({ id:m.name, label:m.name, group:cls,
        thumb:`api/material_ball?name=${encodeURIComponent(m.name)}&res=64`,
        keywords:`${cls} metallic ${m.metallic} rough ${m.roughness}`+(m.ior?` glass ior ${m.ior}`:''),
        meta:m });
    }
  }
  return items;
}
function buildSwatches(){
  // one reusable Browser: folders = material classes, search over all 141+ materials (closes the "no search,
  // no folders, 9px hover labels" defect). Double-click applies; single-click selects.
  const host = $('matbrowser'); if(!host) return;
  const items = matItems();
  const selectFn = m => { MATNAME=m.id; buildMatbar(); refreshAttrs(); if(MATBROWSER) MATBROWSER.setSelected(m.id); };
  const applyFn  = m => { MATNAME=m.id; buildMatbar(); applyMaterial(MODE==='face' && selFaces.size ? 'sel' : 'all'); };
  if(!MATBROWSER){
    MATBROWSER = buildBrowser(host, { items, view:'grid', sortKeys:['label','group'],
      selectedId:MATNAME, onSelect:selectFn, onActivate:applyFn });
  } else {
    MATBROWSER.setItems(items); MATBROWSER.setSelected(MATNAME);
  }
}
function buildMatTabs(){ /* folded into the Browser's folder tree; kept as a no-op so callers stay valid */ }
async function applyMaterial(scope){
  if(!MATNAME){ status('Pick a material first (click a swatch in the Material editor — M).'); openDlg('dlg-material'); return; }
  if(!ACTIVE){ status('Select an object first (click it in the viewport), then apply the material.'); return; }
  const objName = (OBJS.get(ACTIVE)?.d?.name) || 'object';
  const body = {object:ACTIVE, material:MATNAME};
  let where;
  if(scope==='sel'){
    if(MODE!=='face' || !selFaces.size){ status('No faces selected — switch to Face mode and select faces, or use "Apply to object".'); return; }
    body.faces = [...selFaces]; where = `${selFaces.size} face(s) of ${objName}`;
  } else if(selObjs.size > 1){                      // PER-GROUP: the multi-selection is the group
    body.objects = [...selObjs]; delete body.object; where = `${selObjs.size} selected objects`;
  } else { body.all = true; where = `all of ${objName}`; }
  try{
    applyResp(await api('assign', body));
    status(`${MATNAME} → ${where}.`);
  }catch(e){ status('Apply failed: '+e.message); }
}
$('matSel').onclick = ()=>applyMaterial('sel');
$('matAll').onclick = ()=>applyMaterial('all');

/* ================================ uv ================================ */
$('uvbtn').onclick = async ()=>{
  if(!ACTIVE) return;
  $('uvbtn').disabled = true; $('uvmeta').textContent = 'unwrapping…';
  try{
    const d = await api('uv', {object:ACTIVE});
    const fl = d.distortion.flipped;
    $('uvmeta').textContent = `${d.method} — median distortion ${d.distortion.median}` +
      (fl !== null ? `, ${fl} flipped face(s)` : '') + ` (${d.seconds}s)`;
  }catch(e){ $('uvmeta').textContent = 'failed: '+e.message; }
  $('uvbtn').disabled = false;
};
$('uvmaps').onclick = ()=>{ if(needActiveFor('UV maps')) location.href='api/uv/maps.zip?object='+ACTIVE; };
function needActiveFor(what){                                 // F3: say so instead of doing nothing
  if(ACTIVE) return true;
  status('Select an object first — '+what+' exports the active object.');
  notify('Nothing selected: '+what+' exports the active object. Click one in the viewport or the Objects panel (N).');
  return false;
}
$('glbexp').onclick = ()=>{ if(needActiveFor('.glb')) location.href='api/export_glb?object='+ACTIVE; };

/* ================================ glb import ================================ */
$('glbbtn').onclick = ()=>$('glbfile').click();
let IMP_FILE = null;
$('glbfile').onchange = ()=>{
  const f = $('glbfile').files[0]; if(!f) return;
  IMP_FILE = f;
  $('impfname').textContent = f.name + ' · ' + (f.size/1048576).toFixed(1) + ' MB';
  openDlg('dlg-import');
};
$('impclose') && ($('impclose').onclick = $('impcancel').onclick = ()=>{ closeDlg('dlg-import'); IMP_FILE=null; $('glbfile').value=''; });
document.querySelectorAll('input[name="impmode"]').forEach(rb=>{ rb.onchange = ()=>{
  $('implabel').textContent = rb.value==='retopo' ? 'faces (guides resolution)'
                            : rb.value==='voxel' ? 'faces (guides voxel res)'
                            : rb.value==='rebake' ? 'faces (guides atlas grid)' : 'faces';
  $('imptarget').disabled = (rb.value==='asis' || rb.value==='auto');
  $('impreproject').disabled = (rb.value==='asis' || rb.value==='auto' || rb.value==='rebake');
};});
$('impgo') && ($('impgo').onclick = async ()=>{
  const f = IMP_FILE; if(!f) return;
  closeDlg('dlg-import');
  const mode = document.querySelector('input[name="impmode"]:checked').value;
  const q = `?mode=${mode}&target=${$('imptarget').value|0}&reproject=${$('impreproject').checked?1:0}`;
  status(`Importing ${f.name} (${mode})… big files can take a while.`);
  const buf = await f.arrayBuffer();
  try{
    const r = await fetch('api/import_glb'+q, {method:'POST', body:buf});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error||r.status);
    applyResp(d);
    const notes = (d.lod_report&&d.lod_report.groups||[]).map(g=>g.processing||g.route).join(' · ');
    status('Imported: ' + d.imported.map(m=>`${m.object}→${m.preset}`).join(', ') + (notes?'  ['+notes+']':''));
  }catch(e){ status('Import failed: '+e.message); }
  IMP_FILE=null; $('glbfile').value='';
});

/* ================================ renders ================================ */
let renderTimer=null, renderBusy=false, renderQueued=false, photoBusy=false, camTimer=null;
$('quality').addEventListener('input', ()=>{ $('qval').textContent=$('quality').value; scheduleRender(); });
$('live').addEventListener('change', ()=>scheduleRender());
$('adaptive').addEventListener('change', ()=>scheduleRender());
$('brad').addEventListener('input', ()=>$('bradv').textContent=$('brad').value);
$('bstr').addEventListener('input', ()=>$('bstrv').textContent=$('bstr').value);
$('btgt').addEventListener('input', ()=>$('btgtv').textContent=$('btgt').value);
function camParams(){
  const e=camera.position, t=camTarget;
  const fov = ($('fov') && $('fov').value) || 45;
  const lp = $('lightaz') ? `&light_az=${$('lightaz').value}&light_el=${$('lightel').value}&sun=${$('sun').value}&ambient=${$('ambient').value}` : '';
  return `fov=${fov}&eye=${e.x.toFixed(3)},${e.y.toFixed(3)},${e.z.toFixed(3)}`+
         `&target=${t.x.toFixed(3)},${t.y.toFixed(3)},${t.z.toFixed(3)}`+
         `&grid=${$('pgrid').value}`+lp+clipParam();
}
function camMoved(){
  PROG.token++;
  if(PROG.ctrl){ try{ PROG.ctrl.abort(); }catch(_){} PROG.ctrl=null; }   // kill the in-flight round NOW
  if(GPU.on) gpuFrame();                                                  // GPU path redraws live during orbit
  clearTimeout(camTimer);
  camTimer=setTimeout(()=>{
    // CAMERA-FIRST: on settle, go STRAIGHT to the progressive resolve — its first round is the settle
    // frame (no duplicate /api/render), and every later round only sharpens it.
    if(previewWanted()){ startProgressive(); }
    else scheduleRender();
  }, 200);
}
function scheduleRender(){
  PROG.token++;                                            // scene changed — stop any in-flight resolve loop
  if(GPU.on){ gpuRefreshScene().then(()=>gpuFrame()); }   // edit changed the scene -> re-emit WGSL, re-dispatch
  if(!previewWanted()) return;                             // B1
  clearTimeout(renderTimer);
  renderTimer=setTimeout(doPreview, 300);
}
const SESSION_ID = 'ps-' + Math.random().toString(36).slice(2) + Date.now().toString(36);
async function doPreview(){
  if(!previewWanted()) return;                              // B1: no hidden server work behind a shut panel
  if(renderBusy){ renderQueued=true; return; }
  renderBusy=true; rvSpin(true);
  try{
    const q = $('adaptive').checked ? `session=${SESSION_ID}&target_fps=30` : `quality=${$('quality').value}`;
    const r=await fetch(`api/render?${q}&`+camParams());
    if(!r.ok) throw new Error(r.status);
    const blob=await r.blob();
    if(previewWanted()){
      const url=URL.createObjectURL(blob);
      if(!rvShow(url,'live')) URL.revokeObjectURL(url);      // A3 guard said no -> don't leak the blob
      const _hdr=r.headers.get('X-Holostuff-Render')||'';
      rvMeta(_hdr); surfaceRenderWarn(_hdr); perfIngestHeader(_hdr);
      startProgressive();
    }else{ URL.revokeObjectURL(URL.createObjectURL(blob)); }
  }catch(e){ rvMeta('preview failed: '+e.message); }
  rvSpin(false);
  renderBusy=false;
  if(renderQueued){ renderQueued=false; doPreview(); }
}
function rvSpin(on){ const s=$('spin'); if(s) s.hidden=!on; }

/* ================================ A2/A3/A4/A6/G1: the Render View ================================
   A framebuffer, in the sense every render engine means it: an image surface with a toolbar, its own
   zoom/pan, its own history, its own progress, and — the part that was actually broken — its own
   image buffer.

   The bug this replaces: preview, path-traced photo and engine render all wrote into ONE <img> inside
   the settings panel. Finishing a photo cleared photoBusy, so the very next orbit started a preview
   that silently overwrote the finished render (A3). Nothing could be cancelled (A4), nothing was kept
   (G1), and the whole preview/resolve pipeline ran even with the panel shut (B1).

   Ownership rule, enforced in exactly one place (rvShow):
     slot 'live'  -- the server preview and the progressive resolve may write here.
     slot 'final' -- a photo or engine render. The preview may NEVER write over it. It stays until the
                     user renders again, picks a history entry, or presses Live.                     */
const RV = {
  slot: 'live',              // which producer currently owns the surface
  zoom: 1, fit: true, ox: 0, oy: 0, natW: 0, natH: 0,
  hist: [], histCap: 12, seq: 0,
  A: null, B: null, wipe: false, wipeX: 0.5,
  keep: new Set(),           // blob URLs the history / A / B still point at (B3)
  ctrl: null,                // AbortController for the render in flight (A4)
  busy: false, kind: null,
  lastFinal: null,           // {src, meta, w, h, kind, sessionKey} — what Save/2x act on
};

function rvEsc(x){ return String(x==null?'':x).replace(/[&<>"]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[m])); }
function rvOpen(){ openDlg('dlg-renderview'); rvLayout(); }
function rvIsOpen(){ const d=$('dlg-renderview'); return !!(d && d.classList.contains('open')); }
// B1: the preview is SERVER work. It runs only when someone can actually see it.
function previewWanted(){ return rvIsOpen() && $('live').checked && RV.slot==='live' && !RV.busy; }

/* ---------------- the one place an image is allowed onto the surface ---------------- */
/* B3: a 1280x960 PNG is ~1.5 MB; as a base64 data-URL it is a ~2 MB JavaScript STRING, and the history
   holds twelve of them. Frames arrive base64 inside the NDJSON stream, so we decode once, on arrival,
   and everything downstream is a blob URL. Anything the history or an A/B slot still points at is
   registered in RV.keep so the display swap cannot revoke it out from under a thumbnail. */
function rvBlobFromB64(b64){
  return URL.createObjectURL(new Blob([b64ToBytes(b64)], {type:'image/png'}));
}
function rvKeep(url){ if(url && url.startsWith('blob:')) RV.keep.add(url); }
function rvRelease(url){
  if(!url || !url.startsWith('blob:')) return;
  RV.keep.delete(url);
  if($('rvimg').src!==url && $('rvimgB').src!==url) URL.revokeObjectURL(url);
}
function rvShow(src, slot, meta){
  if(slot==='live' && RV.slot==='final') return false;      // A3: never clobber a finished render
  const img = $('rvimg');
  const old = img.src;
  img.src = src;
  img.classList.add('show');
  const em=$('rvempty'); if(em) em.style.display='none';
  // B2: no leak -- but only free what nothing else is holding
  if(old && old.startsWith('blob:') && old!==src && !RV.keep.has(old)) URL.revokeObjectURL(old);
  RV.slot = slot;
  if(meta!==undefined) rvMeta(meta);
  if(RV.fit) img.addEventListener('load', rvFitOnce, {once:true});
  return true;
}
function rvFitOnce(){ rvZoomFit(); }
function rvMeta(html){ const m=$('rvmeta'); if(!m) return; if(/<[a-z]/i.test(html)) m.innerHTML=html; else m.textContent=html; }
function rvProg(pct){ const p=$('rvprog'); if(p) p.firstElementChild.style.width=Math.max(0,Math.min(100,pct))+'%'; }

/* ---------------- zoom + pan (VFB convention: fit / 1:1 / wheel / drag, dbl-click = 1:1) ---------------- */
function rvApplyTransform(){
  for(const id of ['rvimg','rvimgB']){
    const el=$(id); if(!el) continue;
    el.style.transform = `translate(${RV.ox}px, ${RV.oy}px) scale(${RV.zoom})`;
  }
  const lbl=$('rvzoomfit'); if(lbl) lbl.classList.toggle('on', RV.fit);
  const one=$('rvzoom100'); if(one) one.classList.toggle('on', !RV.fit && Math.abs(RV.zoom-1)<0.001);
}
function rvStageSize(){ const st=$('rvstage'); return [st.clientWidth, st.clientHeight]; }
function rvZoomFit(){
  const img=$('rvimg'); if(!img || !img.naturalWidth) return;
  RV.natW=img.naturalWidth; RV.natH=img.naturalHeight;
  const [sw,sh]=rvStageSize();
  RV.zoom = Math.min(sw/RV.natW, sh/RV.natH);
  RV.ox = (sw - RV.natW*RV.zoom)/2; RV.oy = (sh - RV.natH*RV.zoom)/2;
  RV.fit = true; rvApplyTransform();
}
function rvZoomTo(z, cx, cy){
  const img=$('rvimg'); if(!img || !img.naturalWidth) return;
  const [sw,sh]=rvStageSize();
  if(cx===undefined){ cx=sw/2; cy=sh/2; }
  const k = z/RV.zoom;
  RV.ox = cx - (cx-RV.ox)*k; RV.oy = cy - (cy-RV.oy)*k;
  RV.zoom = z; RV.fit=false; rvApplyTransform();
}
function rvLayout(){ if(RV.fit) rvZoomFit(); else rvApplyTransform(); }

/* ---------------- history (G1): thumbnails, click to reload, keys 1-9 ---------------- */
function rvPush(entry){
  entry.n = ++RV.seq;
  rvKeep(entry.src);
  RV.hist.unshift(entry);
  if(RV.hist.length>RV.histCap){
    const drop=RV.hist.pop();
    const held = (RV.A && RV.A.src===drop.src) || (RV.B && RV.B.src===drop.src);
    if(!held) rvRelease(drop.src);
  }
  rvRenderHist();
}
function rvRenderHist(){
  const box=$('rvhist'); if(!box) return;
  box.innerHTML = RV.hist.map((h,i)=>
    `<div class="rvthumb" data-i="${i}" role="button" tabindex="0" title="${rvEsc(h.label||'')}">`+
    `<b>${i+1}</b><img src="${h.src}" alt=""><span class="tl">${rvEsc(h.short||'')}</span></div>`).join('');
  box.querySelectorAll('.rvthumb').forEach(t=>{
    const load=()=>rvLoadHist(+t.dataset.i);
    t.onclick=load;
    t.onkeydown=ev=>{ if(ev.key==='Enter'||ev.key===' '){ ev.preventDefault(); load(); } };
  });
}
function rvLoadHist(i){
  const h=RV.hist[i]; if(!h) return;
  RV.slot='live'; rvShow(h.src, 'final', h.label);          // 'live' first so rvShow's guard lets it through
  RV.lastFinal = h;
  $('rvlive').classList.remove('on');
  status('Loaded render '+(i+1)+' from history.');
}

/* ---------------- A/B compare (G1): two slots and a draggable wipe ---------------- */
function rvSetSlot(which){
  const img=$('rvimg');
  if(!img.classList.contains('show')){ status('Nothing to store yet — render something first.'); return; }
  rvKeep(img.src);
  RV[which] = { src: img.src, label: $('rvmeta').textContent };
  status(which+' set. Press A/B to wipe between them.');
  $('rv'+(which==='A'?'setA':'setB')).classList.add('on');
  if(RV.A && RV.B) rvWipe(true);
}
function rvWipe(on){
  RV.wipe = (on===undefined) ? !RV.wipe : on;
  if(RV.wipe && !(RV.A && RV.B)){ status('Set both A and B first.'); RV.wipe=false; return; }
  const b=$('rvimgB'), line=$('rvwipeline');
  $('rvwipetog').classList.toggle('on', RV.wipe);
  if(RV.wipe){
    rvShow(RV.A.src, RV.slot==='final'?'final':'live');
    b.src=RV.B.src; b.classList.add('show'); line.classList.add('show');
    rvWipeAt(RV.wipeX);
    rvMeta('A / B wipe — drag the divider');
  }else{
    b.classList.remove('show'); line.classList.remove('show');
  }
  rvApplyTransform();
}
function rvWipeAt(f){
  RV.wipeX=Math.max(0,Math.min(1,f));
  const [sw]=rvStageSize();
  $('rvimgB').style.clipPath=`inset(0 0 0 ${RV.wipeX*100}%)`;
  $('rvwipeline').style.left=(RV.wipeX*sw)+'px';
}

/* ---------------- the render itself ---------------- */
function rvSetBusy(on, kind){
  RV.busy=on; RV.kind=kind||null;
  $('rvgo').disabled=on; $('rvcancel').disabled=!on;
  if(!on) setTimeout(()=>{ if(!RV.busy) rvProg(0); }, 1200);
}
function rvCancel(){
  // Two halves. Aborting the fetch closes the stream, which trips the server's finally clause (F4);
  // the explicit call is immediate and doesn't wait for the socket teardown to be noticed.
  try{ navigator.sendBeacon ? navigator.sendBeacon('api/render_cancel?session='+SESSION_ID)
                            : fetch('api/render_cancel?session='+SESSION_ID, {method:'POST'}); }catch(_){ }
  if(RV.ctrl){ try{ RV.ctrl.abort(); }catch(_){ } }
  RV.ctrl=null;
  rvMeta('cancelled');
}
// leaving the page mid-render should not leave the server tracing either
addEventListener('pagehide', ()=>{ if(RV.busy){ try{ navigator.sendBeacon('api/render_cancel?session='+SESSION_ID); }catch(_){ } } });
function photoQuery(){
  // A5: real output controls. Height follows the chosen aspect (or the live viewport).
  const w = +$('photow').value;
  const asp = $('photoaspect').value;
  const vw=$('viewwrap');
  const ratio = asp==='view' ? (vw.clientHeight/Math.max(1,vw.clientWidth)) : parseFloat(asp);
  const h = Math.max(120, Math.round(w*ratio));
  const dofq = $('dof') && $('dof').checked ? `&dof=1&focus=${$('focus').value}&fstop=${$('fstop').value}` : '&dof=0';
  const bg = $('bgmode').value==='solid' ? '&bg='+hexToRgb($('bgcolor').value).map(x=>x.toFixed(3)).join(',') : '';
  const post = `&exposure=${$('exposure').value}&sharpen=${$('sharpen').value}`+($('aov').value?`&aov=${$('aov').value}`:'');
  return `w=${w}&h=${h}&spp=${$('spp').value}&fog=${$('fog').checked?1:0}&grid=${$('photogrid').value}`+
         dofq+bg+post+`&session=${SESSION_ID}&`+camParams().replace(/&grid=\d+/,'');
}
async function rvRenderPhoto(){
  rvSetBusy(true,'photo'); RV.slot='live';                 // release the previous final so the new one lands
  rvProg(2); rvMeta('tracing…');
  RV.ctrl = new AbortController();
  const t0=performance.now();
  let lastSrc=null, meta=null, ok=false;
  try{
    const r = await fetch('api/photo?'+photoQuery(), {signal:RV.ctrl.signal});
    if(!r.ok) throw new Error('HTTP '+r.status);
    const reader=r.body.getReader(), dec=new TextDecoder();
    let buf='';
    while(true){
      const {done, value} = await reader.read();
      if(done) break;
      buf += dec.decode(value,{stream:true});
      let nl;
      while((nl=buf.indexOf('\n'))>=0){
        const line=buf.slice(0,nl); buf=buf.slice(nl+1);
        if(!line.trim()) continue;
        const d=JSON.parse(line);
        if(d.type==='meta'){ meta=d; }
        else if(d.type==='frame'){
          const prev = lastSrc;
          lastSrc = rvBlobFromB64(d.png);                     // B3: blob, not a megabyte-long string
          RV.slot='live'; rvShow(lastSrc,'final');
          if(prev && !RV.keep.has(prev)) URL.revokeObjectURL(prev);   // intermediate samples are disposable
          rvProg(100*d.done/((meta&&meta.spp)||1));
          rvMeta(`tracing ${d.done}/${meta?meta.spp:'?'} spp — ${((performance.now()-t0)/1000|0)}s`);
        }
        else if(d.type==='done'){ ok=true; rvProg(100); }
        else if(d.type==='error'){ throw new Error(d.error); }
      }
    }
  }catch(e){
    if(e.name==='AbortError'){ rvMeta(lastSrc?'cancelled — partial render kept':'cancelled'); }
    else rvMeta('photo failed: '+e.message);
  }
  if(lastSrc){
    const label = (ok?'photo ':'partial ')+(meta?`${meta.w}\u00d7${meta.h}, ${meta.spp} spp`:'')+
                  ` — ${((performance.now()-t0)/1000|0)}s`;
    RV.lastFinal = {src:lastSrc, label, kind:'photo', w:meta&&meta.w, h:meta&&meta.h, post:ok};
    rvPush({src:lastSrc, label, short:(meta?meta.spp+' spp':'photo'), kind:'photo', post:ok});
    if(ok) rvMeta(label);
    $('rvlive').classList.remove('on');
  }
  RV.ctrl=null; rvSetBusy(false);
}
async function rvRenderEngine(){
  rvSetBusy(true,'engine'); RV.slot='live';
  rvMeta('engine: resolving…'); rvProg(5);
  RV.ctrl = new AbortController();
  const hero = (selObjs && selObjs.size===1) ? '&hero='+[...selObjs][0] : (ACTIVE ? '&hero='+ACTIVE : '');
  const baseW = +$('photow').value;
  const asp = $('photoaspect').value;
  const vw=$('viewwrap');
  const ratio = asp==='view' ? (vw.clientHeight/Math.max(1,vw.clientWidth)) : parseFloat(asp);
  const ROUNDS = [[0.35,'\u2153'],[0.65,'\u2154'],[1,'full']];
  let mode='', lastSrc=null;
  try{
    for(let i=0;i<ROUNDS.length;i++){
      const [f,tag]=ROUNDS[i];
      const w=Math.round(baseW*f), h=Math.round(w*ratio);
      rvMeta(`engine: resolving… (${tag})`); rvProg(5+90*(i/ROUNDS.length));
      const r = await fetch(`api/render_engine?w=${w}&h=${h}${hero}&`+camParams(), {signal:RV.ctrl.signal});
      if(!r.ok){ const e=await r.json().catch(()=>({})); throw new Error(e.error||('HTTP '+r.status)); }
      mode = r.headers.get('X-Render-Mode')||'';
      const url = URL.createObjectURL(await r.blob());
      if(lastSrc && lastSrc.startsWith('blob:')) URL.revokeObjectURL(lastSrc);
      lastSrc=url; RV.slot='live'; rvShow(url,'final');
    }
    rvProg(100);
  }catch(e){
    if(e.name==='AbortError') rvMeta(lastSrc?'cancelled — partial render kept':'cancelled');
    else rvMeta('engine render failed: '+e.message);
  }
  if(lastSrc){
    const label='engine rasteriser'+(mode?' ('+mode+')':'');
    RV.lastFinal={src:lastSrc, label, kind:'engine'};
    rvPush({src:lastSrc, label, short:'engine', kind:'engine'});
    rvMeta(label);
    $('rvlive').classList.remove('on');
  }
  RV.ctrl=null; rvSetBusy(false);
}
function rvRender(){
  if(RV.busy) return;
  if($('rvengine').value==='engine') rvRenderEngine(); else rvRenderPhoto();
}

/* ---------------- save / upscale (A6: never destroy the original) ---------------- */
function rvStamp(){ const d=new Date(); const p=n=>String(n).padStart(2,'0');
  return `${d.getFullYear()}${p(d.getMonth()+1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}${p(d.getSeconds())}`; }
function rvSave(){
  const img=$('rvimg');
  if(!img.classList.contains('show')){ status('Nothing to save yet.'); return; }
  const kind=(RV.lastFinal&&RV.lastFinal.kind)||'preview';
  const a=document.createElement('a'); a.href=img.src; a.download=`polystudio_${kind}_${rvStamp()}.png`; a.click();
  status('Saved '+a.download);
}
// B3 fallout: renders are blob URLs now, and the server decodes base64. Posting `img.src` sent it the
// literal string "blob:null/<uuid>", which it cannot fetch -- the 2x button failed with "could not
// read image". Read the blob back out and send the bytes.
async function rvImageAsDataURL(){
  const src = $('rvimg').src;
  if(src.startsWith('data:')) return src;
  const blob = await (await fetch(src)).blob();
  return await new Promise((res, rej)=>{
    const r = new FileReader();
    r.onload = ()=>res(r.result); r.onerror = ()=>rej(new Error('could not read the render'));
    r.readAsDataURL(blob);
  });
}
async function rvUpscale(){
  const img=$('rvimg');
  if(!img.classList.contains('show')){ status('Render something first.'); return; }
  rvMeta('upscaling 2\u00d7…');
  try{
    const dataURL = await rvImageAsDataURL();
    const r=await (await fetch('api/upscale',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({image:dataURL, scale:2.0})})).json();
    if(r.error){ rvMeta('upscale failed: '+r.error); return; }
    const label=`upscaled to ${r.w}\u00d7${r.h} (${r.note})`;
    const url = r.png.startsWith('data:') ? rvBlobFromB64(r.png.replace(/^data:[^,]+,/,'')) : r.png;
    rvPush({src:url, label, short:'2\u00d7 '+r.w, kind:'upscale'});     // A6: a NEW entry, original untouched
    RV.slot='live'; rvShow(url,'final',label);
    RV.lastFinal={src:url, label, kind:'upscale'};
  }catch(e){ rvMeta('upscale failed: '+e.message); }
}

/* ---------------- G1: post without re-tracing ---------------- */
// exposure / sharpen / AOV used to be render-time query parameters, so nudging exposure meant tracing
// the whole image again. The server now caches the traced HDR per session; this re-tonemaps it.
let rvPostTimer=null;
function rvPostChanged(){
  if(!rvIsOpen()) return;
  if(!(RV.lastFinal && RV.lastFinal.kind==='photo' && RV.lastFinal.post)) return;   // only a finished photo
  clearTimeout(rvPostTimer);
  rvPostTimer=setTimeout(async ()=>{
    try{
      const q=`session=${SESSION_ID}&exposure=${$('exposure').value}&sharpen=${$('sharpen').value}`+
              ($('aov').value?`&aov=${$('aov').value}`:'');
      const r=await fetch('api/photo_post?'+q);
      if(!r.ok) return;                                    // no cached HDR (e.g. after a restart) — ignore
      const url=URL.createObjectURL(await r.blob());
      RV.slot='live'; rvShow(url,'final');
      RV.lastFinal.src=url;
      rvMeta(`post: exposure ${(+$('exposure').value).toFixed(2)} · sharpen ${(+$('sharpen').value).toFixed(2)}`);
    }catch(_){ }
  }, 180);
}

/* ---------------- wiring ---------------- */
(function rvWire(){
  const st=$('rvstage'); if(!st) return;
  $('rvgo').onclick = rvRender;
  $('rvcancel').onclick = rvCancel;
  $('rvsettings').onclick = ()=>openDlg('dlg-render');
  $('rvopenfromsettings') && ($('rvopenfromsettings').onclick = rvOpen);
  $('rvsave').onclick = rvSave;
  $('rvupscale').onclick = rvUpscale;
  $('rvsetA').onclick = ()=>rvSetSlot('A');
  $('rvsetB').onclick = ()=>rvSetSlot('B');
  $('rvwipetog').onclick = ()=>rvWipe();
  $('rvhisttog').onclick = ()=>{ const h=$('rvhist'); h.classList.toggle('show');
    $('rvhisttog').classList.toggle('on', h.classList.contains('show')); rvLayout(); };
  $('live') && ($('live').addEventListener('change', ()=>{
    $('rvlive').classList.toggle('on', $('live').checked);   // one state, two controls
    if($('live').checked){ RV.slot='live'; scheduleRender(); }
  }));
  $('rvzoomfit').onclick = rvZoomFit;
  $('rvzoom100').onclick = ()=>rvZoomTo(1);
  $('rvzoomin').onclick  = ()=>rvZoomTo(Math.min(16, RV.zoom*1.25));
  $('rvzoomout').onclick = ()=>rvZoomTo(Math.max(0.05, RV.zoom/1.25));
  $('rvlive').onclick = ()=>{
    const on = !$('rvlive').classList.contains('on');
    $('rvlive').classList.toggle('on', on);
    if(on){ RV.slot='live'; $('live').checked=true; scheduleRender(); }
    else { $('live').checked=false; PROG.token++; if(PROG.ctrl){ try{PROG.ctrl.abort();}catch(_){} } }
  };
  // wheel zoom about the cursor; drag to pan; double-click = 1:1 (VFB convention)
  st.addEventListener('wheel', ev=>{
    if(!$('rvimg').classList.contains('show')) return;
    ev.preventDefault();
    const r=st.getBoundingClientRect();
    rvZoomTo(Math.max(0.05, Math.min(16, RV.zoom*(1-ev.deltaY*0.0015))), ev.clientX-r.left, ev.clientY-r.top);
  }, {passive:false});
  st.addEventListener('dblclick', ()=>rvZoomTo(1));
  st.addEventListener('pointerdown', ev=>{
    if(ev.button!==0) return;
    const r=st.getBoundingClientRect();
    if(RV.wipe){                                            // grab the divider if the press is near it
      const [sw]=rvStageSize();
      if(Math.abs((ev.clientX-r.left) - RV.wipeX*sw) < 14){
        const mv=e2=>rvWipeAt((e2.clientX-r.left)/sw);
        const up=()=>{ removeEventListener('pointermove',mv); removeEventListener('pointerup',up); };
        addEventListener('pointermove',mv); addEventListener('pointerup',up); ev.preventDefault(); return;
      }
    }
    const sx=ev.clientX, sy=ev.clientY, ox=RV.ox, oy=RV.oy;
    st.classList.add('panning');
    const mv=e2=>{ RV.ox=ox+(e2.clientX-sx); RV.oy=oy+(e2.clientY-sy); RV.fit=false; rvApplyTransform(); };
    const up=()=>{ st.classList.remove('panning'); removeEventListener('pointermove',mv); removeEventListener('pointerup',up); };
    addEventListener('pointermove',mv); addEventListener('pointerup',up); ev.preventDefault();
  });
  // D3: drag-resize the whole dialog; the stage takes the slack
  const grip=$('rvgrip'), dlg=$('dlg-renderview');
  grip.addEventListener('pointerdown', ev=>{
    const r=dlg.getBoundingClientRect(), sx=ev.clientX, sy=ev.clientY;
    const stH=$('rvstage').clientHeight;
    const mv=e2=>{
      dlg.style.width  = Math.max(420, r.width  + (e2.clientX-sx))+'px';
      $('rvstage').style.height = Math.max(180, stH + (e2.clientY-sy))+'px';
      rvLayout();
    };
    const up=()=>{ removeEventListener('pointermove',mv); removeEventListener('pointerup',up);
      try{ localStorage.setItem('polystudio:rvsize', JSON.stringify({w:dlg.style.width,h:$('rvstage').style.height})); }catch(_){ } };
    addEventListener('pointermove',mv); addEventListener('pointerup',up); ev.preventDefault();
  });
  try{ const sz=JSON.parse(localStorage.getItem('polystudio:rvsize')||'null');
       if(sz){ dlg.style.width=sz.w; $('rvstage').style.height=sz.h; } }catch(_){ }
  addEventListener('resize', ()=>{ if(rvIsOpen()) rvLayout(); });
  // post sliders re-tonemap the cached HDR instead of re-tracing (G1)
  ['exposure','sharpen'].forEach(id=>{ const el=$(id); if(el) el.addEventListener('input', rvPostChanged); });
  // The diagnostic passes are traced by a separate gbuffer path and are not in the post cache, so
  // switching one is a re-render, not a re-grade. Say so rather than appearing to do nothing.
  $('aov') && ($('aov').onchange = ()=>{
    if(!rvIsOpen()) return;
    rvMeta($('aov').value ? 'Diagnostic pass changed — press Render to trace it.'
                          : 'Back to beauty — press Render to trace it.');
  });
  // A5/A7: live value labels for the new settings
  const spp=$('spp'); if(spp) spp.addEventListener('input', ()=>{ $('sppval').textContent=spp.value; });
  // keys inside the Render View: Enter renders, Esc cancels, 1-9 recall history (VFB convention)
  // (Enter renders, Esc cancels, 1-9 recall history -- KEYMAP, C4)
})();


/* ================================ tool buttons + keyboard ================================ */
function needFace(){ const f=[...selFaces][0]; if(f===undefined){ status('Select a face (Face mode).'); return null;} return f; }
function needVert(){ const v=[...selVerts][0]; if(v===undefined){ status('Select a vertex (Vertex mode).'); return null;} return v; }
async function extrudeSel(){
  if(MODE!=='face' || !selFaces.size){ status('Select polygon(s) first.'); return; }
  // extrude each selected face in turn (indices shift; re-pick by centroid after each op)
  const o = activeObj(); const cents = [...selFaces].map(f=>{
    const P=o.d.positions, fc=o.d.faces[f]; const c=[0,0,0];
    for(const v of fc){ c[0]+=P[v*3]; c[1]+=P[v*3+1]; c[2]+=P[v*3+2]; }
    return c.map(x=>x/fc.length);
  });
  for(const c of cents){
    const d = OBJS.get(ACTIVE).d;
    let best=-1, bd=1e9;
    d.faces.forEach((f,i)=>{
      const P=d.positions; let cx=0,cy=0,cz=0;
      for(const v of f){ cx+=P[v*3]; cy+=P[v*3+1]; cz+=P[v*3+2]; }
      cx/=f.length; cy/=f.length; cz/=f.length;
      const dd=(cx-c[0])**2+(cy-c[1])**2+(cz-c[2])**2;
      if(dd<bd){bd=dd;best=i;}
    });
    if(best>=0) await runOp('extrude', {object:ACTIVE, face:best, dist:0.3});
  }
}
$('tExtrude').onclick=extrudeSel;
$('tInset').onclick=()=>{ const f=needFace(); if(f!=null) runOp('inset',{object:ACTIVE, face:f, ratio:0.3}); };
$('tLoop').onclick=()=>{ const f=needFace(); if(f==null) return;
  if(!lastHit || lastHit.poly!==f || lastHit.id!==ACTIVE){ status('Click the polygon near the edge to cut.'); return; }
  runOp('loopcut',{object:ACTIVE, face:f, edge:lastHit.edge}); };
$('tBevel').onclick=()=>{ const v=needVert(); if(v!=null) runOp('bevel',{object:ACTIVE, vertex:v, ratio:0.25, segments:parseInt($('bevsegs').value)||1}); };
$('tDissolve').onclick=()=>{ const v=needVert(); if(v!=null) runOp('dissolve',{object:ACTIVE, vertex:v}); };
$('tPoke').onclick=()=>{ const f=needFace(); if(f!=null) runOp('poke',{object:ACTIVE, face:f, height:0.25}); };
$('tSubdiv').onclick=()=>ACTIVE&&runOp('subdivide',{object:ACTIVE});
$('tSmoothM').onclick=()=>ACTIVE&&runOp('smooth',{object:ACTIVE, iters:6});
$('tSmoothLimit').onclick=()=>ACTIVE&&runOp('smooth_limit',{object:ACTIVE});
$('tSolid').onclick=()=>ACTIVE&&runOp('solidify',{object:ACTIVE, thickness:0.12});

/* ---- engine-backed topology selection (holographic_meshselect) ---- */
let SYMAXIS = 0;
async function applySelResult(r, add){
  if(r.error){ status(r.error); return; }
  // switch mode FIRST -- setMode clears the selection sets, so populating before it would be wiped
  const wantMode = r.mode==='face' ? 'face' : 'vertex';
  if(MODE!==wantMode) await setMode(wantMode);
  if(!add){ selVerts.clear(); selFaces.clear(); }
  if(r.mode==='face') (r.faces||[]).forEach(f=>selFaces.add(f));
  else (r.verts_touched||[]).forEach(v=>selVerts.add(v));
  status(`Selected ${r.faces? r.faces.length+' face(s)' : (r.edges? r.edges.length+' edge(s)' : (r.verts||r.verts_touched||[]).length+' point(s)')} (leCore)`);
  refreshSelectionVisuals(); refreshAttrs();
}
$('selLoop').onclick=async()=>{
  if(!ACTIVE) return;
  const vs=[...selVerts];
  if(vs.length<2){ status('Select two adjacent points that share an edge, then Edge loop.'); return; }
  try{ await applySelResult(await api('select',{object:ACTIVE,op:'edge_loop',seed_verts:[vs[0],vs[1]]}), false); }catch(e){}
};
$('selRing').onclick=async()=>{
  const f=[...selFaces][0];
  if(f===undefined){ status('Select a polygon, then Face ring.'); return; }
  try{ await applySelResult(await api('select',{object:ACTIVE,op:'face_ring',seed:f}), false); }catch(e){}
};
$('selBoundary').onclick=async()=>{
  if(!ACTIVE) return;
  try{ await applySelResult(await api('select',{object:ACTIVE,op:'boundary'}), false); }catch(e){}
};
$('selSymX').onclick=async()=>{
  if(!ACTIVE) return;
  const mode = MODE==='face' ? 'face' : 'vertex';
  const idx = mode==='face' ? [...selFaces] : [...selVerts];
  if(!idx.length){ status('Select something first, then Add mirror.'); return; }
  try{ await applySelResult(await api('select',{object:ACTIVE,op:'symmetric',elem:mode,indices:idx,axis:SYMAXIS}), true); }catch(e){}
};
for(const [id,ax] of [['symAxisX',0],['symAxisY',1],['symAxisZ',2]]){
  $(id).onclick=()=>{ SYMAXIS=ax; ['symAxisX','symAxisY','symAxisZ'].forEach(i=>$(i).classList.toggle('on', i===id)); };
}

/* ---- SDF shader export (holographic_sdfemit.sdf_dialect / SDF.to_glsl) ---- */
let SHADER_DIALECT='glsl';
async function doShaderExport(){
  if(!ACTIVE){ status('Select an object.'); return; }
  const r = await api_get(`export_shader?object=${ACTIVE}&dialect=${SHADER_DIALECT}`);
  $('shaderout').style.display='block';
  if(!r.analytic){
    $('shadermeta').textContent = r.reason || 'Not exportable.';
    $('shadercode').value = '';
    return;
  }
  const code = SHADER_DIALECT==='glsl' ? (r.shadertoy || r.map) : r.map;
  $('shadercode').value = code;
  const c = r.cost||{};
  let head;
  if(r.mode==='fitted'){
    const f = r.fit||{};
    const kinds = Object.entries(f.kinds||{}).map(([k,n])=>`${n}×${k}`).join(' + ');
    head = `<b>${r.dialect.toUpperCase()} · FITTED</b> (edited mesh → ${kinds}, `+
           `${f.quality}× better than 1 sphere) · cost: ${c.verdict||'?'} (≈${c.alu||'?'} ALU)`+
           `<br><span style="color:var(--warn,#d9a441)">Approximation of the sculpted surface, not the exact mesh — a union of SDF primitives fit to it.</span>`;
  } else {
    head = `<b>${r.dialect.toUpperCase()} · EXACT</b> (this object IS this SDF) · cost: ${c.verdict||'?'} `+
           `(≈${c.alu||'?'} ALU, ${c.nodes||'?'} nodes) · DSL: <code>${r.dsl||''}</code>`;
  }
  $('shadermeta').innerHTML = head;
}
async function api_get(path){
  const r = await fetch('api/'+path);
  return r.json();
}
$('shaderexp').onclick=doShaderExport;
$('shGlsl').onclick=()=>{ SHADER_DIALECT='glsl'; $('shGlsl').classList.add('on'); $('shWgsl').classList.remove('on'); doShaderExport(); };
$('shWgsl').onclick=()=>{ SHADER_DIALECT='wgsl'; $('shWgsl').classList.add('on'); $('shGlsl').classList.remove('on'); doShaderExport(); };
/* ================================ node editor (holographic_nodegraph) ================================ */
let NODE = {open:false, types:{}, graph:{nodes:[],edges:[]}, sel:null, multi:new Set(), pendingWire:null, drag:null, box:null};
let NODE_COMMENTS = [];
let NODE_FRAMES = [];                                          // {ids:[nodeId], title, color}
$('nodecomment') && ($('nodecomment').onclick = ()=>{
  const cv=$('nodecanvas').getBoundingClientRect();
  NODE_COMMENTS.push({x:40+Math.random()*60, y:40+Math.random()*40, text:''});
  renderNodes();
});
$('nodeframe') && ($('nodeframe').onclick = ()=>{
  const ids = NODE.multi.size ? [...NODE.multi] : (NODE.sel?[NODE.sel]:[]);
  if(ids.length<1){ status('Select nodes first (box-select or shift-click), then Frame.'); return; }
  NODE_FRAMES.push({ids, title:'Group', color:'#7aa8ff'});
  renderNodes();
});
async function nodeApi(body){ const r=await fetch('api/nodes/op',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}); return r.json(); }
async function nodeRefresh(){
  NODE.graph = await (await fetch('api/nodes/graph')).json();
  renderNodes();
}
function nodeSockPos(nid, sock, isOut){
  const el = document.querySelector(`.node[data-id="${nid}"] .dot[data-sock="${sock}"][data-out="${isOut?1:0}"]`);
  const cv = $('nodecanvas').getBoundingClientRect();
  if(!el) return null;
  const r = el.getBoundingClientRect();
  return {x:r.left-cv.left+r.width/2, y:r.top-cv.top+r.height/2};
}
function renderWires(){
  const svg = $('nodewires');
  let html='';
  for(const e of (NODE.graph.edges||[])){
    const a=nodeSockPos(e.src, e.src_socket||'out', true), b=nodeSockPos(e.dst, e.dst_socket, false);
    if(a&&b) html+=`<path d="M${a.x},${a.y} C${a.x+50},${a.y} ${b.x-50},${b.y} ${b.x},${b.y}" fill="none" stroke="#7aa2ff" stroke-width="1.6" opacity="0.85"/>`;
  }
  if(NODE.pendingWire && NODE.pendingWire.to)
    html+=`<path d="M${NODE.pendingWire.x},${NODE.pendingWire.y} L${NODE.pendingWire.to.x},${NODE.pendingWire.to.y}" stroke="#e0c060" stroke-width="1.4" stroke-dasharray="4 3" fill="none"/>`;
  svg.innerHTML = html;
}
function renderFramesOnly(){                                  // reposition frame boxes without a full re-render
  const cv=$('nodecanvas'); const boxes=cv.querySelectorAll('.nodeframe');
  let i=0;
  for(const fr of NODE_FRAMES){
    const members = fr.ids.map(id=>NODE.graph.nodes.find(n=>n.id===id)).filter(Boolean);
    if(!members.length) continue;
    const box=boxes[i++]; if(!box) continue;
    const xs=members.map(n=>n.pos[0]), ys=members.map(n=>n.pos[1]);
    const pad=18, x=Math.min(...xs)-pad, y=Math.min(...ys)-pad-16, w=Math.max(...xs)-Math.min(...xs)+150+pad*2, h=Math.max(...ys)-Math.min(...ys)+90+pad*2+16;
    box.style.left=x+'px'; box.style.top=y+'px'; box.style.width=w+'px'; box.style.height=h+'px';
  }
}
function renderNodes(){
  const cv = $('nodecanvas');
  cv.querySelectorAll('.node').forEach(n=>n.remove());
  cv.querySelectorAll('.nodecomment').forEach(n=>n.remove());
  cv.querySelectorAll('.nodeframe').forEach(n=>n.remove());
  NODE_FRAMES = NODE_FRAMES.filter(fr => fr.ids.some(id => NODE.graph.nodes.find(n=>n.id===id)));
  for(const fr of NODE_FRAMES){                              // labeled group box behind its member nodes
    const members = fr.ids.map(id=>NODE.graph.nodes.find(n=>n.id===id)).filter(Boolean);
    if(!members.length) continue;
    const xs=members.map(n=>n.pos[0]), ys=members.map(n=>n.pos[1]);
    const pad=18, x=Math.min(...xs)-pad, y=Math.min(...ys)-pad-16, w=Math.max(...xs)-Math.min(...xs)+150+pad*2, h=Math.max(...ys)-Math.min(...ys)+90+pad*2+16;
    const box=document.createElement('div');
    box.className='nodeframe'; box.style.left=x+'px'; box.style.top=y+'px'; box.style.width=w+'px'; box.style.height=h+'px';
    box.style.borderColor=fr.color; box.style.background=fr.color+'14';
    box.innerHTML=`<input class="frtitle" value="${(fr.title||'Group').replace(/"/g,'')}"><span class="frx" title="remove frame (keeps nodes)">✕</span>`;
    cv.appendChild(box);
    const ti=box.querySelector('.frtitle');
    ti.oninput=()=>{ fr.title=ti.value; };
    ti.addEventListener('pointerdown', e=>e.stopPropagation());
    box.querySelector('.frx').onclick=()=>{ NODE_FRAMES=NODE_FRAMES.filter(f=>f!==fr); renderNodes(); };
    box.onpointerdown=ev=>{                                   // drag the frame -> move all member nodes together
      if(ev.target!==box) return;
      const offs={}; for(const n of members) offs[n.id]=[ev.clientX-n.pos[0], ev.clientY-n.pos[1]];
      const mv=e=>{ for(const n of members){ const o=offs[n.id]; n.pos=[e.clientX-o[0], e.clientY-o[1]];
        const c=cv.querySelector(`.node[data-id="${n.id}"]`); if(c){ c.style.left=n.pos[0]+'px'; c.style.top=n.pos[1]+'px'; } }
        renderFramesOnly(); renderWires(); };
      const up=()=>{ for(const n of members) nodeApi({action:'move', id:n.id, x:n.pos[0], y:n.pos[1]});
        window.removeEventListener('pointermove',mv); window.removeEventListener('pointerup',up); };
      window.addEventListener('pointermove',mv); window.addEventListener('pointerup',up);
    };
  }
  for(const cmt of NODE_COMMENTS){                            // floating annotations (client-only, non-evaluating)
    const box=document.createElement('div');
    box.className='nodecomment'; box.style.left=cmt.x+'px'; box.style.top=cmt.y+'px';
    box.innerHTML=`<textarea rows="2" placeholder="note…">${cmt.text||''}</textarea><span class="cx" title="remove note">✕</span>`;
    cv.appendChild(box);
    const ta=box.querySelector('textarea');
    ta.oninput=()=>{ cmt.text=ta.value; };
    ta.addEventListener('pointerdown', e=>e.stopPropagation());
    box.querySelector('.cx').onclick=()=>{ NODE_COMMENTS=NODE_COMMENTS.filter(c=>c!==cmt); renderNodes(); };
    box.onpointerdown=ev=>{
      if(ev.target!==box) return;
      const ox=ev.clientX-cmt.x, oy=ev.clientY-cmt.y;
      const mv=e=>{ cmt.x=e.clientX-ox; cmt.y=e.clientY-oy; box.style.left=cmt.x+'px'; box.style.top=cmt.y+'px'; };
      const up=()=>{ window.removeEventListener('pointermove',mv); window.removeEventListener('pointerup',up); };
      window.addEventListener('pointermove',mv); window.addEventListener('pointerup',up);
    };
  }
  for(const n of (NODE.graph.nodes||[])){
    const t = NODE.types[n.type] || {inputs:{},outputs:{},drivable:{}};
    const card = document.createElement('div');
    card.className = 'node'+((NODE.sel===n.id||NODE.multi.has(n.id))?' sel':'')+(n.muted?' muted':'');
    card.dataset.id = n.id;
    card.style.left = (n.pos[0])+'px'; card.style.top = (n.pos[1])+'px';
    const ins = Object.assign({}, t.inputs, t.drivable||{});
    const isGroup = String(n.type).startsWith('subgraph');
    const title = isGroup ? `⧈ group · ${n.id}` : `${n.type} · ${n.id}`;
    let shtml = `<header><span>${title}</span>`+
      (isGroup?`<span class="expbtn" title="Expand group back into its nodes">⧉</span>`:'')+
      `<span class="mutebtn" title="Mute / bypass this node">${n.muted?'◌':'●'}</span><span class="x" title="delete">✕</span></header>`;
    for(const [s,ty] of Object.entries(ins))
      shtml += `<div class="sock"><span class="dot ${ty}" data-sock="${s}" data-out="0"></span>${s}<i style="opacity:0.5">:${ty}</i></div>`;
    for(const [s,ty] of Object.entries(t.outputs||{}))
      shtml += `<div class="sock outp">${s}<i style="opacity:0.5">:${ty}</i><span class="dot ${ty}" data-sock="${s}" data-out="1"></span></div>`;
    shtml += `<div class="params insp-host"></div>`;
    card.innerHTML = shtml;
    cv.appendChild(card);
    card.querySelector('header').onpointerdown = ev=>{
      if(ev.target.classList.contains('x')) return;
      NODE.sel = n.id;
      if(!NODE.multi.has(n.id) && !ev.shiftKey) NODE.multi.clear();
      if(ev.shiftKey) NODE.multi.add(n.id);
      // capture per-node offsets so a group drags together
      const members = NODE.multi.size ? [...NODE.multi] : [n.id];
      const offsets = {};
      for(const id of members){ const nn=NODE.graph.nodes.find(x=>x.id===id); if(nn) offsets[id]=[ev.clientX-nn.pos[0], ev.clientY-nn.pos[1]]; }
      NODE.drag = {id:n.id, ox:ev.clientX-n.pos[0], oy:ev.clientY-n.pos[1], node:n, card, members, offsets};
      renderNodes(); renderWires(); ev.preventDefault();
    };
    card.querySelector('.x').onclick = async()=>{ await nodeApi({action:'delete', id:n.id}); if(NODE.sel===n.id) NODE.sel=null; nodeRefresh(); };
    const expb = card.querySelector('.expbtn');
    if(expb) expb.onclick = async(ev)=>{ ev.stopPropagation();
      const r = await nodeApi({action:'expand', id:n.id});
      if(r && r.error) status('Expand failed: '+r.error); else status('Group expanded.');
      NODE.sel=null; NODE.multi.clear(); nodeRefresh(); };
    card.querySelector('.mutebtn').onclick = async(ev)=>{
      ev.stopPropagation();
      const r = await nodeApi({action:'mute', id:n.id, muted:!n.muted});
      status(r.muted ? 'Node muted — bypassed on build.' : 'Node un-muted.');
      nodeRefresh();
    };
    const host = card.querySelector('.insp-host');
    if(host && n.params && Object.keys(n.params).length){
      const meta = (NODE.types[n.type]||{}).meta || {};
      let setTimer=null;
      buildInspector(host, n.params, meta, (np)=>{
        n.params = np;                                       // keep local card state in sync
        clearTimeout(setTimer);
        setTimer=setTimeout(async()=>{
          const r = await nodeApi({action:'set_param', id:n.id, params:np});
          status(r.error || 'Params updated (downstream marked dirty).');
        }, 120);
      });
      // don't let scrubbing the widgets drag the node card
      host.addEventListener('pointerdown', e=>e.stopPropagation());
    }
    card.querySelectorAll('.dot').forEach(dot=>{
      dot.onpointerdown = async ev=>{
        ev.stopPropagation(); ev.preventDefault();
        const isOut = dot.dataset.out==='1', sock = dot.dataset.sock;
        if(isOut){
          const p = nodeSockPos(n.id, sock, true);
          const srcType = (NODE.types[n.type].outputs||{})[sock] || 'any';
          NODE.pendingWire = {src:n.id, src_socket:sock, srcType, x:p.x, y:p.y, to:null};
        } else if(NODE.pendingWire){
          const r = await nodeApi({action:'connect', src:NODE.pendingWire.src, src_socket:NODE.pendingWire.src_socket, dst:n.id, dst_socket:sock});
          status(r.error ? 'Refused: '+r.error : 'Connected.');
          NODE.pendingWire=null; nodeRefresh();
        }
      };
    });
  }
  renderWires();
}
$('nodecanvas') && $('nodecanvas').addEventListener('pointermove', ev=>{
  const cv = $('nodecanvas').getBoundingClientRect();
  if(NODE.drag){
    const d=NODE.drag;
    if(d.members && d.members.length>1){
      for(const id of d.members){
        const nn=NODE.graph.nodes.find(x=>x.id===id); const off=d.offsets[id]; if(!nn||!off) continue;
        nn.pos=[ev.clientX-off[0], ev.clientY-off[1]];
        const card=$('nodecanvas').querySelector(`.node[data-id="${id}"]`);
        if(card){ card.style.left=nn.pos[0]+'px'; card.style.top=nn.pos[1]+'px'; }
      }
    } else {
      d.node.pos=[ev.clientX-d.ox, ev.clientY-d.oy];
      d.card.style.left=d.node.pos[0]+'px'; d.card.style.top=d.node.pos[1]+'px';
    }
    renderWires();
  } else if(NODE.box){                                       // drawing a box-select rectangle
    NODE.box.x2=ev.clientX-cv.left; NODE.box.y2=ev.clientY-cv.top;
    drawSelectBox();
  } else if(NODE.pendingWire){ NODE.pendingWire.to={x:ev.clientX-cv.left, y:ev.clientY-cv.top}; renderWires(); }
});
window.addEventListener('pointerup', (ev)=>{
  if(NODE.drag){
    const d=NODE.drag;
    const ids = (d.members && d.members.length>1) ? d.members : [d.id];
    for(const id of ids){ const nn=NODE.graph.nodes.find(x=>x.id===id); if(nn) nodeApi({action:'move', id, x:nn.pos[0], y:nn.pos[1]}); }
    NODE.drag=null; return;
  }
  if(NODE.box){                                              // finalize: select all nodes inside the rectangle
    const b=NODE.box; const x1=Math.min(b.x1,b.x2), x2=Math.max(b.x1,b.x2), y1=Math.min(b.y1,b.y2), y2=Math.max(b.y1,b.y2);
    if(Math.abs(x2-x1)>6 || Math.abs(y2-y1)>6){
      if(!ev.shiftKey) NODE.multi.clear();
      for(const n of NODE.graph.nodes){ if(n.pos[0]>=x1-40 && n.pos[0]<=x2 && n.pos[1]>=y1-20 && n.pos[1]<=y2) NODE.multi.add(n.id); }
    }
    NODE.box=null; const bx=$('nodeselbox'); if(bx) bx.remove(); renderNodes(); renderWires(); return;
  }
  // wire dragged from an OUTPUT socket and released over empty canvas -> filtered palette (link-drag-search)
  if(NODE.pendingWire && NODE.open){
    const cv = $('nodecanvas'); const box = cv.getBoundingClientRect();
    const overEmpty = ev.target===cv || ev.target===$('nodewires');
    if(overEmpty && NODE.pendingWire.to){
      const pw = NODE.pendingWire;
      openNodePalette(ev.clientX, ev.clientY, {kinds:[pw.srcType], srcType:pw.srcType, src:pw.src, srcSocket:pw.src_socket});
    }
    NODE.pendingWire=null; renderWires();
  }
});
$('nodecanvas') && $('nodecanvas').addEventListener('pointerdown', ev=>{
  if(ev.target===$('nodecanvas') || ev.target===$('nodewires')){
    NODE.pendingWire=null; hideNodePalette();
    const cv=$('nodecanvas').getBoundingClientRect();
    NODE.box={x1:ev.clientX-cv.left, y1:ev.clientY-cv.top, x2:ev.clientX-cv.left, y2:ev.clientY-cv.top};  // begin rubber-band
    if(!ev.shiftKey){ NODE.multi.clear(); NODE.sel=null; }
    renderNodes();
  }
});
function drawSelectBox(){
  let bx=$('nodeselbox');
  if(!bx){ bx=document.createElement('div'); bx.id='nodeselbox'; $('nodecanvas').appendChild(bx); }
  const b=NODE.box; if(!b) return;
  bx.style.left=Math.min(b.x1,b.x2)+'px'; bx.style.top=Math.min(b.y1,b.y2)+'px';
  bx.style.width=Math.abs(b.x2-b.x1)+'px'; bx.style.height=Math.abs(b.y2-b.y1)+'px';
}

/* P1-4: search-to-add — double-click the canvas opens a fuzzy node palette (Browser), lands the pick at the cursor */
let NODEPAL = null;
function nodePaletteItems(filterKinds){
  return Object.values(NODE.types).filter(t=>{
    if(!filterKinds) return true;                          // link-drag: restrict to type-compatible consumers
    const ins = Object.assign({}, t.inputs, t.drivable||{});
    return Object.values(ins).some(ty=>filterKinds.includes(ty) || ty==='any');
  }).map(t=>{
    const outs = Object.values(t.outputs||{}).join(',');
    return { id:t.type, label:t.type, group:(outs||'misc'),
             keywords:`${t.type} ${outs} ${Object.values(t.inputs||{}).join(' ')}` };
  });
}
function openNodePalette(clientX, clientY, dropTarget){
  const pal = $('nodepalette'); const wrap=$('nodecanvas').getBoundingClientRect();
  const px = Math.min(clientX-wrap.left, wrap.width-270), py = Math.min(clientY-wrap.top, wrap.height-310);
  pal.style.left = Math.max(4,px)+'px'; pal.style.top = Math.max(4,py)+'px'; pal.style.display='block';
  const gx = clientX-wrap.left, gy = clientY-wrap.top;
  const items = nodePaletteItems(dropTarget && dropTarget.kinds);
  NODEPAL = buildBrowser(pal, { items, view:'list', sortKeys:['label','group'],
    onActivate: async(it)=>{
      hideNodePalette();
      const r = await nodeApi({action:'add', type:it.id, x:gx-40, y:gy-20});
      if(dropTarget && r && r.id){                          // link-drag: auto-wire from the source socket
        const dstType = NODE.types[it.id]; const ins = Object.assign({}, dstType.inputs, dstType.drivable||{});
        const slot = Object.keys(ins).find(s=> ins[s]===dropTarget.srcType || ins[s]==='any');
        if(slot) await nodeApi({action:'connect', src:dropTarget.src, src_socket:dropTarget.srcSocket, dst:r.id, dst_socket:slot});
      }
      nodeRefresh();
    }});
  const inp = pal.querySelector('.bsearch'); if(inp) setTimeout(()=>inp.focus(),30);
}
function hideNodePalette(){ const p=$('nodepalette'); if(p) p.style.display='none'; NODEPAL=null; }
$('nodecanvas') && $('nodecanvas').addEventListener('dblclick', ev=>{
  if(ev.target===$('nodecanvas') || ev.target===$('nodewires')){ ev.preventDefault(); openNodePalette(ev.clientX, ev.clientY, null); }
});
$('nodetoggle').onclick = async()=>{
  NODE.open = !NODE.open;
  $('nodeeditor').style.display = NODE.open ? 'block' : 'none';
  if(NODE.open && !Object.keys(NODE.types).length){
    const tt = await (await fetch('api/nodes/types')).json();
    for(const t of tt.types||[]) NODE.types[t.type]=t;
    $('nodetypesel').innerHTML = (tt.types||[]).map(t=>`<option value="${t.type}">${t.type}</option>`).join('');
  }
  if(NODE.open){ nodeRefresh(); refreshGroupList(); }
};
$('nodeclose').onclick = ()=>{ NODE.open=false; $('nodeeditor').style.display='none'; };
$('nodeadd').onclick = async()=>{
  const t = $('nodetypesel').value;
  await nodeApi({action:'add', type:t, x:30+Math.random()*120, y:30+Math.random()*80});
  nodeRefresh();
};
$('nodeclear').onclick = async()=>{
  const n = (NODE.graph.nodes||[]).length;
  if(n>0){
    if(!await uiConfirm(`Clear the node graph? ${n} node(s) will be removed. The graph has no undo — a backup JSON will download first.`, 'Clear graph')) return;
    try{
      const blob = new Blob([JSON.stringify(NODE.graph,null,1)], {type:'application/json'});
      const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
      a.download = 'polystudio_graph_backup.json'; a.click(); URL.revokeObjectURL(a.href);
    }catch(e){}
  }
  await nodeApi({action:'clear'}); NODE.sel=null; nodeRefresh();
};
$('nodebuild').onclick = async()=>{
  if(!NODE.sel){ status('Click a node header to select it, then Build.'); return; }
  status('Building node\u2026');
  const r = await nodeApi({action:'build', id:NODE.sel});
  if(r.error){ status('Build refused: '+r.error); return; }
  if(r.objects){ applyResp(r); status(r.analytic ? 'Node built as an EXACT analytic object.' :
                                       (r.analytic===false ? 'Node built (field-only — fitted shader).' : 'Node built.')); }
  else if(r.value!==undefined){ status('Node value: '+JSON.stringify(r.value)); }
};

/* ---- repair verbs ---- */
$('tFill').onclick=()=>ACTIVE&&runOp('fill_holes',{object:ACTIVE});
$('tBridge').onclick=()=>ACTIVE&&runOp('bridge',{object:ACTIVE});
$('tTri').onclick=()=>ACTIVE&&runOp('triangulate',{object:ACTIVE});

/* ---- CAD: boolean, lathe, curvature inspect, STL ---- */
$('boolapply').onclick=async()=>{
  const ids=[...selObjs];
  if(ids.length!==2){ status('Shift-click exactly TWO objects in Object mode: A (kept), then B (consumed).'); return; }
  status('Boolean\u2026');
  try{
    const r = await api('op', {op:'boolean', object:ids[0], other:ids[1],
                               kind:$('boolkind').value, fillet:parseFloat($('boolfillet').value)||0,
                               k:parseFloat($('boolfillet').value)||0.15,
                               chamfer:parseFloat($('boolfillet').value)||0.08});
    applyResp(r); status('Boolean done.');
  }catch(e){}
};
$('lathebtn').onclick=async()=>{
  const txt = ($('latheprof').value.trim() || $('lathepreset').value);
  const prof = txt.split(';').map(s=>s.split(',').map(Number)).filter(p=>p.length===2 && p.every(isFinite));
  if(prof.length<3){ status('Profile needs at least 3 "r,y" pairs.'); return; }
  status('Lathe\u2026');
  try{ applyResp(await api('op', {op:'lathe', profile:prof, name:'Lathe'})); status('Lathe added.'); }catch(e){}
};
let CURV = null;                                          // {oid, orig} while curvature view is on
$('curvbtn').onclick=async()=>{
  if(CURV){                                               // toggle OFF: restore material colours
    const o=OBJS.get(CURV.oid);
    if(o && o.mesh.geometry.attributes.color.array.length===CURV.orig.length){
      o.mesh.geometry.attributes.color.array.set(CURV.orig); o.mesh.geometry.attributes.color.needsUpdate=true;
    }                                                     // rebuilt object: fresh colours already in place
    CURV=null; $('curvbtn').classList.remove('on'); return;
  }
  if(!ACTIVE) return;
  try{
    const r = await api_get(`curvature?object=${ACTIVE}`);
    if(r.error){ status(r.error); return; }
    const o=OBJS.get(ACTIVE); if(!o) return;
    const attr=o.mesh.geometry.attributes.color;
    CURV={oid:ACTIVE, orig:Float32Array.from(attr.array)};
    const s=r.scale||1;
    for(let i=0;i<r.values.length;i++){
      const t=Math.max(-1, Math.min(1, r.values[i]/s));   // -1..1 -> blue-grey-red
      const R=0.55+0.45*Math.max(0,t), G=0.55-0.35*Math.abs(t), B=0.55+0.45*Math.max(0,-t);
      attr.array[i*3]=R; attr.array[i*3+1]=G; attr.array[i*3+2]=B;
    }
    attr.needsUpdate=true;
    $('curvbtn').classList.add('on');
    status(`Curvature: ${r.source} field, eps=${r.eps}.`);
  }catch(e){}
};
$('stlexp').onclick=()=>{ if(needActiveFor('.stl')) window.open(`api/export_stl?object=${ACTIVE}`, '_blank'); };

/* ---- Shader FX + exact modifiers + LOD ---- */
$('fxpreset').onchange=()=>{ $('fxexpr').value = $('fxpreset').value; };
$('fxamt').addEventListener('input', ()=>{ $('fxamtval').textContent = $('fxamt').value; });
$('fxdisplace').onclick=()=>{
  if(!ACTIVE) return;
  runOp('shader_displace', {object:ACTIVE, expr:$('fxexpr').value, amount:parseFloat($('fxamt').value)});
};
$('fxmat').onclick=()=>{
  if(!ACTIVE || !MATNAME) return;
  runOp('shader_material', {object:ACTIVE, expr:$('fxexpr').value, material:MATNAME,
                            threshold:parseFloat($('fxthresh').value)||0});
};
$('modapply').onclick=()=>{
  if(!ACTIVE) return;
  const kind=$('modkind').value, k=parseFloat($('modk').value)||1;
  const args={object:ACTIVE, kind, k};
  if(kind==='elongate') args.h=[k,0,0];
  runOp('sdf_modifier', args);
};
$('pgrid').addEventListener('input', ()=>{ $('pgridval').textContent=$('pgrid').value; scheduleRender(); });
$('photogrid').addEventListener('input', ()=>{ $('photogridval').textContent=$('photogrid').value; });
$('pradius').addEventListener('input', ()=>{ $('pradval').textContent=$('pradius').value; });
$('decbtn').onclick=()=>{
  if(!ACTIVE) return;
  runOp('decimate', {object:ACTIVE, target:parseInt($('dectarget').value)||800});
};

$('shadercopy').onclick=()=>{ $('shadercode').select(); try{document.execCommand('copy');}catch(e){} status('Shader copied.'); };
$('shaderexplain').onclick=async()=>{
  if(!ACTIVE) return;
  const r = await api_get(`explain_shader?object=${ACTIVE}`);
  const out = $('explainout'); out.style.display='block';
  if(r.explanation){
    out.innerHTML = `<b>${r.explanation.headline}</b><br>${r.explanation.full.replace(/\n/g,'<br>')}`;
  } else out.textContent = r.reason || r.error || 'No explanation available.';
};

/* ---- Add from description (leCore codecompose -> kernel + verified SDF tree + mesh) ---- */
$('composebtn').onclick=async()=>{
  const text = $('composetext').value.trim();
  if(!text){ status('Type a description first, e.g. "sphere radius .6 union rounded box size .5 .3 .4 radius .05"'); return; }
  status('Composing\u2026');
  try{
    const r = await api('compose', {text});
    if(r.error){ $('composemeta').style.display='block'; $('composemeta').textContent=r.error; status('Refused.'); return; }
    applyResp(r);
    const meta = $('composemeta'); meta.style.display='block';
    meta.innerHTML = (r.explanation ? `<b>${r.explanation.headline}</b><br>` : '') +
      (r.tree_verified
        ? 'SDF tree built from the same clauses and <b>verified against the kernel</b> (512 pts, <1e-8): bakes analytic-native, exports an EXACT shader.'
        : 'Kernel-only (tree could not be verified): meshed from the kernel; shader export uses the fitted path.');
    status('Described object added.');
  }catch(e){ status('compose failed: '+e.message); }
};

/* ---- Milkdrop motion: batched per-frame equations from the engine, played at 60fps ---- */
let MILK = {playing:false, buf:null, next:null, t0:0, fps:30, idx0:0, baseDist:null, fetching:false};
async function milkFetch(preset){
  const q = preset ? `preset=${encodeURIComponent(preset)}&n=360&fps=30` : 'n=360&fps=30';
  const r = await fetch('api/milkdrop/frames?'+q);
  return r.json();
}
function milkStop(){
  MILK.playing=false; $('milkplay').textContent='Play';
  scene.background = new THREE.Color(0x10131a);
  if(ACTIVE && OBJS.has(ACTIVE)) OBJS.get(ACTIVE).group.scale.setScalar(1);
  if(MILK.baseDist!==null){ camDist = MILK.baseDist; applyCam(); }
}
$('milkplay').onclick=async()=>{
  if(MILK.playing){ milkStop(); return; }
  const preset = $('milkpreset').value;
  if(!preset) return;
  const d = await milkFetch(preset);
  if(d.error){ status(d.error); return; }
  MILK = {playing:true, buf:d, next:null, t0:performance.now(), fps:d.fps, idx0:0, baseDist:camDist, fetching:false};
  $('milkplay').textContent='Stop';
  status(`Milkdrop: ${d.preset} \u2014 per_frame equations live.`);
};
const _milkBg = new THREE.Color();
function milkTick(){
  if(!MILK.playing || !MILK.buf) return;
  const s = MILK.buf.series;
  const tf = (performance.now()-MILK.t0)/1000*MILK.fps;   // fractional frame within this batch
  let i = Math.floor(tf) - MILK.idx0;
  if(i >= s.zoom.length-1){
    if(MILK.next){ MILK.idx0 += s.zoom.length; MILK.buf = MILK.next; MILK.next = null; return; }
    if(!MILK.fetching){ MILK.fetching=true; milkFetch(null).then(d=>{ MILK.next=d; MILK.fetching=false; }); }
    i = s.zoom.length-1;
  } else if(i > s.zoom.length*0.6 && !MILK.next && !MILK.fetching){
    MILK.fetching=true; milkFetch(null).then(d=>{ MILK.next=d; MILK.fetching=false; });
  }
  if(i<0) i=0;
  const f = Math.min(1, tf - Math.floor(tf));
  const L = (a)=> a[i] + (a[Math.min(i+1,a.length-1)]-a[i])*f;   // linear interp between server frames
  camTheta += L(s.rot);                                   // rot: radians/frame, integrated -> orbit
  camDist = MILK.baseDist / Math.max(0.5, L(s.zoom));     // zoom>1 pulls in, exactly Milkdrop's sense
  applyCam();
  _milkBg.setRGB(0.06+0.10*L(s.wave_r), 0.06+0.10*L(s.wave_g), 0.08+0.12*L(s.wave_b));
  scene.background = _milkBg;
  if(ACTIVE && OBJS.has(ACTIVE)) OBJS.get(ACTIVE).group.scale.setScalar(1 + 0.05*(L(s.bass)-1));
}
// object actions (duplicate / delete / undo) — creation lives in the Add menu
$('dupbtn').onclick=()=>ACTIVE&&runOp('duplicate',{object:ACTIVE});
$('delobjbtn').onclick=()=>ACTIVE&&runOp('delete_object',{object:ACTIVE});
$('undobtn').onclick=async()=>applyResp(await api('undo',{}));
$('redobtn') && ($('redobtn').onclick=async()=>applyResp(await api('redo',{})));   // F1
$('modeObj').onclick=()=>setMode('object');
$('modeVert').onclick=()=>setMode('vertex');
$('modeFace').onclick=()=>setMode('face');
$('modeSculpt').onclick=()=>setMode('sculpt');
$('modePaint').onclick=()=>setMode('paint');
function setTool(t){ TOOL=t;
  for(const b of ['toolMove','toolRot','toolScale']) $(b).classList.remove('on');
  $({move:'toolMove',rotate:'toolRot',scale:'toolScale'}[t]).classList.add('on');
  placeGizmo(); refreshAttrs();
}
$('toolMove').onclick=()=>setTool('move');
$('toolRot').onclick=()=>setTool('rotate');
$('toolScale').onclick=()=>setTool('scale');
$('softsel').onchange=fetchSoftWeights;
$('softrad').onchange=fetchSoftWeights;
for(const b of document.querySelectorAll('#sculpttools [data-brush]')){
  b.onclick=()=>{ BRUSH=b.dataset.brush;
    document.querySelectorAll('#sculpttools [data-brush]').forEach(x=>x.classList.remove('on'));
    b.classList.add('on'); refreshAttrs(); };
}

// (undo/redo, modes, tools, select and delete now live in the KEYMAP table -- C4)


/* ================================ C1/C2: one shared hotkey guard ================================ */
// Every keydown listener funnels through these two predicates. Each listener used to roll its own guard:
// some forgot TEXTAREA (so Backspace in a text box DELETED THE SELECTED GEOMETRY), and none checked
// modifiers (so Ctrl+C opened the camera dialog instead of copying, Ctrl+X opened cross-sections, and
// Ctrl+F / Ctrl+L / Ctrl+H / Ctrl+P were all stolen from the browser).
function isTypingTarget(e){
  const el = e && e.target; if(!el) return false;
  const t = (el.tagName||'').toUpperCase();
  return t==='INPUT' || t==='TEXTAREA' || t==='SELECT' || el.isContentEditable === true;
}
// "bare" = no modifier another owner (the browser, or a Ctrl-combo binding here) has a claim on.
// Shift is deliberately NOT checked: Shift+digit recalls a view bookmark.
function bareKey(e){ return !e.ctrlKey && !e.metaKey && !e.altKey; }




/* ================================ G6: one background-job feedback mechanism ================================
   Every long task invented its own status line: the photo wrote into #rmeta, the engine render into the
   same place, the turntable into #ttmeta, upscale into a fragment appended to whatever was there. None of
   them could be cancelled from where you could see them. One chip per running job, with progress and a
   cancel button, is what a user can actually act on.                                                      */
const JOBS = (function(){
  const host = $('jobs');
  const live = new Map(); let seq = 0;
  function draw(){
    if(!host) return;
    host.innerHTML = '';
    for(const [id, j] of live){
      const row = document.createElement('div');
      row.className = 'jobchip' + (j.done ? ' done' : '');
      row.innerHTML = `<span class="jl">${rvEsc(j.text || j.label)}</span>`+
        (j.done ? '' : `<span class="jb"><div style="width:${Math.round(j.pct)}%"></div></span>`);
      if(j.onCancel && !j.done){
        const x = document.createElement('button');
        x.className='jx'; x.textContent='\u2715';
        x.title='Cancel '+j.label; x.setAttribute('aria-label','Cancel '+j.label);
        x.onclick = ()=>{ try{ j.onCancel(); }catch(_){ } };
        row.appendChild(x);
      }
      host.appendChild(row);
    }
  }
  return {
    start(label, onCancel){
      const id = ++seq;
      live.set(id, {label, text:label, pct:0, onCancel, done:false});
      draw();
      return {
        progress(pct, text){ const j=live.get(id); if(!j) return; if(pct!=null) j.pct=pct; if(text) j.text=text; draw(); },
        done(text){ const j=live.get(id); if(!j) return; j.done=true; j.text=text||j.text; draw();
                    setTimeout(()=>{ live.delete(id); draw(); }, 2600); },
        fail(text){ this.done(text||(label+' failed')); }
      };
    }
  };
})();


/* ================================ B5: a real .zip, written here ================================
   The turntable button said "12 frames -> zip" and downloaded a JSON of data-URLs, because the code
   could not zip without a library. It does not need one: a STORE-method zip is a header, the bytes,
   and a CRC32. Deterministic, ~50 lines, no dependency.                                            */
const CRC_TABLE = (()=>{ const t=new Uint32Array(256);
  for(let n=0;n<256;n++){ let c=n; for(let k=0;k<8;k++) c = (c&1) ? (0xEDB88320 ^ (c>>>1)) : (c>>>1); t[n]=c>>>0; }
  return t; })();
function crc32(bytes){ let c=0xFFFFFFFF;
  for(let i=0;i<bytes.length;i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xFF] ^ (c>>>8);
  return (c ^ 0xFFFFFFFF)>>>0; }
function zipStore(files){                       // files: [{name, bytes:Uint8Array}]
  const enc = new TextEncoder();
  const chunks = [], central = [];
  let offset = 0;
  const u16 = v => [v & 255, (v>>>8) & 255];
  const u32 = v => [v & 255, (v>>>8) & 255, (v>>>16) & 255, (v>>>24) & 255];
  for(const f of files){
    const name = enc.encode(f.name), crc = crc32(f.bytes), n = f.bytes.length;
    const local = new Uint8Array([...u32(0x04034b50), ...u16(20), ...u16(0), ...u16(0),
                                  ...u16(0), ...u16(0), ...u32(crc), ...u32(n), ...u32(n),
                                  ...u16(name.length), ...u16(0), ...name]);
    chunks.push(local, f.bytes);
    central.push(new Uint8Array([...u32(0x02014b50), ...u16(20), ...u16(20), ...u16(0), ...u16(0),
                                 ...u16(0), ...u16(0), ...u32(crc), ...u32(n), ...u32(n),
                                 ...u16(name.length), ...u16(0), ...u16(0), ...u16(0), ...u16(0),
                                 ...u32(0), ...u32(offset), ...name]));
    offset += local.length + n;
  }
  const cdSize = central.reduce((a,c)=>a+c.length, 0);
  const end = new Uint8Array([...u32(0x06054b50), ...u16(0), ...u16(0),
                              ...u16(files.length), ...u16(files.length), ...u32(cdSize), ...u32(offset), ...u16(0)]);
  return new Blob([...chunks, ...central, end], {type:'application/zip'});
}
function b64ToBytes(b64){
  const bin = atob(b64.replace(/^data:[^,]+,/, ''));
  const out = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) out[i] = bin.charCodeAt(i);
  return out;
}

async function rvTurntable(){
  if(RV.busy){ status('A render is already running.'); return; }
  const nStr = await uiPrompt('Turntable', 'How many frames around the target?', '12');
  if(nStr===null) return;
  const N = Math.max(2, Math.min(64, parseInt(nStr,10) || 12));
  let cancelled = false;
  const job = JOBS.start('Turntable', ()=>{ cancelled = true; rvCancel(); });
  rvSetBusy(true, 'turntable');
  const th0 = camTheta, frames = [];
  try{
    for(let i=0;i<N && !cancelled;i++){
      camTheta = th0 + (i/N)*Math.PI*2; applyCam();
      RV.ctrl = new AbortController();
      const r = await fetch('api/photo?'+photoQuery(), {signal:RV.ctrl.signal});
      const txt = await r.text();
      let png = null;
      for(const line of txt.split('\n')){
        if(!line.trim()) continue;
        try{ const d = JSON.parse(line); if(d.type==='frame' && d.png) png = d.png; }catch(_){ }
      }
      if(png){
        frames.push({name:`turntable_${String(i).padStart(3,'0')}.png`, bytes:b64ToBytes(png)});
        const u = rvBlobFromB64(png);
        RV.slot='live'; if(!rvShow(u,'final')) URL.revokeObjectURL(u);
      }
      job.progress(100*(i+1)/N, `Turntable ${i+1}/${N}`);
      rvProg(100*(i+1)/N); rvMeta(`turntable frame ${i+1}/${N}`);
    }
  }catch(e){
    if(e.name!=='AbortError'){ job.fail('Turntable failed: '+e.message); rvMeta('turntable failed: '+e.message); }
  }
  camTheta = th0; applyCam(); RV.ctrl=null; rvSetBusy(false);
  if(frames.length){
    const blob = zipStore(frames);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = `polystudio_turntable_${rvStamp()}.zip`; a.click();
    setTimeout(()=>URL.revokeObjectURL(a.href), 4000);
    job.done(`Turntable: ${frames.length} frame(s) \u2192 ${a.download}`);
    rvMeta(`turntable: ${frames.length} frame(s) saved as ${a.download}`);
  }else{
    job.done('Turntable cancelled');
  }
}
$('rvturntable') && ($('rvturntable').onclick = rvTurntable);



/* ================================ D3: resizable dialogs ================================
   .dlgbody only ever inner-scrolled at a fixed width, so tall panels (Render settings, Materials,
   Objects) hid their own content with no way to make the window bigger. Same grip as the Render View. */
(function resizableDialogs(){
  document.querySelectorAll('.dlg').forEach(d=>{
    if(d.id==='dlg-renderview') return;                      // has its own, stage-aware grip
    const body = d.querySelector('.dlgbody'); if(!body) return;
    const grip = document.createElement('div');
    grip.className='rvgrip'; grip.title='Resize'; grip.setAttribute('aria-hidden','true');
    d.appendChild(grip);
    grip.addEventListener('pointerdown', ev=>{
      const r=d.getBoundingClientRect(), sx=ev.clientX, sy=ev.clientY;
      const h0=body.getBoundingClientRect().height;
      const mv=e2=>{
        d.style.width = Math.max(240, Math.min(innerWidth-16, r.width + (e2.clientX-sx)))+'px';
        body.style.maxHeight = Math.max(120, h0 + (e2.clientY-sy))+'px';
      };
      const up=()=>{
        removeEventListener('pointermove',mv); removeEventListener('pointerup',up);
        try{ const all=JSON.parse(localStorage.getItem('polystudio:dlgsize')||'{}');
             all[d.id]={w:d.style.width, h:body.style.maxHeight};
             localStorage.setItem('polystudio:dlgsize', JSON.stringify(all)); }catch(_){ }
      };
      addEventListener('pointermove',mv); addEventListener('pointerup',up); ev.preventDefault();
    });
  });
  try{
    const all=JSON.parse(localStorage.getItem('polystudio:dlgsize')||'{}');
    for(const id in all){ const d=$(id); if(!d) continue;
      const b=d.querySelector('.dlgbody'); if(!b) continue;
      if(all[id].w) d.style.width=all[id].w;
      if(all[id].h) b.style.maxHeight=all[id].h; }
  }catch(_){ }
})();


/* ================================ G7: F2 renames the active object ================================ */
// (F2 renames the active object -- KEYMAP, C4; the body is renameActive())

/* ================================ D4/F5: in-app prompt / confirm / report ================================
   Native prompt(), confirm() and alert() broke the app's visual language, could not be styled, and on
   the Engine-status path dumped a wall of text into an OS box. These are the same three shapes, drawn
   in the app, promise-based so callers read the same way.                                             */
(function askUI(){
  const host=document.createElement('div');
  host.id='uiask';
  host.innerHTML='<div class="abox" role="dialog" aria-modal="true" aria-labelledby="uiask-t">'+
    '<div class="ahead" id="uiask-t"></div><div class="abody"></div>'+
    '<div class="afoot"><button class="cancel">Cancel</button><button class="go">OK</button></div></div>';
  document.body.appendChild(host);
  const head=host.querySelector('.ahead'), body=host.querySelector('.abody'),
        ok=host.querySelector('.go'), cancel=host.querySelector('.cancel');
  let resolve=null, getVal=()=>true, prevFocus=null;
  function close(v){ host.classList.remove('open'); const r=resolve; resolve=null;
    if(prevFocus && prevFocus.focus) try{ prevFocus.focus(); }catch(_){ }
    if(r) r(v); }
  ok.onclick = ()=>close(getVal());
  cancel.onclick = ()=>close(null);
  host.addEventListener('pointerdown', e=>{ if(e.target===host) close(null); });
  host.addEventListener('keydown', e=>{
    e.stopPropagation();                                   // never let app hotkeys fire behind a modal
    if(e.key==='Escape'){ e.preventDefault(); close(null); }
    else if(e.key==='Enter' && e.target.tagName!=='TEXTAREA'){ e.preventDefault(); close(getVal()); }
  });
  function open(title, buildBody, valueFn, okLabel, showCancel){
    prevFocus=document.activeElement;
    head.textContent=title; body.innerHTML=''; buildBody(body);
    getVal=valueFn; ok.textContent=okLabel||'OK';
    cancel.style.display = showCancel===false ? 'none' : '';
    host.classList.add('open');
    const f=body.querySelector('input,textarea,select'); (f||ok).focus(); if(f&&f.select) f.select();
    return new Promise(r=>{ resolve=r; });
  }
  // prompt: resolves to the string, or null if cancelled (same contract as window.prompt)
  // G4 / WCAG 2.1.2 No Keyboard Trap works both ways: a dialog that declares aria-modal must actually
  // contain focus, or a screen-reader user is told the page behind is inert while Tab walks straight
  // into it. Wrap at both ends; Escape (handled above) is the way out.
  host.addEventListener('keydown', e=>{
    if(e.key!=='Tab') return;
    const f=[...host.querySelectorAll('button,input,select,textarea,a[href],[tabindex]:not([tabindex="-1"])')]
             .filter(el=>el.offsetParent!==null && !el.disabled);
    if(!f.length) return;
    const first=f[0], last=f[f.length-1];
    if(e.shiftKey && document.activeElement===first){ e.preventDefault(); last.focus(); }
    else if(!e.shiftKey && document.activeElement===last){ e.preventDefault(); first.focus(); }
    else if(!host.contains(document.activeElement)){ e.preventDefault(); first.focus(); }
  });
  window.uiPrompt = (title, label, def)=>open(title, b=>{
    if(label){ const p=document.createElement('p'); p.textContent=label; b.appendChild(p); }
    const i=document.createElement('input'); i.type='text'; i.value=(def==null?'':String(def));
    i.setAttribute('aria-label', label||title); b.appendChild(i);
  }, ()=>body.querySelector('input').value, 'OK');
  // confirm: resolves true / false
  window.uiConfirm = (msg, okLabel)=>open('Confirm', b=>{
    const p=document.createElement('p'); p.textContent=msg; b.appendChild(p);
  }, ()=>true, okLabel||'Continue').then(v=>v===true);
  // report: a read-only block (replaces alert() for multi-line output)
  window.uiReport = (title, text)=>open(title, b=>{
    const pre=document.createElement('pre'); pre.textContent=text; b.appendChild(pre);
  }, ()=>true, 'Close', false);
})();

/* ================================ F8/G5: touch navigation ================================
   The viewport was unusable on a tablet: orbit needed Alt+LMB / RMB / MMB and zoom needed a wheel,
   none of which a touchscreen has. Browser-3D peers treat any-device operation as table stakes.
   Scheme (the one every touch 3D app uses): one finger orbits, two fingers pinch-zoom and pan,
   a tap selects, a long press opens the context menu. Sculpt and Paint keep one-finger painting --
   dragging IS the tool there -- and orbit moves to two fingers in those modes.                    */
(function touchNav(){
  const canvas = $('view'); if(!canvas) return;
  const pts = new Map();                    // pointerId -> {x, y}
  let mode = null;                          // 'orbit' | 'pinch' | 'paint'
  let startDist = 0, startDist0 = 0, lastMid = null, moved = 0, downAt = 0, longTimer = null, downEv = null;

  const dist = (a,b)=>Math.hypot(a.x-b.x, a.y-b.y);
  const mid  = (a,b)=>({x:(a.x+b.x)/2, y:(a.y+b.y)/2});

  canvas.addEventListener('pointerdown', ev=>{
    if(ev.pointerType!=='touch') return;
    ev.preventDefault(); ev.stopImmediatePropagation();     // the mouse path must not also run
    pts.set(ev.pointerId, {x:ev.clientX, y:ev.clientY});
    try{ canvas.setPointerCapture(ev.pointerId); }catch(_){ }
    if(pts.size===1){
      moved = 0; downAt = performance.now(); downEv = ev;
      mode = (MODE==='sculpt' || MODE==='paint') ? 'paint' : 'orbit';
      if(mode==='paint'){
        if(MODE==='sculpt') beginSculptStroke(ev); else {
          paintStroke = {id:'p'+Math.random().toString(36).slice(2), painted:new Set(), pending:new Set(), timer:null};
          paintAt(ev);
        }
      }
      clearTimeout(longTimer);
      longTimer = setTimeout(()=>{                           // long press = right click
        if(pts.size===1 && moved < 10){
          mode='longpress';
            RMB_DOWN = null;                                   // a long press is a click, not a drag
          canvas.dispatchEvent(new MouseEvent('contextmenu',
            {clientX:ev.clientX, clientY:ev.clientY, bubbles:true, cancelable:true}));
        }
      }, 550);
    }else if(pts.size===2){
      clearTimeout(longTimer);
      if(mode==='paint'){ if(MODE==='sculpt') endSculptStroke(); else endPaintStroke(); }
      const [a,b]=[...pts.values()];
      startDist = startDist0 = dist(a,b) || 1; lastMid = mid(a,b); mode='pinch';
    }
  }, true);

  canvas.addEventListener('pointermove', ev=>{
    if(ev.pointerType!=='touch' || !pts.has(ev.pointerId)) return;
    ev.preventDefault(); ev.stopImmediatePropagation();
    const prev = pts.get(ev.pointerId);
    const dx = ev.clientX-prev.x, dy = ev.clientY-prev.y;
    moved += Math.hypot(dx,dy);
    pts.set(ev.pointerId, {x:ev.clientX, y:ev.clientY});
    if(pts.size===1){
      if(mode==='orbit'){
        camTheta += dx*0.008;
        camPhi = Math.min(2.9, Math.max(0.15, camPhi - dy*0.008));
        applyCam(); placeGizmo(); camMoved();
      }else if(mode==='paint'){
        if(MODE==='sculpt'){ updateBrushCursor(ev); if(sculptStroke) addStrokePoint(ev); }
        else { updateBrushCursor(ev); if(paintStroke) paintAt(ev); }
      }
    }else if(pts.size===2 && mode==='pinch'){
      const [a,b]=[...pts.values()];
      const d = dist(a,b) || 1, m = mid(a,b);
      camDist = Math.min(30, Math.max(0.8, camDist * (startDist/d)));
      startDist = d;
      const right = new THREE.Vector3().setFromMatrixColumn(camera.matrix,0);
      const up    = new THREE.Vector3().setFromMatrixColumn(camera.matrix,1);
      const mdx = m.x-lastMid.x, mdy = m.y-lastMid.y;
      camTarget.addScaledVector(right, -mdx*0.0022*camDist).addScaledVector(up, mdy*0.0022*camDist);
      lastMid = m;
      applyCam(); placeGizmo(); camMoved();
    }
  }, true);

  function release(ev){
    if(ev.pointerType!=='touch') return;
    ev.stopImmediatePropagation();
    clearTimeout(longTimer);
    const wasSingle = pts.size===1, quick = performance.now()-downAt < 400;
    pts.delete(ev.pointerId);
    if(mode==='paint' && pts.size===0){ if(MODE==='sculpt') endSculptStroke(); else endPaintStroke(); }
    if(mode==='orbit' && wasSingle && quick && moved < 10 && downEv){
      selectAtPointer(downEv, false);                        // a tap is a click
    }
    if(pts.size===0){ mode=null; downEv=null; }
    else if(pts.size===1){ mode='orbit'; moved=999; }        // lifting one finger must not fire a tap
  }
  canvas.addEventListener('pointerup', release, true);
  canvas.addEventListener('pointercancel', release, true);
  // stop the browser from scrolling/zooming the page out from under the viewport
  canvas.style.touchAction = 'none';
})();


/* ================================ G4/F9: accessibility pass ================================
   WCAG 2.2 is the standard audits reference now. Four things this app failed outright: no focus
   indicator anywhere, icon-only controls with no accessible name, dialogs with no role, and
   several functions reachable only by dragging (2.5.7 requires a non-drag path).                  */
(function a11y(){
  // 4.1.2 / 1.3.1: name and role for the dialogs and the icon-only toolstrip
  document.querySelectorAll('.dlg').forEach(d=>{
    d.setAttribute('role','dialog');
    const head = d.querySelector('.dlghead span, header span');
    if(head){
      if(!head.id) head.id = d.id+'-title';
      d.setAttribute('aria-labelledby', head.id);
    }
  });
  document.querySelectorAll('#toolstrip button, .rvbar button').forEach(b=>{
    if(b.getAttribute('aria-label')) return;
    const lb = b.querySelector('.lb');
    const name = (lb ? lb.textContent : b.title || b.textContent || '').replace(/\s+/g,' ').trim();
    if(name) b.setAttribute('aria-label', name);
  });
  // 2.5.7 Dragging Movements: arrow keys nudge the selection, so transforms are not drag-only.
  // (arrow-key nudge -- KEYMAP, C4; the body is nudgeSelection())
})();


/* ================================ G3: command palette (Ctrl+K) ================================
   The settled pattern for power-user web apps, and this app has an unfair advantage: leCore ships a
   searchable capability index, so the same box that finds "Bevel edges" also answers "how do I
   fillet" from the engine's own documentation.                                                    */
(function palette(){
  const host = document.createElement('div');
  host.id = 'cmdpal';
  host.innerHTML =
    '<div class="cpbox" role="dialog" aria-modal="true" aria-label="Command palette">'+
    '  <input id="cpinput" type="text" placeholder="Type a command\u2026  (Esc closes)" aria-label="Command" autocomplete="off">'+
    '  <div id="cplist" role="listbox"></div>'+
    '  <div class="cpfoot note">\u2191\u2193 move \u00b7 \u21b5 run \u00b7 results below the rule come from the leCore capability index</div>'+
    '</div>';
  document.body.appendChild(host);

  let items = [], filtered = [], sel = 0, docsTimer = null;
  function harvest(){
    items = [];
    document.querySelectorAll('.menu').forEach(menu=>{
      const group = (menu.querySelector('.mtop')||{}).textContent || '';
      menu.querySelectorAll('.mdrop button').forEach(b=>{
        const label = b.textContent.replace(/\s+/g,' ').trim();
        if(label) items.push({label, group, run:()=>b.click()});
      });
    });
    document.querySelectorAll('#toolstrip button').forEach(b=>{
      const lb = b.querySelector('.lb');
      const label = ((lb?lb.textContent:b.title)||'').replace(/\s+/g,' ').trim();
      if(label) items.push({label, group:'Toolbar', run:()=>b.click()});
    });
  }
  // subsequence match, the thing that makes "bvl" find "Bevel edges"
  function score(q, text){
    const t = text.toLowerCase(); q = q.toLowerCase();
    if(!q) return 1;
    if(t.startsWith(q)) return 1000;
    if(t.includes(q)) return 500 - t.indexOf(q);
    let i = 0, hits = 0;
    for(const ch of t){ if(ch===q[i]){ i++; hits++; if(i===q.length) break; } }
    return i===q.length ? 100 + hits : 0;
  }
  function draw(){
    const box = $('cplist');
    box.innerHTML = filtered.map((it,i)=>
      `<div class="cprow${i===sel?' on':''}" role="option" aria-selected="${i===sel}" data-i="${i}">`+
      `<span>${rvEsc(it.label)}</span><em>${rvEsc(it.group)}</em></div>`).join('') ||
      '<div class="cprow note">No match \u2014 try a plainer word.</div>';
    box.querySelectorAll('.cprow[data-i]').forEach(r=>{
      r.onmouseenter = ()=>{ sel = +r.dataset.i; draw(); };
      r.onclick = ()=>run(+r.dataset.i);
    });
    const on = box.querySelector('.cprow.on'); if(on) on.scrollIntoView({block:'nearest'});
  }
  function refilter(){
    const q = $('cpinput').value.trim();
    filtered = items.map(it=>({it, s:score(q, it.label+' '+it.group)}))
                    .filter(x=>x.s>0).sort((a,b)=>b.s-a.s).slice(0,40).map(x=>x.it);
    sel = 0; draw();
    clearTimeout(docsTimer);
    if(q.length>=3) docsTimer = setTimeout(()=>askEngine(q), 220);
  }
  async function askEngine(q){
    try{
      const d = await (await fetch('api/docs?q='+encodeURIComponent(q))).json();
      if(!d || !d.available || !d.results || !d.results.length) return;
      if($('cpinput').value.trim()!==q) return;                 // the query moved on
      const before = filtered.length;
      d.results.slice(0,5).forEach(r=>{
        filtered.push({label:r.name, group:'leCore \u00b7 '+(r.theme||'engine'),
                       run:()=>{ openDlg('dlg-docs'); const i=$('docsq'); if(i){ i.value=r.name; i.dispatchEvent(new Event('input')); } }});
      });
      if(filtered.length>before) draw();
    }catch(_){ }
  }
  function run(i){
    const it = filtered[i]; close();
    if(it && it.run) setTimeout(it.run, 0);
  }
  function open(){
    harvest(); host.classList.add('open');
    const inp = $('cpinput'); inp.value=''; refilter(); inp.focus();
  }
  function close(){ host.classList.remove('open'); }
  $('cpinput').addEventListener('input', refilter);
  $('cpinput').addEventListener('keydown', e=>{
    if(e.key==='ArrowDown'){ e.preventDefault(); sel=Math.min(filtered.length-1, sel+1); draw(); }
    else if(e.key==='ArrowUp'){ e.preventDefault(); sel=Math.max(0, sel-1); draw(); }
    else if(e.key==='Enter'){ e.preventDefault(); run(sel); }
    else if(e.key==='Escape'){ e.preventDefault(); close(); }
  });
  host.addEventListener('pointerdown', e=>{ if(e.target===host) close(); });
  host.addEventListener('keydown', e=>{                    // G4: aria-modal must contain focus
    if(e.key!=='Tab') return;
    e.preventDefault();                                    // the palette has one focusable: the input
    $('cpinput').focus();
  });
  // (Ctrl+K opens the palette -- KEYMAP, C4)
  window.openCommandPalette = open;
})();



/* ================================ C4: one keymap, one dispatcher ================================
   There were thirteen separate `keydown` listeners, each with its own guard, and no single place that
   knew what a key did. That is how Ctrl+C came to open the camera dialog, how Backspace in a text box
   deleted geometry, and how Alt+1 fired TWICE -- setting the display to textured *and* dropping into
   Object mode, because the mode handler only checked for Ctrl and ignored Alt.

   One table now owns every global binding. It is also the source of the shortcuts dialog, so the docs
   cannot drift from the bindings again: the dialog is generated from this array, not maintained by hand.

   Entry fields:
     combo   'Ctrl+Shift+Z' | 'Alt+1' | 'Escape' | '?'   (canonical order: Ctrl, Alt, Shift, KEY)
     combos  several combos for one action
     keys    display override for the dialog (e.g. '1 … 5')
     when    optional predicate -- an entry whose `when` is false falls through to the next match
     pass    true = do not stop after running; later matches still get a turn
     prevent true = preventDefault (default for anything with Ctrl/Alt, and for browser-owned keys)
     doc     false = do not list it in the shortcuts dialog                                        */
// An event can legitimately name its key more than one way, so match against candidates in order.
//   '?' is Shift+/ and '~' is Shift+`, so a table entry written as '?' must still match a Shift event.
//   On macOS Alt+A emits 'a\u030a' and Alt+1 emits '\u00a1', so e.key alone loses the binding entirely --
//   e.code gives the physical key back. (Alt+A / Alt+1..4 were dead on macOS before this too.)
function keyNames(e){
  const out = [];
  let k = e.key;
  if(k === ' ') k = 'Space';
  else if(k && k.length === 1) k = k.toUpperCase();
  if(k) out.push(k);
  const c = e.code || '';
  if(/^Key[A-Z]$/.test(c)) out.push(c.slice(3));
  else if(/^Digit[0-9]$/.test(c)) out.push(c.slice(5));
  else if(c === 'Backquote') out.push('`');
  else if(c === 'Slash') out.push('/');
  return [...new Set(out)];
}
function keyCombos(e){
  const mods  = (e.ctrlKey || e.metaKey ? 'Ctrl+' : '') + (e.altKey ? 'Alt+' : '');
  const shift = e.shiftKey ? 'Shift+' : '';
  const out = [];
  for(const n of keyNames(e)){
    out.push(mods + shift + n);
    if(shift) out.push(mods + n);        // the shifted-punctuation case: Shift+/ IS '?'
  }
  return [...new Set(out)];
}
function keyCombo(e){ return keyCombos(e)[0]; }   // kept for tests / debugging
const KEYMAP = [
  /* --- general --- */
  {group:'General', combo:'Ctrl+Z', label:'Undo', prevent:true,
   run: async ()=>applyResp(await api('undo',{}))},
  {group:'General', combos:['Ctrl+Shift+Z','Ctrl+Y'], keys:'Ctrl+Shift+Z', label:'Redo', prevent:true,
   run: async ()=>applyResp(await api('redo',{}))},
  {group:'General', combo:'Ctrl+K', label:'Command palette', prevent:true,
   run: ()=>openCommandPalette()},
  {group:'General', combo:'F2', label:'Rename the active object', prevent:true, run: renameActive},
  {group:'General', combo:'?', label:'This list', run: ()=>toggleDlg('dlg-shortcuts')},
  {group:'General', combo:'Escape', label:'Cancel the render in flight', doc:false,
   when: ()=>rvIsOpen() && RV.busy, prevent:true, run: ()=>rvCancel()},
  {group:'General', combo:'Escape', label:'Close the topmost dialog', pass:true, run: ()=>{
     if(typeof hideCtx === 'function') hideCtx();
     const open=[...document.querySelectorAll('.dlg.open')].filter(d=>!d.classList.contains('docked'));
     if(!open.length) return;
     open.sort((a,b)=>(+a.style.zIndex||0)-(+b.style.zIndex||0));
     closeDlg(open[open.length-1].id);
   }},

  /* --- modes and tools --- */
  {group:'Modes', combo:'1', label:'Object mode', keys:'1', run:()=>setMode('object')},
  {group:'Modes', combo:'2', label:'Vertex mode', run:()=>setMode('vertex')},
  {group:'Modes', combo:'3', label:'Face mode',   run:()=>setMode('face')},
  {group:'Modes', combo:'4', label:'Sculpt mode', run:()=>setMode('sculpt')},
  {group:'Modes', combo:'5', label:'Paint mode',  run:()=>setMode('paint')},
  {group:'Transform', combo:'W', label:'Move',   run:()=>setTool('move')},
  {group:'Transform', combo:'E', label:'Rotate', run:()=>setTool('rotate')},
  {group:'Transform', combo:'R', label:'Scale',  run:()=>setTool('scale')},
  {group:'Transform', keys:'Ctrl (hold)', label:'Snap to grid', doc:true, run:null},
  {group:'Transform', keys:'\u2190 \u2191 \u2193 \u2192', label:'Nudge selection (Shift = vertical, Alt = fine)',
   combos: ['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].flatMap(a=>
             ['', 'Alt+', 'Shift+', 'Alt+Shift+'].map(m=>m+a)),
   prevent:true, run: nudgeSelection},

  /* --- selection --- */
  {group:'Select', combo:'A', label:'All', run:()=>selAll()},
  {group:'Select', combo:'Alt+A', label:'None', prevent:true, run:()=>selNone()},
  {group:'Select', combo:'Ctrl+I', label:'Invert', prevent:true, run:()=>selInvert()},
  {group:'Select', keys:'Alt+click', label:'Edge loop', doc:true, run:null},
  {group:'Select', combos:['Delete','Backspace'], keys:'Delete', label:'Delete selection', run: deleteSelection},

  /* --- panels --- */
  {group:'Panels', combo:'N', label:'Objects & attributes', run:()=>objPanelToggleDocked()},
  {group:'Panels', combo:'M', label:'Material editor',      run:()=>toggleDlg('dlg-material')},
  {group:'Panels', combo:'P', label:'Render view (the image)', run:()=>toggleDlg('dlg-renderview')},
  {group:'Panels', combo:'Shift+P', label:'Render settings', run:()=>toggleDlg('dlg-render')},
  {group:'Panels', combo:'C', label:'Camera & depth of field', run:()=>openDlg('dlg-camera')},
  {group:'Panels', combo:'L', label:'Lighting & environment', run:()=>openDlg('dlg-lighting')},
  {group:'Panels', combo:'H', label:'History & branches', run:()=>{ openDlg('dlg-history'); histRefresh(); }},
  {group:'Panels', combo:'X', label:'Cross-sections', run:()=>openDlg('dlg-section')},

  /* --- view --- */
  {group:'View', combo:'F', label:'Frame the selection', run:()=>frameSelected()},
  {group:'View', combos:['`','~','Shift+`'], keys:'~', label:'Performance HUD', run:()=>{
     PERF.on = !PERF.on;
     const hud=$('perfhud'); if(hud) hud.style.display = PERF.on ? 'block' : 'none';
     if(PERF.on) perfRender();
   }},
  {group:'View', combo:'Alt+1', label:'Display: textured',  keys:'Alt+1 \u2026 4', prevent:true, run:()=>setDisplay('textured')},
  {group:'View', combo:'Alt+2', label:'Display: flat',      doc:false, prevent:true, run:()=>setDisplay('flat')},
  {group:'View', combo:'Alt+3', label:'Display: wireframe', doc:false, prevent:true, run:()=>setDisplay('wireframe')},
  {group:'View', combo:'Alt+4', label:'Display: vertex',    doc:false, prevent:true, run:()=>setDisplay('vertex')},
  {group:'View', combos:['Shift+1','Shift+2','Shift+3','Shift+4','Shift+5','Shift+6','Shift+7','Shift+8','Shift+9'],
   keys:'Shift+1 \u2026 9', label:'Recall a view bookmark', run:(e)=>{
     const i = +e.key - 1; if(BOOKMARKS[i]) recallBookmark(i);
   }},
  {group:'View', keys:'Alt+drag \u00b7 right-drag \u00b7 1 finger', label:'Orbit', doc:true, run:null},
  {group:'View', keys:'MMB \u00b7 Alt+right-drag \u00b7 2 fingers', label:'Pan / dolly', doc:true, run:null},
  {group:'View', keys:'wheel \u00b7 pinch', label:'Zoom', doc:true, run:null},

  /* --- render view (only while it has the surface) --- */
  {group:'Render view', combo:'Enter', label:'Render', when:()=>rvIsOpen(), prevent:true, run:()=>rvRender()},
  {group:'Render view', keys:'1 \u2026 9', label:'Recall a render from history', doc:true,
   combos:['1','2','3','4','5','6','7','8','9'],
   when:()=>rvIsOpen() && $('rvhist').classList.contains('show'),
   run:(e)=>rvLoadHist(+e.key - 1)},
  {group:'Render view', keys:'Esc', label:'Cancel the render', doc:true, run:null},

  /* --- node editor --- */
  {group:'Node editor', combo:'Ctrl+G', label:'Group the selected nodes', when:()=>NODE.open,
   prevent:true, run: groupSelectedNodes},
];

/* --- dispatch ------------------------------------------------------------------------------- */
const KEYMAP_INDEX = (()=>{
  const m = new Map();
  for(const entry of KEYMAP){
    if(!entry.run) continue;                                  // documentation-only rows
    for(const c of (entry.combos || (entry.combo ? [entry.combo] : []))){
      if(!m.has(c)) m.set(c, []);
      m.get(c).push(entry);
    }
  }
  // SPECIFICITY: an entry with a `when` guard beats an unconditional one on the same key, whatever
  // order they appear in the table. Without this, '1' switched to Object mode even with the render
  // view's history strip open, so "press 1-9 to recall a render" was documented but unreachable.
  for(const list of m.values()){
    list.sort((a,b)=>(b.when?1:0)-(a.when?1:0));              // stable: ties keep table order
  }
  return m;
})();
addEventListener('keydown', async e=>{
  if(isTypingTarget(e)) return;                               // C1: one guard, applied once
  for(const combo of keyCombos(e)){
    const list = KEYMAP_INDEX.get(combo);
    if(!list) continue;
    let handled = false;
    for(const entry of list){
      if(entry.when && !entry.when()) continue;
      if(entry.prevent) e.preventDefault();
      try{ await entry.run(e); }catch(err){ status('Shortcut failed: '+err.message); }
      handled = true;
      if(!entry.pass) break;
    }
    if(handled) return;                                       // first combo that actually ran wins
  }
});

/* --- the actions that used to live inline in a listener ------------------------------------- */
async function deleteSelection(e){
  if(NODE.open && (NODE.multi.size || NODE.sel)){             // node editor owns Delete while it is open
    e.preventDefault();
    const ids = NODE.multi.size ? [...NODE.multi] : [NODE.sel];
    for(const id of ids) await nodeApi({action:'delete', id});
    NODE.multi.clear(); NODE.sel = null; nodeRefresh(); return;
  }
  if(MODE==='face' && selFaces.size) runOp('delete_faces', {object:ACTIVE, faces:[...selFaces]});
  else if(MODE==='object' && selObjs.size)
    for(const id of [...selObjs]) await runOp('delete_object', {object:id});
}
// WCAG 2.2 2.5.7: a non-drag path for transforms. Also the fastest way to place something precisely.
async function nudgeSelection(e){
  const map = {ArrowLeft:[-1,0,0], ArrowRight:[1,0,0], ArrowUp:[0,0,-1], ArrowDown:[0,0,1]};
  let d = map[e.key];
  if(!d || !ACTIVE || MODE!=='object' || !selObjs.size) return;
  if(e.shiftKey) d = [0, d[0] || (-d[2]), 0];                 // Shift = vertical
  const step = ($('snapGrid') && $('snapGrid').checked) ? (parseFloat($('snapStep').value)||0.25)
             : (e.altKey ? 0.01 : 0.1);                       // Alt = fine
  try{
    for(const id of [...selObjs]){
      const r = await api('op', {op:'translate', object:id, delta:[d[0]*step, d[1]*step, d[2]*step]});
      if(r && !r.error) applyResp(r);
    }
    placeGizmo();
    status(`Nudged ${selObjs.size} object(s) by ${step}`);
  }catch(err){ status('Nudge failed: '+err.message); }
}
async function renameActive(){
  if(!ACTIVE){ status('Select an object to rename.'); return; }
  const cur = (OBJS.get(ACTIVE)||{d:{}}).d.name || '';
  const name = await uiPrompt('Rename object', 'New name:', cur);
  if(name===null || !name.trim()) return;
  try{
    const r = await api('op', {op:'rename', object:ACTIVE, name:name.trim()});
    if(r && r.error){ status('Rename failed: '+r.error); return; }
    applyResp(r); refreshObjList(); refreshAttrs(); status('Renamed to "'+name.trim()+'".');
  }catch(err){ status('Rename failed: '+err.message); }
}
async function groupSelectedNodes(){
  const ids = NODE.multi.size ? [...NODE.multi] : [];
  if(ids.length < 2){ status('Select at least two nodes (drag a box or shift-click), then Ctrl+G to group.'); return; }
  const r = await nodeApi({action:'collapse', ids});
  if(r && r.error){ status('Group failed: '+r.error); return; }
  NODE.multi.clear(); NODE.sel = r.group || null;
  status('Collapsed '+ids.length+' nodes into a group — the graph computes exactly as before. ⧉ on the group expands it.');
  nodeRefresh();
}

/* --- the shortcuts dialog, generated from the table ----------------------------------------- */
function buildShortcutsDialog(){
  const body = document.querySelector('#dlg-shortcuts .dlgbody');
  if(!body) return;
  const groups = [];
  for(const e of KEYMAP){
    if(e.doc === false) continue;
    let g = groups.find(x=>x.name===e.group);
    if(!g){ g = {name:e.group, rows:[]}; groups.push(g); }
    const keys = e.keys || e.combo || (e.combos||[])[0] || '';
    g.rows.push({label:e.label, keys});
  }
  body.innerHTML = groups.map(g=>
    `<div class="mglabel" style="padding-left:0">${rvEsc(g.name)}</div>` +
    g.rows.map(r=>{
      const kb = /[a-z]/.test(r.keys) && r.keys.includes(' ') && !/\+/.test(r.keys)
        ? rvEsc(r.keys)                                        // prose ("wheel · pinch") stays prose
        : r.keys.split(' ').map(k=>`<kbd>${rvEsc(k)}</kbd>`).join(' ');
      return `<div class="scrow"><span>${rvEsc(r.label)}</span>${kb}</div>`;
    }).join('')
  ).join('');
}
buildShortcutsDialog();


/* ================================ boot ================================ */
function resize(){
  const w=$('viewwrap').clientWidth, h=$('viewwrap').clientHeight;
  renderer.setSize(w,h,false); camera.aspect=w/h; camera.updateProjectionMatrix();
}
addEventListener('resize', resize);
(function loop(){ requestAnimationFrame(loop);
  try{ milkTick(); }catch(e){}
  try{ animTick(); }catch(e){}
  try{ renderViews(); }catch(e){}
  try{ perfTick(); }catch(e){}
  try{ updateDimTag(); }catch(e){}
})();
(async function boot(){
  resize(); applyCam();
  status('Connecting to the engine\u2026');
  MATS = await api('materials');
  MATCLASS = 'metal' in MATS.classes ? 'metal' : Object.keys(MATS.classes)[0];
  MATNAME = MATS.default;
  buildMatTabs(); buildSwatches(); buildMatbar();
  try{
    const mp = await (await fetch('api/milkdrop/presets')).json();
    $('milkpreset').innerHTML = (mp.presets||[]).map(p=>`<option value="${p.name}">${p.title}</option>`).join('');
  }catch(e){}
  applyResp(await api('scene'));
  // first-run welcome (in-session flag; shows once per load)
  // F2: shown once, not on every reload. The matbar already persisted its state this way.
  try{ if($('welcome') && localStorage.getItem('polystudio:welcome')!=='seen') $('welcome').style.display='block'; }
  catch(_){ if($('welcome')) $('welcome').style.display='block'; }
  updateHint();
})();

// A persistent one-line hint showing the current mode's key action, so the viewport is never a mystery.
function updateHint(){
  const el = $('hint'); if(!el) return;
  const byMode = {
    object: 'Object mode — click to select · W/E/R to move/rotate/scale · Add ▸ to create',
    vertex: 'Vertex mode — drag to box-select points · move with the gizmo',
    face:   'Face mode — click faces · Mesh ▸ Extrude / Inset / Bevel',
    sculpt: 'Sculpt mode — drag on the surface to sculpt · pick a brush at left',
    paint:  'Paint mode — pick a material, then paint faces',
  };
  el.textContent = byMode[MODE] || '';
}

/* ================================ pro UI: menus, dialogs, toolbar ================================ */
// Menu bar: click a top item to open its dropdown; click-away or Esc closes. Menu items that carry data-dlg
// open a floating dialog instead of firing an action.
(function(){
  const menus = [...document.querySelectorAll('#menubar .menu')];
  function closeMenus(){ menus.forEach(m=>m.classList.remove('open')); }
  menus.forEach(m=>{
    const top = m.querySelector('.mtop');
    top.addEventListener('click', e=>{
      e.stopPropagation();
      const wasOpen = m.classList.contains('open');
      closeMenus();
      if(!wasOpen){ m.classList.add('open'); refreshMenuDisabled(); }
    });
    top.addEventListener('mouseenter', ()=>{ if(menus.some(x=>x.classList.contains('open'))){ closeMenus(); m.classList.add('open'); refreshMenuDisabled(); } });
  });
  document.addEventListener('click', closeMenus);
  // any button inside a dropdown closes the menu after it runs
  document.querySelectorAll('#menubar .mdrop button').forEach(b=>b.addEventListener('click', ()=>setTimeout(closeMenus,0)));
})();

// Floating dialogs: open/close/drag. data-dlg buttons anywhere open the matching dialog.
function openDlg(id){
  if(id==='dlg-scatter') scatterFillSelects();
  const d = $(id); if(!d) return;
  const wasOpen = d.classList.contains('open');
  d.classList.add('open');
  d.style.zIndex = String(90 + (openDlg._z = (openDlg._z||0)+1));
  // D1: don't stack every dialog in the same corner -- cascade each newly opened one.
  if(!wasOpen) cascadeDlg(d);
  if(id==='dlg-renderview'){ rvLayout(); scheduleRender(); }   // B1: preview starts when it can be SEEN
}
function closeDlg(id){
  const d=$(id); if(d) d.classList.remove('open');
  // B1: closing the render view stops all background render traffic immediately.
  if(id==='dlg-renderview'){
    PROG.token++;
    if(PROG.ctrl){ try{ PROG.ctrl.abort(); }catch(_){ } PROG.ctrl=null; }
    if(RV.busy) rvCancel();
  }
}
/* D1/D2: cascade + clamp. Every dialog carried the same `right:24px; top:56px`, so Render, Camera,
   Lighting and Shortcuts all landed on top of one another; and a dragged dialog could be pushed off
   the right/bottom edge with no way back. */
function cascadeDlg(d){
  if(d.classList.contains('docked')) return;
  // Leave deliberately-centred dialogs alone: they position with left:50% + translateX(-50%), and
  // cascading them read '50%' as 50px while the transform still pulled them half a width further left,
  // which put them off the left edge. Centred is already a fine place to be.
  const centred = (d.style.left||'').indexOf('%') >= 0 || getComputedStyle(d).transform !== 'none';
  if(centred){ d.dataset.placed='1'; return; }
  if(d.dataset.placed==='1'){ clampDlg(d); return; }
  const n = (cascadeDlg._n = (cascadeDlg._n||0)+1);
  const step = 26, ring = ((n-1)%6);
  const r = d.getBoundingClientRect();
  if(d.style.right && !d.style.left){
    d.style.left = Math.max(8, innerWidth - r.width - 24 - ring*step)+'px';
    d.style.right = 'auto';
  }else if(!d.style.left){
    d.style.left = (96 + ring*step)+'px';
  }else{
    d.style.left = (parseFloat(d.style.left) + ring*step)+'px';
  }
  d.style.top = (Math.max(44, parseFloat(d.style.top||'60')) + ring*step)+'px';
  d.dataset.placed='1';
  clampDlg(d);
}
function clampDlg(d){
  if(d.dataset.centred==='1' || (d.style.left||'').indexOf('%')>=0 || getComputedStyle(d).transform!=='none') return;
  const r = d.getBoundingClientRect();
  const maxL = Math.max(8, innerWidth  - Math.min(r.width, innerWidth-16) - 8);
  const maxT = Math.max(40, innerHeight - 60);
  d.style.left = Math.min(Math.max(0, parseFloat(d.style.left||r.left)), maxL)+'px';
  d.style.top  = Math.min(Math.max(38, parseFloat(d.style.top||r.top)),  maxT)+'px';
}
addEventListener('resize', ()=>document.querySelectorAll('.dlg.open').forEach(clampDlg));
function toggleDlg(id){ const d=$(id); if(d) d.classList.contains('open') ? closeDlg(id) : openDlg(id); }
document.querySelectorAll('[data-dlg]').forEach(b=>b.addEventListener('click', ()=>openDlg(b.dataset.dlg)));
document.querySelectorAll('.dlg .dlghead .x').forEach(x=>{
  const shut=()=>closeDlg(x.closest('.dlg').id);            // via closeDlg so B1's stop-work runs
  x.addEventListener('click', shut);
  x.setAttribute('role','button'); x.setAttribute('tabindex','0');
  if(!x.getAttribute('aria-label')) x.setAttribute('aria-label','Close dialog');
  x.addEventListener('keydown', e=>{ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); shut(); } });
});
// drag dialogs by their header
document.querySelectorAll('.dlg .dlghead').forEach(head=>{
  head.addEventListener('pointerdown', ev=>{
    if(ev.target.classList.contains('x')) return;
    const dlg = head.closest('.dlg'); const r = dlg.getBoundingClientRect();
    dlg.style.left = r.left+'px'; dlg.style.top = r.top+'px'; dlg.style.right='auto';
    const ox = ev.clientX - r.left, oy = ev.clientY - r.top;
    dlg.style.zIndex = String(90 + (openDlg._z = (openDlg._z||0)+1));
    function mv(e){
      const w=dlg.getBoundingClientRect().width;
      dlg.style.left=Math.min(Math.max(0,e.clientX-ox), Math.max(0,innerWidth-Math.min(w,innerWidth-16)-8))+'px';
      dlg.style.top =Math.min(Math.max(36,e.clientY-oy), innerHeight-60)+'px';
      dlg.dataset.placed='1';                                  // D2: keep the user's placement
    }
    function up(){ removeEventListener('pointermove',mv); removeEventListener('pointerup',up); }
    addEventListener('pointermove',mv); addEventListener('pointerup',up); ev.preventDefault();
  });
});

// Toolbar expand toggle (icons <-> icons+labels)
$('stripexpand').onclick = ()=>{
  const strip = $('toolstrip');
  strip.classList.toggle('expanded');
  const expanded = strip.classList.contains('expanded');
  const w = expanded ? 172 : 58;
  $('viewwrap').style.left = w+'px';
  // label + tooltip reflect the action the button now performs
  const btn = $('stripexpand');
  btn.title = expanded ? 'Collapse toolbar' : 'Expand toolbar';
  const lb = btn.querySelector('.lb'); if(lb) lb.textContent = 'Collapse';   // only visible when expanded
  resize();
};

// Toolbar buttons that open dialogs (data-dlg already handled above); mode/tool buttons wired elsewhere.

/* ---- Add menu: primitives + (compose/lathe handled by their dialogs) ---- */
document.querySelectorAll('#menubar [data-prim]').forEach(b=>{
  b.addEventListener('click', async ()=>{ applyResp(await api('new',{primitive:b.dataset.prim})); status(b.dataset.prim+' added.'); });
});

/* ---- Select All / None / Invert: shared functions used by menu + keys ---- */
function selAll(){
  const o=activeObj(); if(!o) return;
  if(MODE==='object') for(const id of OBJS.keys()) selObjs.add(id);
  else if(MODE==='vertex') for(let i=0;i<o.d.counts.v;i++) selVerts.add(i);
  else if(MODE==='face') for(let i=0;i<o.d.counts.f;i++) selFaces.add(i);
  refreshSelectionVisuals(); refreshAttrs();
}
function selNone(){ selObjs.clear(); selVerts.clear(); selFaces.clear(); softW=null; refreshSelectionVisuals(); refreshObjList(); refreshAttrs(); }
function selInvert(){
  const o=activeObj(); if(!o) return;
  if(MODE==='vertex'){ const s=new Set(); for(let i=0;i<o.d.counts.v;i++) if(!selVerts.has(i)) s.add(i); selVerts=s; }
  else if(MODE==='face'){ const s=new Set(); for(let i=0;i<o.d.counts.f;i++) if(!selFaces.has(i)) s.add(i); selFaces=s; }
  else if(MODE==='object'){ const s=new Set(); for(const id of OBJS.keys()) if(!selObjs.has(id)) s.add(id); selObjs=s; }
  refreshSelectionVisuals(); refreshAttrs();
}
$('selAllBtn').onclick=selAll; $('selNoneBtn').onclick=selNone; $('selInvBtn').onclick=selInvert;

/* ---- OBJ import / export ---- */
$('objexp').onclick=()=>{ if(needActiveFor('.obj')) window.open(`api/export_obj?object=${ACTIVE}`,'_blank'); };
$('objbtn').onclick=()=>$('objfile').click();
$('objfile').onchange=async()=>{
  const f=$('objfile').files[0]; if(!f) return;
  status('Importing '+f.name+'\u2026');
  try{
    const buf=await f.arrayBuffer();
    const r=await fetch('api/import_obj?name='+encodeURIComponent(f.name.replace(/\.obj$/i,'')),{method:'POST',body:buf});
    const d=await r.json();
    if(d.error){ status('Import failed: '+d.error); } else { applyResp(d); status('Imported '+f.name+' (centered at origin).'); }
  }catch(e){ status('Import failed: '+e.message); }
  $('objfile').value='';
};

/* ---- Material editor: live value labels + save ---- */
(function(){
  const pairs=[['cm_metal','cm_metalv',2],['cm_rough','cm_roughv',2],['cm_trans','cm_transv',2],['cm_ior','cm_iorv',2],['cm_emit','cm_emitv',1]];
  for(const [sl,lb,dp] of pairs){ const s=$(sl); if(s) s.addEventListener('input',()=>{ $(lb).textContent=parseFloat(s.value).toFixed(dp); }); }
})();
function hexToRgb(h){ const n=parseInt(h.slice(1),16); return [(n>>16&255)/255,(n>>8&255)/255,(n&255)/255]; }
$('cm_save').onclick=async()=>{
  const name=$('cm_name').value.trim();
  if(!name){ status('Give the material a name first.'); return; }
  status('Saving material\u2026');
  try{
    const body={ name, color:hexToRgb($('cm_color').value),
      metallic:parseFloat($('cm_metal').value), roughness:parseFloat($('cm_rough').value),
      transmission:parseFloat($('cm_trans').value), ior:parseFloat($('cm_ior').value), emission:parseFloat($('cm_emit').value) };
    const r=await fetch('api/material/custom',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){ status('Refused: '+d.error); return; }
    MATS = await api('materials');                     // refresh library (adds/updates the custom tab)
    MATCLASS='custom'; MATNAME=d.name; buildMatTabs(); buildSwatches(); buildMatbar();
    $('cm_ball').src = `api/material_ball?name=${d.name}&res=96&t=${Date.now()}`;
    status('Saved "'+d.name+'" — now selectable, paintable, path-traced.');
  }catch(e){ status('Save failed: '+e.message); }
};

/* ---- global hotkeys for the three primary dialogs (ignore when typing) ---- */
// (panel hotkeys now live in the KEYMAP table -- C4)

/* ================================ P0-1: engine reference (ask leCore) ================================ */
(function(){
  let docsTimer = null, docsLoaded = false;
  function renderDocs(d){
    const box = $('docsresults');
    if(!d || !d.available){ box.innerHTML = '<div class="note">Engine index not found in this build.</div>'; return; }
    if(!d.results.length){ box.innerHTML = '<div class="note">No matches. Try a plainer word (e.g. "fillet", "noise", "constraint").</div>'; return; }
    box.innerHTML = d.results.map(r=>{
      const al = (r.aliases&&r.aliases.length) ? `<div class="dalias">also: ${r.aliases.slice(0,5).join(' · ')}</div>` : '';
      const ex = r.example ? `<div class="dex">${esc(r.example)}</div>` : '';
      return `<div class="doccard"><div class="dname"><span>${esc(r.name)}</span><span class="dtheme">${esc(r.theme||'')}</span></div>`+
             `<div class="ddoes">${esc(r.does||'')}</div>${al}${ex}</div>`;
    }).join('');
  }
  function esc(s){ return String(s==null?'':s).replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m])); }
  async function search(q){
    try{
      const r = await fetch('api/docs?q='+encodeURIComponent(q||''));
      const d = await r.json();
      if(d.themes && !$('docsthemes').textContent) $('docsthemes').textContent = d.count+' capabilities · themes: '+(d.themes||[]).join(', ');
      renderDocs(d);
    }catch(e){ $('docsresults').innerHTML = '<div class="note">Search failed: '+e.message+'</div>'; }
  }
  const inp = $('docsq');
  if(inp){
    inp.addEventListener('input', ()=>{ clearTimeout(docsTimer); docsTimer=setTimeout(()=>search(inp.value), 160); });
  }
  // populate on first open of the dialog
  const openBtns = document.querySelectorAll('[data-dlg="dlg-docs"]');
  openBtns.forEach(b=>b.addEventListener('click', ()=>{ if(!docsLoaded){ docsLoaded=true; search(''); } setTimeout(()=>inp&&inp.focus(),50); }));
})();

/* ================================ P0-2: material quick-strip ================================ */
function buildMatbar(){
  const bar = $('matbar'); if(!bar || !MATS) return;
  const cls = MATS.classes[MATCLASS] || [];
  // show up to ~10 from the current class, current material first
  const list = [...cls].sort((a,b)=> (a.name===MATNAME?-1:0) - (b.name===MATNAME?-1:0)).slice(0,10);
  const sw = m => {
    // instant DISTINCT swatch from the material's own albedo (the payload carries it) — a shaded-ball look
    // via a radial gradient, so ten metals no longer render as ten identical dark thumbnails
    const [r,g,b] = (m.albedo||[0.7,0.7,0.7]).map(c=>Math.round(Math.min(1,Math.max(0,c))*255));
    const hi = `rgb(${Math.min(255,r+70)},${Math.min(255,g+70)},${Math.min(255,b+70)})`;
    const lo = `rgb(${Math.round(r*0.35)},${Math.round(g*0.35)},${Math.round(b*0.35)})`;
    return `background:radial-gradient(circle at 34% 30%, ${hi}, rgb(${r},${g},${b}) 55%, ${lo})`;
  };
  bar.innerHTML = `<span class="qlabel">${MATCLASS}</span>` + list.map(m=>
    `<div class="qsw${m.name===MATNAME?' on':''}" data-mat="${encodeURIComponent(m.name)}" title="${m.name}" style="${sw(m)}"></div>`).join('')
    + `<span class="qhide" id="matbarhide" title="Hide this bar (the Materials panel — M — is the full browser)">✕</span>`;
  const hideBtn = bar.querySelector('#matbarhide');
  if(hideBtn) hideBtn.onclick = (e)=>{ e.stopPropagation(); document.body.classList.add('matbar-hidden');
    try{ localStorage.setItem('polystudio:matbar','hidden'); }catch(_){}} ;
  bar.querySelectorAll('.qsw').forEach(el=>{
    const name = decodeURIComponent(el.dataset.mat);
    el.onclick = ()=>{ MATNAME=name; buildMatbar(); if(MATBROWSER) MATBROWSER.setSelected(name); refreshAttrs(); status('Material: '+name); };
    el.ondblclick = ()=>{ MATNAME=name; applyMaterial(MODE==='face' && selFaces.size ? 'sel' : 'all'); buildMatbar(); };
  });
}

/* ================================ P0-3: context-menu framework ================================ */
// One reusable menu. items: [{label, kbd?, act, disabled?} | {sep:true} | {head:'…'}]
function ctxMenu(items, x, y){
  const m = $('ctxmenu');
  m.innerHTML = items.map(it=>{
    if(it.sep) return '<div class="sep"></div>';
    if(it.head) return `<div class="chead">${it.head}</div>`;
    return `<div class="ci${it.disabled?' disabled':''}" data-k="${it._k}">${it.label}${it.kbd?`<kbd>${it.kbd}</kbd>`:''}</div>`;
  }).join('');
  // wire clicks by index
  const live = items.filter(it=>!it.sep && !it.head);
  m.querySelectorAll('.ci').forEach((el,i)=>{
    const it = live[i];
    if(it && !it.disabled) el.onclick = ()=>{ hideCtx(); it.act(); };
  });
  m.style.display='block';
  // keep on-screen
  const w=m.offsetWidth, h=m.offsetHeight, vw=innerWidth, vh=innerHeight;
  m.style.left = Math.min(x, vw-w-6)+'px';
  m.style.top  = Math.min(y, vh-h-6)+'px';
}
function hideCtx(){ $('ctxmenu').style.display='none'; }
addEventListener('click', e=>{ if(!$('ctxmenu').contains(e.target)) hideCtx(); });
// (Escape hides the context menu -- KEYMAP, C4)
addEventListener('blur', hideCtx);

// viewport right-click → mode-aware menu
canvas.addEventListener('contextmenu', ev=>{
  ev.preventDefault();
  // if the right button was DRAGGED (orbit), do not open the menu — a context menu is a click, not a drag
  if(RMB_DOWN && Math.hypot(ev.clientX-RMB_DOWN.x, ev.clientY-RMB_DOWN.y) > 5){ RMB_DOWN=null; return; }
  RMB_DOWN=null;
  const hit = pickScene(ev);
  if(hit && MODE==='object' && !selObjs.has(hit.id)){ /* right-click selects under cursor first */
    selObjs = new Set([hit.id]); ACTIVE = hit.id; refreshSelectionVisuals(); refreshObjList(); refreshAttrs();
  } else if(hit && ACTIVE!==hit.id){                 // other modes: focus the clicked object so
    ACTIVE = hit.id; refreshObjList(); refreshAttrs(); // "Object properties…" shows THIS object
  }
  const items = [];
  const haveObj = !!ACTIVE, haveFaces = selFaces.size>0, haveVert = selVerts.size>0;
  if(MODE==='object'){
    items.push({head:'Object'});
    items.push({label:`Apply material${MATNAME?' ('+MATNAME+')':''}`, act:()=>applyMaterial('all'), disabled:!haveObj||!MATNAME});
    items.push({label:'Duplicate', kbd:'', act:()=>ACTIVE&&runOp('duplicate',{object:ACTIVE}), disabled:!haveObj});
    items.push({label:'Delete', kbd:'Del', act:()=>ACTIVE&&runOp('delete_object',{object:ACTIVE}), disabled:!haveObj});
    items.push({sep:true});
    items.push({label:'Frame selected', kbd:'F', act:()=>frameSelected(), disabled:!haveObj});
    items.push({label:'Object properties…', kbd:'N', act:()=>openObjProps(), disabled:!haveObj});
  } else if(MODE==='face'){
    items.push({head:'Polygons'+(haveFaces?` (${selFaces.size})`:'')});
    items.push({label:'Extrude', act:()=>$('tExtrude').onclick&&$('tExtrude').onclick(), disabled:!haveFaces});
    items.push({label:'Inset', act:()=>$('tInset').onclick&&$('tInset').onclick(), disabled:!haveFaces});
    items.push({label:'Poke (fan)', act:()=>$('tPoke').onclick&&$('tPoke').onclick(), disabled:!haveFaces});
    items.push({sep:true});
    items.push({label:`Apply material${MATNAME?' ('+MATNAME+')':''}`, act:()=>applyMaterial('sel'), disabled:!haveFaces||!MATNAME});
    items.push({label:'Delete faces', kbd:'Del', act:()=>runOp('delete_faces',{object:ACTIVE,faces:[...selFaces]}), disabled:!haveFaces});
    items.push({sep:true});
    items.push({label:'Fill holes', act:()=>$('tFill').onclick&&$('tFill').onclick(), disabled:!haveObj});
    items.push({label:'Triangulate n-gons', act:()=>$('tTri').onclick&&$('tTri').onclick(), disabled:!haveObj});
  } else if(MODE==='vertex'){
    items.push({head:'Points'+(haveVert?` (${selVerts.size})`:'')});
    items.push({label:'Bevel vertex', act:()=>$('tBevel').onclick&&$('tBevel').onclick(), disabled:!haveVert});
    items.push({label:'Dissolve vertex', act:()=>$('tDissolve').onclick&&$('tDissolve').onclick(), disabled:!haveVert});
    items.push({sep:true});
    items.push({label:'Select all', kbd:'A', act:selAll});
    items.push({label:'Select none', kbd:'Alt+A', act:selNone});
  } else {
    items.push({head:MODE});
    items.push({label:'Object properties…', kbd:'N', act:()=>openObjProps(), disabled:!haveObj});
  }
  ctxMenu(items, ev.clientX, ev.clientY);
});

// Frame-selected: fit the spherical camera to the active/selected objects' bounds
function frameSelected(){
  const ids = selObjs.size ? [...selObjs] : (ACTIVE?[ACTIVE]:[]);
  if(!ids.length) return;
  const box = new THREE.Box3();
  for(const id of ids){ const o=OBJS.get(id); if(o) box.expandByObject(o.mesh); }
  if(box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
  const radius = Math.max(sz.x, sz.y, sz.z) * 0.5 + 0.3;
  // distance so the sphere fits the vertical FOV with a little margin
  const fov = camera.fov * Math.PI/180;
  camTarget.copy(c);
  camDist = Math.max(1.2, (radius / Math.sin(fov/2)) * 1.15);
  applyCam(); placeGizmo(); scheduleRender();
}
// (F frames the selection -- KEYMAP, C4)

/* ================================ P0-4: persistent notifications (WARN/errors) ================================ */
// Transient info stays in the status line; things the user must SEE (thin-mesh WARN, errors) become a
// dismissible banner that doesn't get overwritten by the next status message.
function notify(msg, kind){
  const box = $('notify'); if(!box || !msg) return;
  // de-dup: same message already shown -> skip
  if([...box.children].some(c=>c.dataset.msg===msg)) return;
  const row = document.createElement('div');
  row.className = 'nrow'+(kind==='err'?' err':'');
  row.dataset.msg = msg;
  row.innerHTML = `<span>${msg.replace(/[&<>]/g,m=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[m]))}</span><span class="nx">✕</span>`;
  row.querySelector('.nx').onclick = ()=>row.remove();
  box.appendChild(row);
}
function clearNotify(pred){
  const box=$('notify'); if(!box) return;
  [...box.children].forEach(c=>{ if(!pred || pred(c.dataset.msg)) c.remove(); });
}
// pull any "WARN …" out of a render header string into the banner
function surfaceRenderWarn(header){
  clearNotify(m=>m.startsWith('Thin geometry'));
  if(header && header.indexOf('WARN')>=0){
    const w = header.slice(header.indexOf('WARN')+4).split('; bake')[0].trim();
    if(w) notify('Thin geometry: '+w);
  }
}

/* ================================ P0-4 (A-7): disabled menu states ================================ */
// Gray out any [data-needs] item whose precondition isn't currently met, evaluated when its menu opens.
function refreshMenuDisabled(){
  const have = {
    object: !!ACTIVE,
    face:   (MODE==='face' && selFaces.size>0),
    vertex: (MODE==='vertex' && selVerts.size>0),
  };
  document.querySelectorAll('#menubar [data-needs]').forEach(b=>{
    const need = b.dataset.needs;
    const ok = have[need] !== undefined ? have[need] : true;
    b.disabled = !ok;
    b.style.opacity = ok ? '' : '0.4';
    b.title = ok ? '' : ({object:'Select an object first', face:'Select faces in Face mode first',
                          vertex:'Select vertices in Vertex mode first'}[need] || '');
  });
}

/* ================================ P0.5-2: reusable typed Inspector ================================ */
// buildInspector(container, params{}, meta{}, onCommit(newParams)) -> renders a typed widget per param.
// float→scrubber-slider (drag the number to scrub too), int→spinner, enum→dropdown, bool→toggle,
// vec→N linked scrubbers, string→validated text. Declared {min,max,step,int,enum} shape+clamp the widget.
function buildInspector(container, params, meta, onCommit){
  meta = meta || {};
  const state = JSON.parse(JSON.stringify(params||{}));
  container.classList.add('insp'); container.innerHTML='';
  const commit = ()=>{ try{ onCommit(JSON.parse(JSON.stringify(state))); }catch(e){} };
  for(const key of Object.keys(state)){
    const m = meta[key] || {};
    const val = state[key];
    const row = document.createElement('div'); row.className='prow';
    row.innerHTML = `<span class="plabel" title="${key}">${key}</span>`;
    const holder = document.createElement('div'); holder.className='scrub'; row.appendChild(holder);
    if(Array.isArray(val)){                                  // vec: N linked number scrubbers
      val.forEach((comp,i)=>{
        const inp = numField(comp, m, v=>{ state[key][i]=v; commit(); });
        holder.appendChild(inp);
      });
    } else if(typeof val==='boolean'){
      const b=document.createElement('input'); b.type='checkbox'; b.className='pbool'; b.checked=val;
      b.onchange=()=>{ state[key]=b.checked; commit(); }; holder.appendChild(b);
    } else if(m.enum){
      const sel=document.createElement('select'); sel.className='penum';
      sel.innerHTML=m.enum.map(o=>`<option${o===val?' selected':''}>${o}</option>`).join('');
      sel.onchange=()=>{ state[key]=sel.value; commit(); }; holder.appendChild(sel);
    } else if(typeof val==='number'){
      // slider + scrubber number, both clamped to declared range
      const hasRange = (m.min!==undefined && m.max!==undefined);
      if(hasRange){
        const sl=document.createElement('input'); sl.type='range';
        sl.min=m.min; sl.max=m.max; sl.step=m.step||0.01; sl.value=val;
        const nf=numField(val, m, v=>{ state[key]=v; sl.value=v; commit(); });
        sl.oninput=()=>{ let v=parseFloat(sl.value); if(m.int) v=Math.round(v); state[key]=v; nf.value=fmt(v,m); };
        sl.onchange=commit;
        holder.appendChild(sl); holder.appendChild(nf);
      } else {
        holder.appendChild(numField(val, m, v=>{ state[key]=v; commit(); }));
      }
    } else {                                                 // string / expr
      const inp=document.createElement('input'); inp.type='text'; inp.className='ptext'; inp.value=val;
      inp.onchange=()=>{ state[key]=inp.value; commit(); }; holder.appendChild(inp);
    }
    container.appendChild(row);
  }
  return state;
  function fmt(v,m){ return m.int ? String(Math.round(v)) : (Math.round(v*1000)/1000); }
  // a number field you can type in OR drag horizontally to scrub (C4D virtual slider)
  function numField(v, m, set){
    const inp=document.createElement('input'); inp.type='text'; inp.className='pnum'; inp.value=fmt(v,m);
    const clamp=x=>{ if(m.min!==undefined) x=Math.max(m.min,x); if(m.max!==undefined) x=Math.min(m.max,x); if(m.int) x=Math.round(x); return x; };
    inp.onchange=()=>{ let x=parseFloat(inp.value); if(!isFinite(x)) return; x=clamp(x); inp.value=fmt(x,m); set(x); };
    let dragging=false, sx=0, sv=0;
    inp.addEventListener('pointerdown', e=>{ dragging=true; sx=e.clientX; sv=parseFloat(inp.value)||0; inp.setPointerCapture(e.pointerId); e.preventDefault(); });
    inp.addEventListener('pointermove', e=>{ if(!dragging) return; const step=(m.step||0.01); let x=clamp(sv+(e.clientX-sx)*step); inp.value=fmt(x,m); set(x); });
    inp.addEventListener('pointerup', e=>{ dragging=false; try{inp.releasePointerCapture(e.pointerId);}catch(_){} });
    return inp;
  }
}

/* ================================ P0.5-1: reusable Browser component ================================ */
// buildBrowser(container, opts) — opts:
//   items: [{id,label,thumb?,group?,keywords?,color?,meta?}]
//   view: 'grid'|'list'  sortKeys: ['label',...]  onSelect(item)  onActivate(item)  onContext(item,x,y)
//   selectedId: id
// Renders folder tree from group paths + search + sort + grid/list. Returns an API {refresh, setItems}.
function buildBrowser(container, opts){
  const state = { view: opts.view||'grid', sort: (opts.sortKeys&&opts.sortKeys[0])||'label',
                  folder: '(all)', q: '', items: opts.items||[], sel: opts.selectedId||null };
  container.classList.add('browser');
  container.innerHTML =
    `<div class="btools">
       <input class="bsearch" type="text" placeholder="search…" autocomplete="off">
       <select class="bsort">${(opts.sortKeys||['label']).map(k=>`<option value="${k}">sort: ${k}</option>`).join('')}</select>
       <button class="bview" title="grid / list">${state.view==='grid'?'☰':'▦'}</button>
     </div>
     <div class="bmain"><div class="bfolders"></div><div class="bitems ${state.view}"></div></div>`;
  const $q = container.querySelector('.bsearch'), $sort = container.querySelector('.bsort'),
        $view = container.querySelector('.bview'), $folders = container.querySelector('.bfolders'),
        $items = container.querySelector('.bitems');
  $q.oninput = ()=>{ state.q = $q.value.toLowerCase(); render(); };
  $sort.onchange = ()=>{ state.sort = $sort.value; render(); };
  $view.onclick = ()=>{ state.view = state.view==='grid'?'list':'grid'; $view.textContent = state.view==='grid'?'☰':'▦'; render(); };
  function folders(){
    const set = new Set(['(all)']);
    state.items.forEach(it=>{ if(it.group) set.add(it.group); });
    return [...set];
  }
  function filtered(){
    let list = state.items.slice();
    if(state.folder!=='(all)') list = list.filter(it=>it.group===state.folder);
    if(state.q) list = list.filter(it=>{
      const hay = (it.label+' '+(it.keywords||'')+' '+(it.group||'')).toLowerCase();
      return hay.indexOf(state.q)>=0;
    });
    list.sort((a,b)=>String(a[state.sort]||a.label).localeCompare(String(b[state.sort]||b.label)));
    return list;
  }
  function render(){
    $folders.innerHTML = folders().map(f=>`<div class="bf${f===state.folder?' on':''}" data-f="${encodeURIComponent(f)}">${f}</div>`).join('');
    $folders.querySelectorAll('.bf').forEach(el=>el.onclick=()=>{ state.folder=decodeURIComponent(el.dataset.f); render(); });
    $items.className = 'bitems '+state.view;
    const list = filtered();
    $items.innerHTML = list.map(it=>{
      const thumb = it.thumb ? `<img loading="lazy" src="${it.thumb}">`
                  : (it.color ? `<span class="bidot" style="background:${it.color}"></span>` : '');
      return `<div class="bi${it.id===state.sel?' on':''}" data-id="${encodeURIComponent(it.id)}" title="${(it.label||'').replace(/"/g,'')}">${thumb}<div class="bilabel">${it.label||''}</div></div>`;
    }).join('') || '<div class="note" style="padding:8px">No matches.</div>';
    $items.querySelectorAll('.bi').forEach(el=>{
      const id = decodeURIComponent(el.dataset.id);
      const item = state.items.find(x=>String(x.id)===id);
      el.onclick = ()=>{ state.sel=id; render(); opts.onSelect && opts.onSelect(item); };
      el.ondblclick = ()=>{ opts.onActivate && opts.onActivate(item); };
      if(opts.onContext) el.oncontextmenu = e=>{ e.preventDefault(); opts.onContext(item, e.clientX, e.clientY); };
    });
  }
  render();
  return {
    refresh: render,
    setItems(items){ state.items=items; render(); },
    setSelected(id){ state.sel=id; render(); },
  };
}

/* ================================ P1-1: WebGPU client raymarcher ================================ */
// The engine emits an exact per-object WGSL map(); /api/scene_wgsl composes them into scene_map(p)->vec2(dist,id).
// Here we wrap that in a compute sphere-tracer, run it on the client GPU, and blit to an overlay canvas. The
// server raymarch stays as fallback + ground truth. Detect support; degrade cleanly when navigator.gpu is absent.
const GPU = { ok:false, dev:null, ctx:null, canvas:null, pipeline:null, on:false, wgsl:null, mats:null, busy:false, dims:[0,0] };

async function gpuInit(){
  if(GPU.dev) return true;
  if(!navigator.gpu){ $('gpustatus').textContent='(not supported in this browser)'; return false; }
  try{
    const adapter = await navigator.gpu.requestAdapter();
    if(!adapter){ $('gpustatus').textContent='(no GPU adapter)'; return false; }
    GPU.dev = await adapter.requestDevice();
    GPU.ok = true; $('gpustatus').textContent='(ready)';
    // overlay canvas sits atop the server <img>, shown only when GPU preview is on
    const c = document.createElement('canvas'); c.id='gpucanvas';
    // pointer-events:none is NOT optional: this canvas covers the whole viewport, and without it the
    // three.js canvas underneath never sees a click, drag or wheel -- the viewport looks alive but is
    // completely dead to input. (Reported exactly that way.)
    c.style.cssText='position:absolute; inset:0; width:100%; height:100%; display:none; z-index:2;'
                  + 'pointer-events:none;';
    $('viewwrap').appendChild(c); GPU.canvas=c;
    return true;
  }catch(e){ $('gpustatus').textContent='(init failed)'; return false; }
}

function gpuShaderSource(sceneWgsl){
  // full compute shader: the injected scene_map + a sphere-trace kernel writing an rgba8 storage texture
  return sceneWgsl + `
struct Cam { origin: vec3<f32>, _p0: f32, fwd: vec3<f32>, _p1: f32, right: vec3<f32>, _p2: f32, up: vec3<f32>, _p3: f32, dims: vec2<f32>, _p4: vec2<f32> };
@group(0) @binding(0) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> cam: Cam;
@group(0) @binding(2) var<storage, read> matcol: array<vec4<f32>>;

fn calcNormal(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.0009, 0.0);
  return normalize(vec3<f32>(
    scene_map(p+e.xyy).x - scene_map(p-e.xyy).x,
    scene_map(p+e.yxy).x - scene_map(p-e.yxy).x,
    scene_map(p+e.yyx).x - scene_map(p-e.yyx).x));
}

@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<u32>(u32(cam.dims.x), u32(cam.dims.y));
  if(gid.x >= dims.x || gid.y >= dims.y){ return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / cam.dims * 2.0 - 1.0;
  let aspect = cam.dims.x / cam.dims.y;
  let rd = normalize(cam.fwd + cam.right*uv.x*aspect*0.55 + cam.up*(-uv.y)*0.55);
  var t = 0.0; var hit = -1.0; var col = vec3<f32>(0.06, 0.07, 0.10);
  for(var i=0; i<96; i=i+1){
    let p = cam.origin + rd*t;
    let d = scene_map(p);
    if(d.x < 0.001){ hit = d.y; 
      let n = calcNormal(p);
      let l = normalize(vec3<f32>(-0.5, 0.8, 0.4));
      let diff = max(dot(n,l), 0.0)*0.85 + 0.15;
      let mi = i32(hit);
      let base = matcol[mi].xyz;
      col = base*diff;
      break;
    }
    t = t + d.x;
    if(t > 24.0){ break; }
  }
  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(pow(col, vec3<f32>(0.4545)), 1.0));
}`;
}

async function gpuBuildPipeline(sceneWgsl){
  const dev = GPU.dev;
  const mod = dev.createShaderModule({ code: gpuShaderSource(sceneWgsl) });
  // surface compile errors honestly
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter(m=>m.type==='error');
  if(errs.length){ notify('GPU shader error: '+errs[0].message, 'err'); return false; }
  GPU.pipeline = dev.createComputePipeline({ layout:'auto', compute:{ module:mod, entryPoint:'main' } });
  GPU.wgsl = sceneWgsl;
  return true;
}

async function gpuRefreshScene(){
  if(!GPU.ok) return false;
  try{
    const r = await (await fetch('api/scene_wgsl')).json();
    if(!r.available || (r.excluded && r.excluded.length)){
      // non-analytic objects present -> leCore's field cache covers the WHOLE scene uniformly
      return await gpuFieldScene();
    }
    GPU.mats = r.materials;
    const src = r.wgsl.replace(/\\n/g,'\n');
    const okp = await gpuBuildPipeline(src);
    if(okp){ GPU.mode='exact'; $('gpustatus').textContent = `(${r.count} objects on GPU — exact analytic)`; }
    return okp;
  }catch(e){ $('gpustatus').textContent='(scene emit failed)'; return false; }
}

function gpuFrame(){
  if(!GPU.on || !GPU.ok || !GPU.pipeline || GPU.busy) return;
  const dev=GPU.dev, cv=GPU.canvas;
  const w = Math.max(2, canvas.clientWidth|0), h = Math.max(2, canvas.clientHeight|0);
  if(cv.width!==w || cv.height!==h){ cv.width=w; cv.height=h; GPU.dims=[0,0]; }
  GPU.busy=true;
  const _gpuT0 = performance.now();
  try{
    // (re)create the target texture + context on size change
    if(GPU.dims[0]!==w || GPU.dims[1]!==h){
      GPU.ctx = cv.getContext('webgpu');
      GPU.ctx.configure({ device:dev, format:navigator.gpu.getPreferredCanvasFormat(), usage: GPUTextureUsage.COPY_DST|GPUTextureUsage.RENDER_ATTACHMENT, alphaMode:'premultiplied' });
      GPU.tex = dev.createTexture({ size:[w,h], format:'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC });
      GPU.dims=[w,h];
    }
    // camera uniform (matches the three.js spherical camera)
    const sp=Math.sin(camPhi), cp=Math.cos(camPhi);
    const ox=camTarget.x+camDist*sp*Math.cos(camTheta), oy=camTarget.y+camDist*cp, oz=camTarget.z+camDist*sp*Math.sin(camTheta);
    const fx=camTarget.x-ox, fy=camTarget.y-oy, fz=camTarget.z-oz; const fl=Math.hypot(fx,fy,fz);
    const fwd=[fx/fl,fy/fl,fz/fl];
    const rx=fwd[2],rz=-fwd[0]; const rl=Math.hypot(rx,0,rz)||1; const right=[rx/rl,0,rz/rl];
    const up=[right[1]*fwd[2]-right[2]*fwd[1], right[2]*fwd[0]-right[0]*fwd[2], right[0]*fwd[1]-right[1]*fwd[0]];
    const camBuf = dev.createBuffer({ size:96, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST });
    const cd = new Float32Array(24);
    cd.set([ox,oy,oz,0, fwd[0],fwd[1],fwd[2],0, right[0],right[1],right[2],0, up[0],up[1],up[2],0, w,h,0,0]);
    dev.queue.writeBuffer(camBuf, 0, cd);
    // material colors storage
    const mc = new Float32Array(Math.max(1,GPU.mats.length)*4);
    GPU.mats.forEach((m,i)=>{ mc[i*4]=m[0]; mc[i*4+1]=m[1]; mc[i*4+2]=m[2]; mc[i*4+3]=1; });
    const matBuf = dev.createBuffer({ size:mc.byteLength, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST });
    dev.queue.writeBuffer(matBuf, 0, mc);
    const entries = [
      { binding:0, resource:GPU.tex.createView() },
      { binding:1, resource:{ buffer:camBuf } },
      { binding:2, resource:{ buffer:matBuf } } ];
    if(GPU.mode==='field'){
      entries.push({ binding:3, resource:GPU.fieldTex.createView() });
      entries.push({ binding:4, resource:GPU.sampler });
      entries.push({ binding:5, resource:{ buffer:GPU.fieldInfo } });
      entries.push({ binding:6, resource:GPU.palTex.createView() });
    }
    const bind = dev.createBindGroup({ layout:GPU.pipeline.getBindGroupLayout(0), entries });
    const enc = dev.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(GPU.pipeline); pass.setBindGroup(0, bind);
    pass.dispatchWorkgroups(Math.ceil(w/8), Math.ceil(h/8)); pass.end();
    enc.copyTextureToTexture({texture:GPU.tex}, {texture:GPU.ctx.getCurrentTexture()}, [w,h]);
    dev.queue.submit([enc.finish()]);
    PERF.gpuMs = (performance.now()-_gpuT0).toFixed(1)+'ms';
  }catch(e){ notify('GPU frame error: '+e.message, 'err'); GPU.on=false; $('gpupreview').checked=false; gpuSetVisible(false); }
  GPU.busy=false;
}

function gpuSetVisible(on){ if(GPU.canvas) GPU.canvas.style.display = on?'block':'none'; }

$('gpupreview') && ($('gpupreview').onchange = async (e)=>{
  if(e.target.checked){
    const ready = await gpuInit();
    if(!ready){ e.target.checked=false; return; }
    const scene = await gpuRefreshScene();
    if(!scene){ e.target.checked=false; return; }
    GPU.on=true; gpuSetVisible(true); gpuFrame();
  } else { GPU.on=false; gpuSetVisible(false); }
});

/* ================================ P1-3: frame budget + perf HUD ================================ */
// Surfaces what's already measured: client FPS + three.js draw calls/triangles, last server render bake/trace,
// GPU dispatch time. Toggle with the ~ key. Frame-budget mode steps preview quality down when over budget.
const PERF = { on:false, fps:0, frames:0, lastT:performance.now(), bake:'-', trace:'-', srvFrame:'-',
               gpuMs:'-', engine:'server', budget:0, overCount:0 };

function perfTick(){
  PERF.frames++;
  const now=performance.now();
  if(now-PERF.lastT >= 500){
    PERF.fps = Math.round(PERF.frames*1000/(now-PERF.lastT));
    PERF.frames=0; PERF.lastT=now;
    if(PERF.on) perfRender();
    // frame-budget enforcement (client FPS proxy): if consistently under target, step preview quality down
    if(PERF.budget>0 && $('quality') && !GPU.on){
      const targetFps = 1000/PERF.budget;
      if(PERF.fps < targetFps*0.8){ PERF.overCount++; } else { PERF.overCount=0; }
      if(PERF.overCount>=3){
        const q=$('quality'); const nv=Math.max(0.15, (+q.value)-0.1);
        if(nv < +q.value){ q.value=nv; $('qval').textContent=nv.toFixed(2); notify(`Holding ${PERF.budget}ms budget: preview quality → ${nv.toFixed(2)}`); scheduleRender(); }
        PERF.overCount=0;
      }
    }
  }
}
function perfRender(){
  const info = renderer.info;
  const tris = info.render.triangles, calls = info.render.calls;
  PERF.engine = GPU.on ? 'WebGPU (client)' : 'server';
  const budgetTxt = PERF.budget>0 ? `${PERF.budget}ms target` : 'off';
  const rows = [
    ['engine', PERF.engine],
    ['fps', PERF.fps],
    ['draw calls', calls],
    ['triangles', tris.toLocaleString()],
    ['—', ''],
    ['server bake', PERF.bake],
    ['server trace', PERF.trace],
    ['server frame', PERF.srvFrame],
    ['gpu dispatch', GPU.on?PERF.gpuMs:'—'],
    ['budget', budgetTxt],
  ];
  $('perfhud').innerHTML = '<div class="phhead">performance</div>' + rows.map(([k,v])=>
    k==='—' ? '<div class="phhead">server render</div>'
            : `<div class="ph"><span>${k}</span><b>${v}</b></div>`).join('');
}
// parse the server render header for bake/trace/frame
function perfIngestHeader(h){
  if(!h) return;
  let m;
  if((m=h.match(/bake=([\d.]+)s/))) PERF.bake=m[1]+'s';
  if((m=h.match(/trace=([\d.]+)s/))) PERF.trace=m[1]+'s';
  if((m=h.match(/frame=([\d.]+)ms/))) PERF.srvFrame=m[1]+'ms';
  if(PERF.on) perfRender();
}
// (~ toggles the perf HUD -- KEYMAP, C4)
$('framebudget') && ($('framebudget').onchange = e=>{ PERF.budget = +e.target.value; PERF.overCount=0; if(PERF.on) perfRender(); });

/* ================================ P1-7: starter library (example graphs) ================================ */
const EXAMPLES = [
  { id:'sphere-box-morph', label:'Sphere ⇄ box morph (blend node)', group:'blend',
    desc:'Two shapes into a blend node — drag the t slider to morph sphere into box, or wire a field to blend spatially.',
    graph:{ nodes:{ s:{type:'sdf_sphere', params:{radius:0.6}}, bx:{type:'sdf_box', params:{size:[0.5,0.5,0.5]}},
                    bl:{type:'sdf_blend', params:{t:0.5, mode:'morph', k:0.3}} },
            edges:[{src:'s', src_socket:'out', dst:'bl', dst_socket:'a'},
                   {src:'bx', src_socket:'out', dst:'bl', dst_socket:'b'}] },
    pos:{ s:[60,40], bx:[60,180], bl:[300,100] } },
  { id:'field-blend', label:'Field-driven blend', group:'blend',
    desc:'A sine field masks the blend so a sphere becomes a box across space — the texture-controlled morph.',
    graph:{ nodes:{ s:{type:'sdf_sphere', params:{radius:0.7}}, bx:{type:'sdf_box', params:{size:[0.55,0.55,0.55]}},
                    f:{type:'formula_field', params:{expr:'0.5+0.5*sin(4*x)'}},
                    bl:{type:'sdf_blend', params:{mode:'morph'}} },
            edges:[{src:'s', src_socket:'out', dst:'bl', dst_socket:'a'},
                   {src:'bx', src_socket:'out', dst:'bl', dst_socket:'b'},
                   {src:'f', src_socket:'out', dst:'bl', dst_socket:'field'}] },
    pos:{ s:[60,30], bx:[60,150], f:[60,290], bl:[320,120] } },
  { id:'mandelbulb', label:'Mandelbulb fractal', group:'fractal',
    desc:'Exact-GLSL Mandelbulb — build the node to get an analytic fractal solid.',
    graph:{ nodes:{ mb:{type:'sdf_mandelbulb', params:{power:8.0, iterations:8, bailout:2.0}} }, edges:[] },
    pos:{ mb:[120,80] } },
  { id:'menger', label:'Menger sponge', group:'fractal',
    desc:'Exact-GLSL Menger sponge fractal.',
    graph:{ nodes:{ mg:{type:'sdf_menger', params:{iterations:3, size:1.0}} }, edges:[] },
    pos:{ mg:[120,80] } },
  { id:'twist-box', label:'Twisted rounded box', group:'sdf',
    desc:'A box fed through a twist warp — analytic, exportable to Shadertoy.',
    graph:{ nodes:{ b:{type:'sdf_box', params:{size:[0.5,0.5,0.5]}}, t:{type:'sdf_twist', params:{k:1.8}} },
            edges:[{src:'b', src_socket:'out', dst:'t', dst_socket:'a'}] },
    pos:{ b:[60,60], t:[280,60] } },
  { id:'smooth-union', label:'Smooth-union of sphere + box', group:'sdf',
    desc:'Two primitives blended with a smooth-min — the classic SDF metaball look.',
    graph:{ nodes:{ s:{type:'sdf_sphere', params:{radius:0.55}}, b:{type:'sdf_box', params:{size:[0.5,0.5,0.5]}},
                    u:{type:'sdf_smooth_union', params:{k:0.3}} },
            edges:[{src:'s', src_socket:'out', dst:'u', dst_socket:'a'},
                   {src:'b', src_socket:'out', dst:'u', dst_socket:'b'}] },
    pos:{ s:[60,40], b:[60,150], u:[300,95] } },
  { id:'formula-displace', label:'Formula-displaced sphere', group:'field',
    desc:'A sine formula field warps a sphere surface — Shadertoy-style geometry.',
    graph:{ nodes:{ s:{type:'sdf_sphere', params:{radius:0.7}},
                    f:{type:'formula_field', params:{expr:'0.08*sin(9*x)*sin(9*z)'}},
                    d:{type:'sdf_displace_field', params:{amount:0.12}} },
            edges:[{src:'s', src_socket:'out', dst:'d', dst_socket:'sdf'},
                   {src:'f', src_socket:'out', dst:'d', dst_socket:'field'}] },
    pos:{ s:[60,40], f:[60,160], d:[300,95] } },
  { id:'curl-displace', label:'Curl-noise displaced sphere', group:'field',
    desc:'Divergence-free curl noise driving surface displacement — organic detail.',
    graph:{ nodes:{ s:{type:'sdf_sphere', params:{radius:0.7}},
                    f:{type:'curl_field', params:{scale:1.6, seed:3, octaves:3}},
                    d:{type:'sdf_displace_field', params:{amount:0.1}} },
            edges:[{src:'s', src_socket:'out', dst:'d', dst_socket:'sdf'},
                   {src:'f', src_socket:'out', dst:'d', dst_socket:'field'}] },
    pos:{ s:[60,40], f:[60,160], d:[300,95] } },
];
let EXBROWSER=null;
async function loadExample(ex){
  const r = await nodeApi({action:'load', graph:ex.graph, pos:ex.pos});
  if(r && r.error){ status('Load failed: '+r.error); return; }
  // open node editor if closed
  if(!NODE.open){ NODE.open=true; $('nodeeditor').style.display='block';
    if(!Object.keys(NODE.types).length){ const tt=await (await fetch('api/nodes/types')).json(); for(const t of tt.types||[]) NODE.types[t.type]=t; } }
  nodeRefresh();
  status(`Loaded "${ex.label}" — Build the output node in the Node editor to see it.`);
  const dlg=$('dlg-examples'); if(dlg) dlg.classList.remove('open');
}
document.querySelectorAll('[data-dlg="dlg-examples"]').forEach(b=>b.addEventListener('click', ()=>{
  const host=$('examplelist'); if(!host) return;
  if(!EXBROWSER){
    EXBROWSER = buildBrowser(host, {
      items: EXAMPLES.map(e=>({id:e.id, label:e.label, group:e.group, keywords:e.desc})),
      view:'list', sortKeys:['label','group'],
      onSelect: e=>status(EXAMPLES.find(x=>x.id===e.id).desc),
      onActivate: e=>loadExample(EXAMPLES.find(x=>x.id===e.id)),
    });
  }
}));

/* ================================ P1-6: camera dialog, DOF, bookmarks ================================ */
// FOV drives the three.js preview camera live; the value also rides camParams() into server render/photo.
$('fov') && ($('fov').oninput = ()=>{
  const f=+$('fov').value; $('fovval').textContent=f+'°';
  camera.fov=f; camera.updateProjectionMatrix(); scheduleRender();
});
$('focus') && ($('focus').oninput = ()=>$('focusval').textContent=(+$('focus').value).toFixed(1));
$('fstop') && ($('fstop').oninput = ()=>$('fstopval').textContent=(+$('fstop').value).toFixed(1));
$('dof') && ($('dof').onchange = ()=>{ if($('dof').checked) status('Depth of field on — render a photo to see it (live preview stays sharp).'); });

// pick focus distance by clicking an object: use the picked point's distance from the eye
let focusPickArmed=false;
$('focuspick') && ($('focuspick').onclick = ()=>{ focusPickArmed=true; status('Click an object to set the focus distance…'); });
canvas.addEventListener('pointerdown', ev=>{
  if(!focusPickArmed || ev.button!==0) return;
  const hit = pickScene(ev);
  if(hit && hit.point){
    const d = camera.position.distanceTo(hit.point);
    $('focus').value = Math.min(12, Math.max(0.5, d)).toFixed(1); $('focusval').textContent=(+$('focus').value).toFixed(1);
    if(!$('dof').checked){ $('dof').checked=true; }
    status(`Focus set to ${(+$('focus').value).toFixed(1)} — render a photo to see the depth of field.`);
  }
  focusPickArmed=false;
}, true);

// view bookmarks (client-side): store spherical camera + fov, recall exactly
let BOOKMARKS=[];
function renderBookmarks(){
  const el=$('bmlist'); if(!el) return;
  el.innerHTML = BOOKMARKS.length ? BOOKMARKS.map((b,i)=>
    `<div style="display:flex;justify-content:space-between;gap:8px;padding:3px 0">
       <span style="cursor:pointer;color:var(--acc)" data-recall="${i}">▸ ${b.name} <kbd>⇧${i+1}</kbd></span>
       <span style="cursor:pointer;opacity:0.6" data-del="${i}">✕</span></div>`).join('')
    : '<span style="opacity:0.6">No bookmarks yet — press Shift+1-9 to recall.</span>';
  el.querySelectorAll('[data-recall]').forEach(s=>s.onclick=()=>recallBookmark(+s.dataset.recall));
  el.querySelectorAll('[data-del]').forEach(s=>s.onclick=()=>{ BOOKMARKS.splice(+s.dataset.del,1); renderBookmarks(); });
}
function recallBookmark(i){
  const b=BOOKMARKS[i]; if(!b) return;
  camTheta=b.th; camPhi=b.ph; camDist=b.d; camTarget.set(b.tx,b.ty,b.tz);
  if($('fov')){ $('fov').value=b.fov; $('fovval').textContent=b.fov+'°'; camera.fov=b.fov; camera.updateProjectionMatrix(); }
  applyCam(); placeGizmo(); scheduleRender(); status('View: '+b.name);
}
$('bmsave') && ($('bmsave').onclick = ()=>{
  const name = 'View '+(BOOKMARKS.length+1);
  BOOKMARKS.push({name, th:camTheta, ph:camPhi, d:camDist, tx:camTarget.x, ty:camTarget.y, tz:camTarget.z,
                  fov:+($('fov')?$('fov').value:45)});
  renderBookmarks(); status('Saved '+name+' (recall with Shift+'+BOOKMARKS.length+')');
});
// (Shift+digit recalls a bookmark -- KEYMAP, C4)

// (turntable moved into the Render View toolbar -- see rvTurntable(), B5)

// 'C' opens the camera dialog
// (C opens the camera dialog -- KEYMAP, C4)
renderBookmarks();

/* ================================ P1-5: lighting dialog ================================ */
['lightaz','lightel','sun','ambient'].forEach(id=>{
  const el=$(id); if(!el) return;
  const lbl={lightaz:'lightazval',lightel:'lightelval',sun:'sunval',ambient:'ambval'}[id];
  const unit={lightaz:'°',lightel:'°',sun:'',ambient:''}[id];
  el.oninput = ()=>{ const v=+el.value; $(lbl).textContent = (id==='sun'||id==='ambient'?v.toFixed(id==='ambient'?2:1):v)+unit; scheduleRender(); };
});
// (L opens the lighting dialog -- KEYMAP, C4)

/* ================================ CAD sweep: shell, cross-sections, layers ================================ */
$('tShell') && ($('tShell').onclick = async ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  const t = parseFloat(await uiPrompt('Shell', 'Wall thickness (exact for analytic objects):','0.06'));
  if(!isFinite(t)) return;
  try{
    const r = await api('op', {op:'shell', object:ACTIVE, thickness:t});
    applyResp(r);
    status(r.note || 'Shelled.');
    if(r.exact===false) notify('Shell used the mesh offset route (no analytic tree on this object) — thickness is approximate, not exact.');
  }catch(e){}
});
// 3D clip: rides camParams so preview + photo both cut
function clipParam(){
  return ($('clipon') && $('clipon').checked)
    ? `&clip=${$('clipaxis').value},${$('clipoff').value},${$('clipflip').dataset.sign||'1'}` : '';
}
['clipon','clipaxis'].forEach(id=>{ const el=$(id); if(el) el.onchange=()=>scheduleRender(); });
$('clipoff') && ($('clipoff').oninput = ()=>{ $('clipoffval').textContent=(+$('clipoff').value).toFixed(2); scheduleRender(); });
$('clipflip') && ($('clipflip').onclick = ()=>{ const b=$('clipflip'); b.dataset.sign = (b.dataset.sign==='-1')?'1':'-1'; b.textContent = b.dataset.sign==='-1'?'flip ↺':'flip'; scheduleRender(); });
// 2D exact section
$('secoff') && ($('secoff').oninput = ()=>$('secoffval').textContent=(+$('secoff').value).toFixed(2));
$('secbtn') && ($('secbtn').onclick = async ()=>{
  const obj = ($('seconly').checked && ACTIVE) ? `&object=${encodeURIComponent(ACTIVE)}` : '';
  const url = `api/section?axis=${$('secaxis').value}&offset=${$('secoff').value}`+
              `&count=${$('seccount').value}&span=${$('secspan').value}&res=240${obj}`;
  $('secmeta').textContent='sampling…';
  try{
    const r=await fetch(url);
    if(!r.ok){ $('secmeta').textContent='failed: '+(await r.json()).error; return; }
    $('secimg').src = URL.createObjectURL(await r.blob());
    $('secmeta').textContent = r.headers.get('X-Holostuff-Section')||'';
  }catch(e){ $('secmeta').textContent='failed: '+e.message; }
});
// layers editor: "0.08 gold, 0.10 copper"
$('layersbtn') && ($('layersbtn').onclick = async ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  const txt = $('layersbox').value.trim();
  const layers = txt ? txt.split(',').map(s=>{ const m=s.trim().split(/\s+/); return [parseFloat(m[0]), m.slice(1).join(' ')]; }) : [];
  try{
    const r = await api('op', {op:'set_layers', object:ACTIVE, layers});
    status(layers.length ? `Layers set (${layers.map(l=>l[1]).join(' → ')}) — render a section to reveal them.` : 'Layers cleared.');
  }catch(e){}
});
// (X opens cross-sections -- KEYMAP, C4)

/* ============== P1-1 REVISED: leCore-fed field mode for the GPU tracer ==============
   Verdict of the pipeline audit: leCore already ships a rev-cached, per-object-incremental scene volume
   (float16 distance + uint8 palette, /api/field_meta + /api/field_tex.bin) built FOR GPU upload. The client
   must consume that cache, not re-derive scene composition. Two GPU modes, both leCore-fed:
     'exact' — all-analytic scenes trace the engine's own emitted WGSL map (infinite resolution);
     'field' — anything else traces leCore's baked volume as a 3D texture (uniform: meshes, sculpt, edits).
   Empirically verified layout: z-fastest, i = z + res*(y + res*x). */
GPU.mode='exact'; GPU.fieldRev=-1; GPU.fieldTex=null; GPU.palTex=null; GPU.fieldInfo=null; GPU.sampler=null;

function gpuFieldShader(){
  return `
struct Cam { origin: vec3<f32>, _p0: f32, fwd: vec3<f32>, _p1: f32, right: vec3<f32>, _p2: f32, up: vec3<f32>, _p3: f32, dims: vec2<f32>, _p4: vec2<f32> };
struct FieldInfo { lo: vec3<f32>, ground: f32, hi: vec3<f32>, res: f32 };
@group(0) @binding(0) var outTex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> cam: Cam;
@group(0) @binding(2) var<storage, read> matcol: array<vec4<f32>>;
@group(0) @binding(3) var fieldTex: texture_3d<f32>;
@group(0) @binding(4) var fieldSamp: sampler;
@group(0) @binding(5) var<uniform> fi: FieldInfo;
@group(0) @binding(6) var palTex: texture_3d<u32>;

fn scene_d(p: vec3<f32>) -> f32 {
  let ext = fi.hi - fi.lo;
  let uvw = (p - fi.lo) / ext;
  // outside the baked box: conservative distance to the box keeps the march safe
  let q = abs(p - 0.5*(fi.lo+fi.hi)) - 0.5*ext;
  let outside = length(max(q, vec3<f32>(0.0)));
  if(outside > 0.0){ return outside + 0.01; }
  let d = textureSampleLevel(fieldTex, fieldSamp, uvw, 0.0).r;
  return min(d, p.y - fi.ground);
}
fn calcN(p: vec3<f32>) -> vec3<f32> {
  let e = vec2<f32>(0.012, 0.0);
  return normalize(vec3<f32>(scene_d(p+e.xyy)-scene_d(p-e.xyy), scene_d(p+e.yxy)-scene_d(p-e.yxy), scene_d(p+e.yyx)-scene_d(p-e.yyx)));
}
@compute @workgroup_size(8,8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = vec2<u32>(u32(cam.dims.x), u32(cam.dims.y));
  if(gid.x >= dims.x || gid.y >= dims.y){ return; }
  let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + 0.5) / cam.dims * 2.0 - 1.0;
  let aspect = cam.dims.x / cam.dims.y;
  let rd = normalize(cam.fwd + cam.right*uv.x*aspect*0.55 + cam.up*(-uv.y)*0.55);
  var t = 0.0; var col = vec3<f32>(0.06, 0.07, 0.10);
  for(var i=0; i<128; i=i+1){
    let p = cam.origin + rd*t;
    let d = scene_d(p);
    if(d < 0.004){
      let n = calcN(p);
      let l = normalize(vec3<f32>(-0.5, 0.8, 0.4));
      let diff = max(dot(n,l), 0.0)*0.85 + 0.15;
      let ext = fi.hi - fi.lo;
      let vox = clamp((p - fi.lo)/ext, vec3<f32>(0.0), vec3<f32>(0.999)) * fi.res;
      let pid = textureLoad(palTex, vec3<i32>(vox), 0).r;
      var base = matcol[pid].xyz;
      if(p.y - fi.ground < 0.02){ base = matcol[0].xyz; }
      col = base*diff;
      break;
    }
    t = t + max(d, 0.004);
    if(t > 30.0){ break; }
  }
  textureStore(outTex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(pow(col, vec3<f32>(0.4545)), 1.0));
}`;
}

async function gpuFieldScene(){
  const meta = await (await fetch('api/field_meta?res=64')).json();
  if(meta.error){ $('gpustatus').textContent='(field bake failed)'; return false; }
  const dev=GPU.dev, res=meta.res;
  if(meta.rev!==GPU.fieldRev || !GPU.fieldTex){
    const buf = await (await fetch(`api/field_tex.bin?res=64&rev=${meta.rev}`)).arrayBuffer();
    const n=res*res*res;
    const f16 = new Uint16Array(buf, 0, n);
    const pal = new Uint8Array(buf, n*2, n);
    GPU.fieldTex = dev.createTexture({size:[res,res,res], dimension:'3d', format:'r16float', usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    // layout: i = z + res*(y + res*x) — z fastest → upload as width=z, height=y, depth=x, then swizzle in
    // shader? Simpler: upload with width=res rows matching z-fastest = treat texture axes as (z,y,x) and
    // sample uvw.zyx. We swap in JS instead: sample coordinate swizzle done here by uploading as-is and
    // swizzling uvw in the sampler call would complicate WGSL; we upload transposed once (CPU, ~1MB).
    const f16t = new Uint16Array(n), palt = new Uint8Array(n);
    for(let x=0;x<res;x++) for(let y=0;y<res;y++){
      const src=res*(y+res*x), dst=x+res*y;
      for(let z=0;z<res;z++){ const s=src+z, d2=dst+res*res*z; f16t[d2]=f16[s]; palt[d2]=pal[s]; }
    }
    dev.queue.writeTexture({texture:GPU.fieldTex}, f16t, {bytesPerRow:res*2, rowsPerImage:res}, [res,res,res]);
    GPU.palTex = dev.createTexture({size:[res,res,res], dimension:'3d', format:'r8uint', usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    dev.queue.writeTexture({texture:GPU.palTex}, palt, {bytesPerRow:res, rowsPerImage:res}, [res,res,res]);
    GPU.sampler = GPU.sampler || dev.createSampler({magFilter:'linear', minFilter:'linear'});
    GPU.fieldInfo = dev.createBuffer({size:32, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    const fi=new Float32Array(8); fi.set([...meta.lo, meta.ground, ...meta.hi, res]);
    dev.queue.writeBuffer(GPU.fieldInfo, 0, fi);
    GPU.mats = meta.palette.map(m=>m.albedo);
    GPU.fieldRev = meta.rev;
  }
  const mod = dev.createShaderModule({code: gpuFieldShader()});
  const info = await mod.getCompilationInfo();
  const errs = info.messages.filter(m=>m.type==='error');
  if(errs.length){ notify('GPU field shader error: '+errs[0].message,'err'); return false; }
  GPU.pipeline = dev.createComputePipeline({layout:'auto', compute:{module:mod, entryPoint:'main'}});
  GPU.mode='field';
  $('gpustatus').textContent = `(whole scene via leCore field cache, ${res}³ rev ${meta.rev})`;
  return true;
}

/* ================================ display modes (textured/flat/wire/vertex) ================================ */
let SHOWWIRE = false;    // overlay wireframe on any display type (for selection/editing while shaded)
function setDisplay(mode){
  DISPLAY = mode;
  for(const [id,o] of OBJS) applyDisplay(o);
  refreshSelectionVisuals();
  document.querySelectorAll('#dispbar [data-disp]').forEach(b=>b.classList.toggle('on', b.dataset.disp===mode));
  const label = {textured:'Textured',flat:'Flat',wireframe:'Wireframe',vertex:'Vertex',bbox:'Bounding boxes'}[mode]||mode;
  const cap = $('dispcap'); if(cap) cap.textContent = label + (SHOWWIRE?' + wire':'');
  status('Display: '+mode + (SHOWWIRE?' + wireframe':''));
}
function ensureBox(o){
  // lazily build a wire bounding box for this object, cached; refreshed when geometry changes
  const geom = o.mesh.geometry;
  geom.computeBoundingBox();
  const bb = geom.boundingBox;
  const key = `${bb.min.x.toFixed(3)},${bb.min.y.toFixed(3)},${bb.min.z.toFixed(3)},${bb.max.x.toFixed(3)},${bb.max.y.toFixed(3)},${bb.max.z.toFixed(3)}`;
  if(o.box && o._boxKey===key) return o.box;
  if(o.box){ o.group.remove(o.box); o.box.geometry.dispose(); }
  const box3 = new THREE.Box3(bb.min.clone(), bb.max.clone());
  o.box = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(
    bb.max.x-bb.min.x, bb.max.y-bb.min.y, bb.max.z-bb.min.z)), boxMat);
  const c = new THREE.Vector3(); box3.getCenter(c); o.box.position.copy(c);
  o._boxKey = key; o.group.add(o.box);
  return o.box;
}
function objAsBox(o, forceBox){
  // render this object as a bounding box only (hide solid/wire/points); used by bbox mode + unfocused-boxes
  const box = ensureBox(o);
  const selected = selObjs.has(o.d.id) || o.d.id===ACTIVE;
  box.material = selected ? boxMatSel : boxMat;
  box.visible = true;
  o.mesh.visible = false; o.wire.visible = false; o.points.visible = false;
  o._boxed = true;
}

function applyDisplay(o){
  // reset any prior boxed state
  if(o.box){ o.box.visible = false; }
  o._boxed = false;
  const isSel = selObjs.has(o.d.id) || o.d.id===ACTIVE;
  // per-object box override, OR whole-scene bbox mode, OR unfocused-boxes speed mode for non-selected objects
  if(o._forceBox || DISPLAY==='bbox' || (BOXUNFOCUSED && !isSel)){
    objAsBox(o);
    o._wireForced = false;
    return;
  }
  const shownWire = (DISPLAY==='wireframe') || SHOWWIRE;
  // base mesh material by mode
  if(DISPLAY==='textured'){ o.mesh.material = o.texMat || (o.smooth ? meshMat : meshMatF); o.mesh.visible = true; }
  else if(DISPLAY==='flat'){ o.mesh.material = o.smooth ? flatMat : flatMatF; o.mesh.visible = true; }
  else if(DISPLAY==='wireframe'){ o.mesh.visible = false; }   // wireframe-only: hide the solid
  else if(DISPLAY==='vertex'){
    o.mesh.visible = false; o.points.material.size = 0.05;
    // show real vertex colours as the point cloud (unless actively editing this object)
    if(!(o.d.id===ACTIVE && (MODE==='vertex'||MODE==='face'))){
      const src=o.mesh.geometry.attributes.color, dst=o.points.geometry.attributes.color;
      if(src && dst){ for(let i=0;i<dst.count;i++) dst.setXYZ(i, src.getX(i),src.getY(i),src.getZ(i)); dst.needsUpdate=true; }
    }
  }
  o._wireForced = shownWire;
}
function toggleBoxUnfocused(){ BOXUNFOCUSED=!BOXUNFOCUSED; for(const [,o] of OBJS) applyDisplay(o); refreshSelectionVisuals();
  const b=$('boxunfocused'); if(b) b.classList.toggle('on', BOXUNFOCUSED); status('Boxes for unfocused objects: '+(BOXUNFOCUSED?'on':'off')); }
function toggleWireOverlay(){ SHOWWIRE=!SHOWWIRE; for(const [,o] of OBJS) applyDisplay(o); refreshSelectionVisuals();
  const b=$('wireoverlay'); if(b) b.classList.toggle('on', SHOWWIRE); status('Wireframe overlay: '+(SHOWWIRE?'on':'off')); }

/* ================================ split viewports (1 / 2 / 4) ================================ */
// Each view has its own camera + spherical state. View 0 is the interactive "main" (drives ACTIVE camera vars).
// Standard modeling layout: perspective + Top/Front/Right orthos for the 4-up.
let SPLIT = 1;
const VIEWS = [
  { name:'Persp', persp:true,  th:()=>camTheta, ph:()=>camPhi },   // 0: the live camera (mirrors global vars)
  { name:'Top',   persp:false, dir:[0, 1, 0] },
  { name:'Front', persp:false, dir:[0, 0, 1] },
  { name:'Right', persp:false, dir:[1, 0, 0] },
];
VIEWS.forEach((v,i)=>{
  if(i===0){ v.cam = camera; }
  else {
    v.cam = new THREE.OrthographicCamera(-2,2,2,-2,0.01,200);
  }
});
function viewRects(w,h){
  if(SPLIT===1) return [[0,0,w,h]];
  if(SPLIT===2) return [[0,0,w/2,h],[w/2,0,w/2,h]];
  return [[0,h/2,w/2,h/2],[w/2,h/2,w/2,h/2],[0,0,w/2,h/2],[w/2,0,w/2,h/2]];  // TL TR BL BR
}
function setSplit(n){
  SPLIT=n;
  document.querySelectorAll('#dispbar [data-split]').forEach(b=>b.classList.toggle('on', +b.dataset.split===n));
  resize(); status(n===1?'Single view':n+' views');
}
function updateOrtho(v, rectW, rectH){
  const d = camDist, aspect = rectW/rectH;
  const c = v.cam;
  c.left=-d*aspect*0.5; c.right=d*aspect*0.5; c.top=d*0.5; c.bottom=-d*0.5;
  const dir = v.dir;
  c.position.set(camTarget.x+dir[0]*d*2, camTarget.y+dir[1]*d*2, camTarget.z+dir[2]*d*2);
  c.up.set(dir[1]?0:0, dir[1]?0:1, dir[1]?1:0);   // Top view up = +Z, others +Y
  c.lookAt(camTarget); c.updateProjectionMatrix();
}
function renderViews(){
  const w=$('viewwrap').clientWidth, h=$('viewwrap').clientHeight;
  const rects = viewRects(w, h);
  renderer.setScissorTest(SPLIT>1);
  const nv = SPLIT===1?1:(SPLIT===2?2:4);
  for(let i=0;i<nv;i++){
    const [x,y,rw,rh]=rects[i]; const v=VIEWS[i];
    if(SPLIT>1){ renderer.setViewport(x,y,rw,rh); renderer.setScissor(x,y,rw,rh); }
    else renderer.setViewport(0,0,w,h);
    if(v.persp){ v.cam.aspect=rw/rh; v.cam.updateProjectionMatrix(); }
    else updateOrtho(v, rw, rh);
    renderer.render(scene, v.cam);
  }
  renderer.setScissorTest(false);
}

/* display + split toolbar wiring */
document.querySelectorAll('#dispbar [data-disp]').forEach(b=>b.onclick=()=>setDisplay(b.dataset.disp));
document.querySelectorAll('#dispbar [data-split]').forEach(b=>b.onclick=()=>setSplit(+b.dataset.split));
$('wireoverlay') && ($('wireoverlay').onclick = toggleWireOverlay);
$('boxunfocused') && ($('boxunfocused').onclick = toggleBoxUnfocused);
// (Alt+1..4 set the display mode -- KEYMAP, C4. This used to ALSO fire the bare-digit mode switch.)

/* ================================ history & branches UI ================================ */
async function histRefresh(){
  if(!ACTIVE){ $('histlist').innerHTML='<span class="note">Select an object.</span>'; return; }
  const h = await (await fetch(`api/history?object=${encodeURIComponent(ACTIVE)}`)).json();
  if(h.error){ $('histlist').textContent=h.error; return; }
  const bsel=$('histbranch'); bsel.innerHTML='';
  Object.keys(h.branches).forEach(b=>{ const o=document.createElement('option'); o.value=b; o.textContent=b+(b===h.current?' (current)':''); if(b===h.current)o.selected=true; bsel.appendChild(o); });
  const msel=$('histmergefrom'); msel.innerHTML='';
  Object.keys(h.branches).filter(b=>b!==h.current).forEach(b=>{ const o=document.createElement('option'); o.value=b; o.textContent='merge from: '+b; msel.appendChild(o); });
  const ops=h.branches[h.current];
  let rows = [`<div data-i="0" class="histrow ${h.cursor===0?'cur':''}" style="cursor:pointer">◉ base${h.baked_from?' (baked after '+h.baked_from+')':''}</div>`];
  ops.forEach((op,i)=>rows.push(`<div data-i="${i+1}" class="histrow ${h.cursor===i+1?'cur':''}" style="cursor:pointer">${h.cursor===i+1?'▶':'&nbsp;'} ${i+1}. ${op}</div>`));
  $('histlist').innerHTML=rows.join('');
  $('histlist').querySelectorAll('.histrow').forEach(r=>r.onclick=async ()=>{
    const res=await (await fetch('api/history/op',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:'checkout',object:ACTIVE,index:+r.dataset.i})})).json();
    if(res.error){ $('histmeta').textContent=res.error; return; }
    applyResp(res); histRefresh();
    $('histmeta').textContent = res.failed && res.failed.length ? res.failed.length+' step(s) failed to replay' : '';
    status('History: step '+r.dataset.i);
  });
  $('histmeta').textContent = h.last_replay && h.last_replay.failed.length
    ? 'Last replay: '+h.last_replay.failed.length+' step(s) could not re-apply' : '';
}
$('histbranch') && ($('histbranch').onchange = async e=>{
  const res=await (await fetch('api/history/op',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'switch',object:ACTIVE,name:e.target.value.replace(' (current)','')})})).json();
  if(res.error){ $('histmeta').textContent=res.error; return; }
  applyResp(res); histRefresh(); status('Branch: '+e.target.value);
});
$('histnewbranch') && ($('histnewbranch').onclick = async ()=>{
  const name=await uiPrompt('New branch', 'Name (branches from the current history point):','variant');
  if(!name) return;
  const res=await (await fetch('api/history/op',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'branch',object:ACTIVE,name})})).json();
  if(res.error){ $('histmeta').textContent=res.error; return; }
  applyResp(res); histRefresh(); status('Branched: '+name+' — edits now land here.');
});
$('histmerge') && ($('histmerge').onclick = async ()=>{
  const from=$('histmergefrom').value; if(!from){ $('histmeta').textContent='No other branch to merge.'; return; }
  const res=await (await fetch('api/history/op',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action:'merge',object:ACTIVE,from})})).json();
  if(res.error){ $('histmeta').textContent=res.error; return; }
  applyResp(res); histRefresh();
  $('histmeta').textContent = `Merged ${res.merged_ops} op(s)` + (res.conflicts.length? ` — ${res.conflicts.length} conflict(s) skipped (shown honestly, not guessed)` : ', no conflicts');
  status('Merged '+from);
});
document.querySelectorAll('[data-dlg="dlg-history"]').forEach(b=>b.addEventListener('click', histRefresh));
// (H opens history -- KEYMAP, C4)

/* ================================ photo tools ================================ */
let PHOTO_B64 = null;
$('photofile') && ($('photofile').onchange = e=>{
  const f=e.target.files[0]; if(!f) return;
  const rd=new FileReader();
  rd.onload=()=>{ PHOTO_B64=rd.result; $('photoprev').src=rd.result; $('photoprev').style.display='block';
    ['pt_light','pt_depth','pt_texture','pt_shapes','pt_scene'].forEach(id=>$(id).disabled=false); };
  rd.readAsDataURL(f);
});
async function photoCall(path, extra){
  if(!PHOTO_B64){ $('pt_meta').textContent='Choose an image first.'; return null; }
  $('pt_meta').textContent='working…';
  try{
    const r=await fetch('api/photo/'+path, {method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(Object.assign({image:PHOTO_B64}, extra||{}))});
    const j=await r.json();
    if(j.error){ $('pt_meta').textContent='failed: '+j.error; return null; }
    return j;
  }catch(e){ $('pt_meta').textContent='failed: '+e.message; return null; }
}
$('pt_light') && ($('pt_light').onclick = async ()=>{
  const j=await photoCall('light'); if(!j) return;
  if($('lightaz')){ $('lightaz').value=Math.round(j.light_az); $('lightazval').textContent=Math.round(j.light_az)+'°';
    $('lightel').value=Math.round(j.light_el); $('lightelval').textContent=Math.round(j.light_el)+'°'; scheduleRender(); }
  $('pt_meta').textContent=`Light ≈ az ${j.light_az}°, el ${j.light_el}° — applied to Lighting. ${j.note}`;
});
$('pt_depth') && ($('pt_depth').onclick = async ()=>{
  const j=await photoCall('depth', {step:4, relief:0.5}); if(!j) return;
  applyResp(j); $('pt_meta').textContent=j.note||'Relief mesh added.'; status('Depth relief mesh added.');
});
$('pt_scene') && ($('pt_scene').onclick = async ()=>{
  if(!await uiConfirm('Bootstrap a full scene from this image? This clears the current scene.', 'Bootstrap')) return;
  const j=await photoCall('scene', {relief:0.6, step:5}); if(!j) return;
  applyResp(j);
  if(j.env_preview){                                          // set the generated backdrop in the viewport
    const img=new Image();
    img.onload=()=>{ const tex=new THREE.Texture(img); tex.needsUpdate=true; scene.background=tex; scheduleRender(); };
    img.src='data:image/png;base64,'+j.env_preview;
  }
  if(j.light && $('lightaz')){ $('lightaz').value=Math.round(j.light.az); $('lightazval').textContent=Math.round(j.light.az)+'°';
    $('lightel').value=Math.round(j.light.el); $('lightelval').textContent=Math.round(j.light.el)+'°'; }
  $('pt_meta').textContent=`Scene bootstrapped: terrain + palette backdrop + light (az ${j.light?j.light.az:'?'}°). ${j.note||''}`;
  status('Scene bootstrapped from image.');
});
$('pt_texture') && ($('pt_texture').onclick = async ()=>{
  const j=await photoCall('texture', {res:20}); if(!j) return;
  $('pt_glsl').style.display='block'; $('pt_glsl').value=j.glsl||'';
  $('pt_meta').textContent=`Match quality ${j.quality} (1=identical signature). ${j.note} — GLSL below; paste into a shader-material node.`;
});
$('pt_shapes') && ($('pt_shapes').onclick = async ()=>{
  const j=await photoCall('shapes', {k:5, step:6}); if(!j) return;
  if(j.object!==undefined){ applyResp(j); status('Fitted-shapes model added.'); }
  const kinds = j.kinds ? Object.entries(j.kinds).map(([k,n])=>`${n} ${k}`).join(', ') : '';
  $('pt_meta').textContent=`${j.n_primitives||0} primitives${kinds?' ('+kinds+')':''}, residual ${j.residual}. ${j.note}`;
});

/* post-processing control labels */
$('exposure') && ($('exposure').oninput = ()=>$('exposureval').textContent=(+$('exposure').value).toFixed(2));
$('sharpen') && ($('sharpen').oninput = ()=>$('sharpenval').textContent=(+$('sharpen').value).toFixed(2));

/* ================================ semantic command bar ================================ */
async function runCommand(){
  const cmd = $('cmdinput').value.trim(); if(!cmd) return;
  $('cmdmeta').textContent='interpreting…';
  try{
    const r=await fetch('api/semantic', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({command:cmd})});
    const j=await r.json();
    if(j.applied){ applyResp(j); $('cmdmeta').textContent='✓ '+(j.actions||[]).join('; '); status((j.actions||[]).join('; ')); $('cmdinput').value=''; }
    else {
      const sugg=(j.suggestions||[]).join(' ');
      const q=(j.questions||[]).join(' ');
      $('cmdmeta').textContent = (sugg||q) ? (sugg+' '+q).trim() : (j.note||'Not understood.');
    }
  }catch(e){ $('cmdmeta').textContent='failed: '+e.message; }
}
$('cmdrun') && ($('cmdrun').onclick = runCommand);
$('cmdinput') && ($('cmdinput').addEventListener('keydown', e=>{ if(e.key==='Enter') runCommand(); }));
$('cmddescribe') && ($('cmddescribe').onclick = async ()=>{
  try{ const j=await (await fetch('api/describe_scene')).json(); $('cmddesc').textContent=j.text||'empty scene'; }
  catch(e){ $('cmddesc').textContent='failed: '+e.message; }
});

/* ================================ retopo (field-guided, deformation-aware) ================================ */
$('tRetopo') && ($('tRetopo').onclick = ()=>{ if(!ACTIVE){ status('Select an object first.'); return; } openDlg('dlg-retopo'); });
$('retopomode') && ($('retopomode').onchange = e=>{ $('retopodefnote').style.display = e.target.value==='deformation'?'block':'none'; });
$('retopogo') && ($('retopogo').onclick = async ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  const mode = $('retopomode').value;
  $('retopometa').textContent='building cross field…';
  const body = {op:'retopo', object:ACTIVE, mode};
  if(mode==='deformation'){
    // use the object's current mesh as the deformation target vs its analytic/rest form isn't tracked here,
    // so we send the CURRENT vertices as the deformed pose against themselves — a no-op guide unless the user
    // has an explicit rest pose. Honest: without a captured rest pose this behaves like smoothest flow.
    const o = OBJS.get(ACTIVE);
    if(o){ const pos=o.mesh.geometry.attributes.position; const dv=[];
      for(let i=0;i<pos.count;i++) dv.push([pos.getX(i),pos.getY(i),pos.getZ(i)]);
      body.deformed_vertices = dv; }
  }
  try{
    const r = await api('op', body);
    applyResp(r);
    const f = r.object && r.object.counts ? r.object.counts.f : '?';
    $('retopometa').textContent = `Retopologized — ${f} faces. Check quality below.`;
    status('Retopologized ('+mode+').');
  }catch(e){ $('retopometa').textContent='failed'; }
});
$('retopoquality') && ($('retopoquality').onclick = async ()=>{
  if(!ACTIVE) return;
  $('retopometa').textContent='analysing field…';
  try{
    const r = await (await fetch(`api/field_report?object=${encodeURIComponent(ACTIVE)}`)).json();
    if(r.error){ $('retopometa').textContent=r.error; return; }
    $('retopometa').textContent = `${r.singularities} singularities · field energy ${r.energy} · ${r.consistent?'globally consistent ✓':'inconsistent field ✗'}. Fewer singularities + lower energy = cleaner flow.`;
  }catch(e){ $('retopometa').textContent='failed'; }
});

/* ================================ printability validation ================================ */
$('tValidate') && ($('tValidate').onclick = ()=>{ if(!ACTIVE){ status('Select an object first.'); return; } openDlg('dlg-validate'); });
$('validatego') && ($('validatego').onclick = async ()=>{
  if(!ACTIVE) return;
  $('validateresult').innerHTML='<span class="note">checking…</span>';
  try{
    const r = await (await fetch(`api/validate?object=${encodeURIComponent(ACTIVE)}&min_feature=${$('minfeat').value}`)).json();
    if(r.error){ $('validateresult').innerHTML='<span class="note">'+r.error+'</span>'; return; }
    const rows = r.checks.map(ch=>
      `<div style="display:flex;justify-content:space-between;gap:8px;padding:3px 0">
         <span>${ch.pass?'✓':'✗'} ${ch.name}</span>
         <span class="note" style="text-align:right">${ch.detail}</span></div>`).join('');
    const verdict = r.printable
      ? '<div style="color:#7fdca0;font-weight:500;margin-bottom:4px">Ready to slice ✓</div>'
      : '<div style="color:#f0a060;font-weight:500;margin-bottom:4px">Not print-ready — fix the failing gates</div>';
    $('validateresult').innerHTML = verdict + rows + '<div class="note" style="margin-top:6px">'+r.note+'</div>';
  }catch(e){ $('validateresult').innerHTML='<span class="note">failed: '+e.message+'</span>'; }
});
$('repairgo') && ($('repairgo').onclick = async ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  $('validateresult').innerHTML='<span class="note">repairing…</span>';
  try{
    const r = await (await fetch('api/repair',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({object:ACTIVE})})).json();
    if(r.error){ $('validateresult').innerHTML='<span class="note">'+r.error+'</span>'; return; }
    applyResp(r);
    const rp=r.repair||{};
    const msg = (rp.now_watertight?'Now watertight ✓':'Still has holes')+' · '+(rp.now_manifold?'manifold ✓':'non-manifold')+
      ' — '+(rp.before?rp.before.faces:'?')+' → '+(rp.after?rp.after.faces:'?')+' faces';
    $('validateresult').innerHTML='<div style="color:'+(rp.now_watertight?'#7fdca0':'#f0a060')+';font-weight:500">'+msg+'</div>'+
      '<div class="note" style="margin-top:4px">Re-run the check to confirm.</div>';
    status('Repair: '+msg);
  }catch(e){ $('validateresult').innerHTML='<span class="note">repair failed: '+e.message+'</span>'; }
});

/* ================================ data → geometry ================================ */
let DATA_CHANNELS = null;
function parseDataInput(){
  const raw = $('datainput').value.trim();
  if(!raw) return null;
  // CSV if it has commas AND newlines with consistent structure, else a flat number list
  if(raw.includes('\n') && raw.split('\n')[0].split(',').length>1){
    return {csv:raw, column:+$('datacol').value};
  }
  const nums = raw.split(/[\s,]+/).map(Number).filter(n=>!isNaN(n));
  return {series:nums};
}
$('dataanalyze') && ($('dataanalyze').onclick = async ()=>{
  const body = parseDataInput();
  if(!body){ $('datachannels').textContent='Paste some numbers first.'; return; }
  $('datachannels').textContent='analyzing…';
  try{
    const r = await (await fetch('api/data/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
    if(r.error){ $('datachannels').textContent=r.error; return; }
    DATA_CHANNELS = r.channels;
    const sel=$('datachan'); sel.innerHTML='';
    r.channels.forEach((ch,i)=>{ const o=document.createElement('option'); o.value=i; o.textContent=`channel ${i+1} (${ch.length} pts)`; sel.appendChild(o); });
    $('datachannels').textContent = r.n_channels>1
      ? `Found ${r.n_channels} interleaved channels (stride ${r.stride}, score ${r.interleave_score}). Pick one to build.`
      : `Single series, ${r.channels[0].length} points (interleave score ${r.interleave_score} ≈ baseline).`;
  }catch(e){ $('datachannels').textContent='failed: '+e.message; }
});
async function buildFromData(mode){
  let series;
  if(DATA_CHANNELS){ series = DATA_CHANNELS[+$('datachan').value] || DATA_CHANNELS[0]; }
  else { const b=parseDataInput(); if(!b){ $('datameta').textContent='Paste numbers first.'; return; }
    if(b.series) series=b.series; else { $('datameta').textContent='Analyze the CSV first.'; return; } }
  $('datameta').textContent='building…';
  try{
    const r = await (await fetch('api/data/to_geometry',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({series,mode})})).json();
    if(r.error){ $('datameta').textContent=r.error; return; }
    applyResp(r); $('datameta').textContent=r.note||'built'; status('Built geometry from data.');
  }catch(e){ $('datameta').textContent='failed: '+e.message; }
}
$('datalathe') && ($('datalathe').onclick = ()=>buildFromData('lathe'));
$('dataribbon') && ($('dataribbon').onclick = ()=>buildFromData('heightfield'));

/* ================================ halve (cut + cross-section) ================================ */
$('tHalve') && ($('tHalve').onclick = ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  // populate material dropdown from the library
  fetch('api/materials').then(r=>r.json()).then(ml=>{
    const names=[]; for(const cls in ml.classes) ml.classes[cls].forEach(m=>names.push(m.name));
    const sel=$('halvemat'); sel.innerHTML=names.map(n=>`<option>${n}</option>`).join('');
  });
  openDlg('dlg-halve');
});
$('halveoff') && ($('halveoff').oninput=()=>$('halveoffval').textContent=(+$('halveoff').value).toFixed(2));
$('halvego') && ($('halvego').onclick = async ()=>{
  if(!ACTIVE) return;
  $('halvemeta').textContent='cutting…';
  const body={op:'halve', object:ACTIVE, axis:$('halveaxis').value, offset:+$('halveoff').value,
              sign:$('halveflip').checked?-1:1, cut_material:$('halvemat').value};
  try{
    const r=await api('op', body);
    if(r.error){ $('halvemeta').textContent=r.error; return; }
    applyResp(r); $('halvemeta').textContent='Halved — cut face got '+$('halvemat').value; status('Halved.');
  }catch(e){ $('halvemeta').textContent='failed'; }
});

/* ================================ flute (ribbed relief) ================================ */
$('tFlute') && ($('tFlute').onclick = ()=>{ if(!ACTIVE){ status('Select an object first.'); return; } openDlg('dlg-flute'); });
$('flutelobes') && ($('flutelobes').oninput=()=>$('flutelobesval').textContent=$('flutelobes').value);
$('flutedepth') && ($('flutedepth').oninput=()=>$('flutedepthval').textContent=(+$('flutedepth').value).toFixed(2));
$('flutetwist') && ($('flutetwist').oninput=()=>$('flutetwistval').textContent=(+$('flutetwist').value).toFixed(1));
$('flutego') && ($('flutego').onclick = async ()=>{
  if(!ACTIVE) return;
  $('flutemeta').textContent='fluting…';
  try{
    const r=await api('op', {op:'flute', object:ACTIVE, lobes:+$('flutelobes').value, depth:+$('flutedepth').value, twist:+$('flutetwist').value});
    if(r.error){ $('flutemeta').textContent=r.error; return; }
    applyResp(r); $('flutemeta').textContent='Fluted with '+$('flutelobes').value+' ribs'; status('Fluted.');
  }catch(e){ $('flutemeta').textContent='failed'; }
});

/* ================================ scene save / load ================================ */
$('scenesave') && ($('scenesave').onclick = async ()=>{
  try{
    const j = await (await fetch('api/scene/save')).json();
    if(ANIM.tracks.size || ANIM.cam.length){                   // embed animation, tracks keyed by save order
      const order=[...OBJS.keys()];
      const tr={};
      for(const [id,keys] of ANIM.tracks){ const i=order.indexOf(id); if(i>=0&&keys.length) tr[i]=keys; }
      j.animation={fps:ANIM.fps, len:ANIM.len, tracks:tr, cam:ANIM.cam};
    }
    const blob = new Blob([JSON.stringify(j)], {type:'application/json'});
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'polystudio_scene.json'; a.click();
    status('Scene saved ('+j.objects.length+' objects).');
  }catch(e){ status('Save failed: '+e.message); }
});
$('sceneload') && ($('sceneload').onclick = ()=>{
  const inp = document.createElement('input'); inp.type='file'; inp.accept='.json,application/json';
  inp.onchange = async ()=>{
    const f = inp.files[0]; if(!f) return;
    if(OBJS && OBJS.size > 0 && !await uiConfirm('Opening a scene replaces everything currently in the viewport. Continue?', 'Open')) return;
    try{
      const scene = JSON.parse(await f.text());
      const r = await (await fetch('api/scene/load', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(scene)})).json();
      if(r.error){ status('Load failed: '+r.error); return; }
      applyResp(r); status('Scene loaded ('+r.loaded+' objects).');
      ANIM.tracks.clear(); ANIM.cam=[];
      if(scene.animation){                                     // remap tracks: save order == load order
        const order=[...OBJS.keys()];
        ANIM.fps=scene.animation.fps||30; ANIM.len=scene.animation.len||90;
        $('tlFps').value=ANIM.fps; $('tlLen').value=ANIM.len; $('tlScrub').max=ANIM.len;
        for(const [idx,keys] of Object.entries(scene.animation.tracks||{})){
          const id=order[+idx]; if(id) ANIM.tracks.set(id, keys);
        }
        ANIM.cam=scene.animation.cam||[];
        tlShow(true); status('Scene loaded with animation ('+r.loaded+' objects).');
      }
    }catch(e){ status('Load failed: '+e.message); }
  };
  inp.click();
});

/* ================================ sweep (tube along a path) ================================ */
const SWEEP_PRESETS = {
  handle: [[0,-0.25,0],[0.18,-0.1,0],[0.18,0.1,0],[0,0.25,0]],
  spout:  [[0,0,0],[0.12,0.08,0],[0.26,0.24,0]],
  stem:   [[0,0,0],[0.02,0.5,0],[-0.03,1.0,0]],
  hook:   [[0,0,0],[0,0.3,0],[0.12,0.42,0],[0.24,0.32,0]],
};
$('sweeppreset') && ($('sweeppreset').onchange = ()=>{
  const v=$('sweeppreset').value;
  $('sweeppath').style.display = v==='custom' ? 'block' : 'none';
  if(v==='spout'){ $('sweeptaper').value=0.4; $('sweeptaperval').textContent='0.40'; }
});
$('sweeprad') && ($('sweeprad').oninput=()=>$('sweepradval').textContent=(+$('sweeprad').value).toFixed(3));
$('sweeptaper') && ($('sweeptaper').oninput=()=>$('sweeptaperval').textContent=(+$('sweeptaper').value).toFixed(2));
$('sweepsides') && ($('sweepsides').oninput=()=>$('sweepsidesval').textContent=$('sweepsides').value);
$('sweepgo') && ($('sweepgo').onclick = async ()=>{
  const preset=$('sweeppreset').value;
  let path;
  if(preset==='custom'){
    try{ path=JSON.parse($('sweeppath').value); }catch(e){ $('sweepmeta').textContent='Bad JSON points.'; return; }
  } else path=SWEEP_PRESETS[preset];
  $('sweepmeta').textContent='sweeping…';
  try{
    const r=await (await fetch('api/sweep',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path, radius:+$('sweeprad').value, taper:+$('sweeptaper').value, sides:+$('sweepsides').value, name:preset[0].toUpperCase()+preset.slice(1)})})).json();
    if(r.error){ $('sweepmeta').textContent=r.error; return; }
    applyResp(r); $('sweepmeta').textContent='Swept — '+(r.note||''); status('Sweep created.');
  }catch(e){ $('sweepmeta').textContent='failed: '+e.message; }
});

/* first-run welcome dismissal */
$('welcomeclose') && ($('welcomeclose').onclick = ()=>{ if($('welcome')) $('welcome').style.display='none';
  try{ localStorage.setItem('polystudio:welcome','seen'); }catch(_){ } });

/* ================================ merge selected objects ================================ */
$('tMerge') && ($('tMerge').onclick = async ()=>{
  const ids = [...selObjs];
  if(ids.length < 2){ status('Select 2+ objects to merge (Shift-click in Object mode).'); return; }
  try{
    const r = await api('op', {op:'merge_objects', objects:ids});
    if(r.error){ status(r.error); return; }
    selObjs.clear(); if(r.object){ selObjs.add(r.object); ACTIVE=r.object; }
    applyResp(r); status('Merged '+(r.merged||ids.length)+' objects into one.');
  }catch(e){ status('Merge failed.'); }
});

/* ================================ band / rim (material stripe) ================================ */
$('tBand') && ($('tBand').onclick = ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  fetch('api/materials').then(r=>r.json()).then(ml=>{
    const names=[]; for(const cls in ml.classes) ml.classes[cls].forEach(m=>names.push(m.name));
    $('bandmat').innerHTML=names.map(n=>`<option>${n}</option>`).join('');
    // default to gold if present
    if(names.includes('gold')) $('bandmat').value='gold';
  });
  openDlg('dlg-band');
});
$('bandmode') && ($('bandmode').onchange=()=>{ $('bandmidrow').style.display = $('bandmode').value==='mid' ? 'block' : 'none'; });
$('bandw') && ($('bandw').oninput=()=>$('bandwval').textContent=(+$('bandw').value).toFixed(2));
$('bandc') && ($('bandc').oninput=()=>$('bandcval').textContent=(+$('bandc').value).toFixed(2));
$('bandgo') && ($('bandgo').onclick = async ()=>{
  if(!ACTIVE) return;
  const mode=$('bandmode').value, w=+$('bandw').value;
  let body={op:'band', object:ACTIVE, material:$('bandmat').value};
  if(mode==='top') body.rim='top', body.width=w;
  else if(mode==='bottom') body.rim='bottom', body.width=w;
  else { const c=+$('bandc').value; body.lo=Math.max(0,c-w/2); body.hi=Math.min(1,c+w/2); }
  $('bandmeta').textContent='applying…';
  try{
    const r=await api('op', body);
    if(r.error){ $('bandmeta').textContent=r.error; return; }
    applyResp(r); $('bandmeta').textContent='Banded '+(r.banded||0)+' faces with '+body.material; status('Band applied.');
  }catch(e){ $('bandmeta').textContent='failed'; }
});

/* ================================ draped cloth ================================ */
$('addCloth') && ($('addCloth').onclick = async ()=>{
  try{
    const r = await api('op', {op:'cloth', size:2.6, folds:4, material:'cotton', name:'Cloth'});
    if(r.error){ status(r.error); return; }
    if(r.object){ selObjs.clear(); selObjs.add(r.object); ACTIVE=r.object; }
    applyResp(r); status('Draped cloth added — drop objects onto it with Mesh ▸ transforms.');
  }catch(e){ status('Could not add cloth.'); }
});

/* ================================ shortcuts overlay + welcome recall ================================ */
$('shortcutsBtn') && ($('shortcutsBtn').onclick = ()=>openDlg('dlg-shortcuts'));
$('welcomeBtn') && ($('welcomeBtn').onclick = ()=>{ if($('welcome')) $('welcome').style.display='block'; });

/* ================================ reference image underlay (A0-1) ================================ */
$('refload') && ($('refload').onclick = ()=>{
  const inp=document.createElement('input'); inp.type='file'; inp.accept='image/*';
  inp.onchange=()=>{
    const f=inp.files[0]; if(!f) return;
    const url=URL.createObjectURL(f);
    const img=$('refunderlay');
    img.src=url; img.style.display='block';
    img.style.opacity=(+$('refopacity').value/100);
    $('refbar').style.display='flex';
    status('Reference loaded — match the camera by orbiting, then model over it. Adjust opacity below.');
  };
  inp.click();
});
$('refopacity') && ($('refopacity').oninput=()=>{ const img=$('refunderlay'); if(img) img.style.opacity=(+$('refopacity').value/100); });
$('refclear') && ($('refclear').onclick=()=>{
  const img=$('refunderlay'); if(img){ img.style.display='none'; img.src=''; }
  $('refbar').style.display='none'; status('Reference removed.');
});

/* ================================ taper ================================ */
$('tTaper') && ($('tTaper').onclick = ()=>{ if(!ACTIVE){ status('Select an object first.'); return; } openDlg('dlg-taper'); });
$('taperf') && ($('taperf').oninput=()=>$('taperfval').textContent=(+$('taperf').value).toFixed(2));
$('tapergo') && ($('tapergo').onclick = async ()=>{
  if(!ACTIVE) return;
  $('tapermeta').textContent='tapering…';
  try{
    const r=await api('op', {op:'taper', object:ACTIVE, axis:$('taperaxis').value, factor:+$('taperf').value});
    if(r.error){ $('tapermeta').textContent=r.error; return; }
    applyResp(r); $('tapermeta').textContent='Tapered.'; status('Tapered.');
  }catch(e){ $('tapermeta').textContent='failed'; }
});

/* ================================ material capture from reference (A3-2) ================================ */
let refSampling = false;
$('refsample') && ($('refsample').onclick = ()=>{
  const img=$('refunderlay');
  if(!img || img.style.display==='none'){ status('Load a reference image first.'); return; }
  refSampling = !refSampling;
  $('refsample').classList.toggle('on', refSampling);
  $('refunderlay').style.pointerEvents = refSampling ? 'auto' : 'none';
  $('refunderlay').style.cursor = refSampling ? 'crosshair' : '';
  status(refSampling ? 'Click the reference to sample a colour into a new material.' : 'Sampling off.');
});
$('refunderlay') && ($('refunderlay').addEventListener('click', async (ev)=>{
  if(!refSampling) return;
  const img=$('refunderlay');
  // map click to source pixel (object-fit:contain letterboxing)
  const r=img.getBoundingClientRect();
  const iw=img.naturalWidth, ih=img.naturalHeight;
  const scale=Math.min(r.width/iw, r.height/ih);
  const dw=iw*scale, dh=ih*scale;
  const ox=(r.width-dw)/2, oy=(r.height-dh)/2;
  const px=Math.floor((ev.clientX-r.left-ox)/scale);
  const py=Math.floor((ev.clientY-r.top-oy)/scale);
  if(px<0||py<0||px>=iw||py>=ih){ status('Click inside the image.'); return; }
  const cv=document.createElement('canvas'); cv.width=iw; cv.height=ih;
  const cx=cv.getContext('2d'); cx.drawImage(img,0,0);
  const d=cx.getImageData(px,py,1,1).data;
  const col=[+(d[0]/255).toFixed(3), +(d[1]/255).toFixed(3), +(d[2]/255).toFixed(3)];
  const name='sampled_'+Math.round(col[0]*255)+'_'+Math.round(col[1]*255)+'_'+Math.round(col[2]*255);
  try{
    await api('material/custom', {name, color:col, roughness:0.55});
    MATS = await api('materials'); buildMatTabs(); buildSwatches(); buildMatbar();
    status('Captured material "'+name+'" — rgb('+d[0]+','+d[1]+','+d[2]+'). Find it in the material list.');
  }catch(e){ status('Capture failed.'); }
}));

/* ================================ variant sweep ================================ */
$('varop') && ($('varop').onchange=()=>{
  if($('varop').value==='taper'){ $('varlo').value=0.3; $('varhi').value=1.2; }
  else { $('varlo').value=0; $('varhi').value=0.15; }
});
$('varcount') && ($('varcount').oninput=()=>$('varcval').textContent=$('varcount').value);
$('vargo') && ($('vargo').onclick = async ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  const vop=$('varop').value;
  const body={op:'variants', object:ACTIVE, vary_op:vop,
              param: vop==='flute'?'depth':'factor',
              lo:+$('varlo').value, hi:+$('varhi').value, count:+$('varcount').value};
  if(vop==='flute') body.base_params={lobes:16};
  $('varmeta').textContent='building…';
  try{
    const r=await api('op', body);
    if(r.error){ $('varmeta').textContent=r.error; return; }
    applyResp(r); $('varmeta').textContent='Built '+(r.count||0)+' variants in a row.'; status('Built '+(r.count||0)+' variants.');
  }catch(e){ $('varmeta').textContent='failed'; }
});

/* ================================ weathering (procedural material mask) ================================ */
$('tWeather') && ($('tWeather').onclick = ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  fetch('api/materials').then(r=>r.json()).then(ml=>{
    const names=[]; for(const cls in ml.classes) ml.classes[cls].forEach(m=>names.push(m.name));
    $('weathermat').innerHTML=names.map(n=>`<option>${n}</option>`).join('');
  });
  openDlg('dlg-weather');
});
$('weathera') && ($('weathera').oninput=()=>$('weatheraval').textContent=(+$('weathera').value).toFixed(2));
$('weathergo') && ($('weathergo').onclick = async ()=>{
  if(!ACTIVE) return;
  $('weathermeta').textContent='weathering…';
  try{
    const r=await api('op', {op:'weather', object:ACTIVE, material:$('weathermat').value, mask:$('weathermask').value, amount:+$('weathera').value});
    if(r.error){ $('weathermeta').textContent=r.error; return; }
    applyResp(r); $('weathermeta').textContent='Weathered '+(r.weathered||0)+' faces'; status('Weathering applied.');
  }catch(e){ $('weathermeta').textContent='failed'; }
});

/* ================================ CAD toolkit: extruded profile, array, measure ================================ */
$('exprofile') && ($('exprofile').onchange=()=>{ $('expoints').style.display = $('exprofile').value==='custom' ? 'block' : 'none'; });
$('exd') && ($('exd').oninput=()=>$('exdval').textContent=(+$('exd').value).toFixed(2));
$('exgo') && ($('exgo').onclick = async ()=>{
  const preset=$('exprofile').value;
  const body={preset, w:+$('exw').value, h:+$('exh').value, t:+$('ext').value, height:+$('exd').value};
  if(preset==='custom'){
    try{ body.points=JSON.parse($('expoints').value); }catch(e){ $('exmeta').textContent='Bad JSON points.'; return; }
  }
  $('exmeta').textContent='extruding…';
  try{
    const r=await (await fetch('api/extrude_profile',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
    if(r.error){ $('exmeta').textContent=r.error; return; }
    applyResp(r); $('exmeta').textContent='Created.'; status('Profile extruded.');
  }catch(e){ $('exmeta').textContent='failed'; }
});

$('tArray') && ($('tArray').onclick = ()=>{ if(!ACTIVE){ status('Select an object first.'); return; } openDlg('dlg-array'); });
$('arrkind') && ($('arrkind').onchange=()=>{
  const rad=$('arrkind').value==='radial';
  $('arrlinrow').style.display=rad?'none':'flex'; $('arrradrow').style.display=rad?'block':'none';
});
$('arrcount') && ($('arrcount').oninput=()=>$('arrcval').textContent=$('arrcount').value);
$('arrdeg') && ($('arrdeg').oninput=()=>$('arrdegval').textContent=$('arrdeg').value);
$('arrgo') && ($('arrgo').onclick = async ()=>{
  if(!ACTIVE) return;
  const kind=$('arrkind').value;
  const body={op:'array', object:ACTIVE, kind, count:+$('arrcount').value};
  if(kind==='linear') body.delta=[+$('arrdx').value,+$('arrdy').value,+$('arrdz').value];
  else { body.center=[+$('arrcx').value,0,+$('arrcz').value]; body.degrees=+$('arrdeg').value; }
  $('arrmeta').textContent='building…';
  try{
    const r=await api('op', body);
    if(r.error){ $('arrmeta').textContent=r.error; return; }
    applyResp(r); $('arrmeta').textContent='Built '+((r.arrayed||[]).length)+' copies.'; status('Array built.');
  }catch(e){ $('arrmeta').textContent='failed'; }
});


/* ================================ wall by two plan points ================================ */
$('wallgo') && ($('wallgo').onclick = async ()=>{
  const body={op:'wall', x1:+$('wx1').value, z1:+$('wz1').value, x2:+$('wx2').value, z2:+$('wz2').value,
              thickness:+$('wth').value, height:+$('whh').value};
  $('wallmeta').textContent='building…';
  try{
    const r=await api('op', body);
    if(r.error){ $('wallmeta').textContent=r.error; return; }
    applyResp(r);
    $('wallmeta').textContent='Wall added ('+(r.length||0).toFixed(2)+' long).'; status('Wall added.');
    if($('wchain').checked){ $('wx1').value=$('wx2').value; $('wz1').value=$('wz2').value; }
  }catch(e){ $('wallmeta').textContent='failed'; }
});

/* ================================ draft angle check ================================ */
$('tDraft') && ($('tDraft').onclick = ()=>{ if(!ACTIVE){ status('Select an object first.'); return; } openDlg('dlg-draft'); });
$('draftmin') && ($('draftmin').oninput=()=>$('draftminval').textContent=(+$('draftmin').value).toFixed(1));
$('draftgo') && ($('draftgo').onclick = async ()=>{
  if(!ACTIVE) return;
  $('draftmeta').textContent='checking…';
  try{
    const pull=$('draftpull').value.split(',').map(Number);
    const r=await api('op', {op:'draft_check', object:ACTIVE, pull, min_degrees:+$('draftmin').value});
    if(r.error){ $('draftmeta').textContent=r.error; return; }
    applyResp(r);
    const dd=r.draft||{};
    $('draftmeta').textContent = dd.flagged+' of '+dd.checked+' faces flagged (painted).';
    status('Draft check: '+dd.flagged+' problem faces.');
  }catch(e){ $('draftmeta').textContent='failed'; }
});

/* ================================ scene units ================================ */
$('tUnits') && ($('tUnits').onclick = async ()=>{
  try{
    const cur = await (await fetch('api/units')).json();
    const name = await uiPrompt('Scene units', 'Unit name (mm / cm / m / in):', cur.name||'cm'); if(name===null) return;
    const per = await uiPrompt('Scene units', 'How many '+name+' is 1 engine unit?', cur.per_unit||10); if(per===null) return;
    const r = await (await fetch('api/units',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name, per_unit:+per})})).json();
    if(r.error){ status(r.error); return; }
    status('Units: 1 engine unit = '+r.per_unit+' '+r.name+'. Measure now reports both.');
  }catch(e){ status('Units failed.'); }
});

/* ================================ surface graph (Substance-style surfacing) ================================ */
const SG_PRESETS = {
  bricks:    {gen:'bricks',   scale:3,  axis:'z', bal:0.5, inv:false, m1:'plaster', s1:0.05, m2:'clay'},
  checker:   {gen:'checker',  scale:6,  axis:'y', bal:0.5, inv:false, m1:'marble',  s1:0.5,  m2:'obsidian'},
  patina:    {gen:'perlin',   scale:6,  axis:'y', bal:0.6, inv:false, m1:'keep',    s1:0.6,  m2:'copper'},
  scratched: {gen:'scratches',scale:10, axis:'y', bal:0.5, inv:false, m1:'keep',    s1:0.5,  m2:'steel_brushed'},
  dirt:      {gen:'gradient', scale:4,  axis:'y', bal:0.5, inv:true,  m1:'keep',    s1:0.65, m2:'clay'},
};
function sgFillMats(){
  fetch('api/materials').then(r=>r.json()).then(ml=>{
    const names=['keep']; for(const cls in ml.classes) ml.classes[cls].forEach(m=>names.push(m.name));
    for(const id of ['sgmat1','sgmat2']){
      const cur=$(id).value;
      $(id).innerHTML=names.map(n=>`<option>${n}</option>`).join('');
      if(names.includes(cur)) $(id).value=cur;
    }
  });
}
$('tSurface') && ($('tSurface').onclick = ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  sgFillMats(); openDlg('dlg-surface');
});
$('tTess') && ($('tTess').onclick = ()=>{ if(ACTIVE) runOp('tessellate',{object:ACTIVE,levels:1}); });
$('sgpreset') && ($('sgpreset').onchange=()=>{
  const p=SG_PRESETS[$('sgpreset').value]; if(!p) return;
  $('sggen').value=p.gen; $('sgscale').value=p.scale; $('sgscaleval').textContent=p.scale;
  $('sgaxis').value=p.axis; $('sgbal').value=p.bal; $('sgbalval').textContent=(+p.bal).toFixed(2);
  $('sginv').checked=p.inv; $('sgstop1').value=p.s1;
  // set materials after the lists are (re)filled
  setTimeout(()=>{ $('sgmat1').value=p.m1; $('sgmat2').value=p.m2; }, 150);
});
$('sgscale') && ($('sgscale').oninput=()=>$('sgscaleval').textContent=$('sgscale').value);
$('sgbal') && ($('sgbal').oninput=()=>$('sgbalval').textContent=(+$('sgbal').value).toFixed(2));
$('sggo') && ($('sggo').onclick = async ()=>{
  if(!ACTIVE) return;
  $('sgmeta').textContent='surfacing…';
  const body={op:'surface_graph', object:ACTIVE, generator:$('sggen').value, scale:+$('sgscale').value,
              axis:$('sgaxis').value, seed:+$('sgseed').value, balance:+$('sgbal').value, invert:$('sginv').checked,
              ramp:[{material:$('sgmat1').value, upto:+$('sgstop1').value},
                    {material:$('sgmat2').value, upto:1.0}]};
  try{
    const r=await api('op', body);
    if(r.error){ $('sgmeta').textContent=r.error; return; }
    applyResp(r);
    const cc=r.surface&&r.surface.counts||{};
    $('sgmeta').textContent='Applied: '+Object.entries(cc).map(([k,v])=>k+' '+v).join(', ');
    status('Surface graph applied.');
  }catch(e){ $('sgmeta').textContent='failed'; }
});

/* ================================ viewport dimension annotations (F1-1) ================================ */
let DIMS_ON=false, DIMS_UNITS=null;
$('tDims') && ($('tDims').onclick = async ()=>{
  DIMS_ON=!DIMS_ON;
  if(DIMS_ON){ try{ DIMS_UNITS=await (await fetch('api/units')).json(); }catch(e){ DIMS_UNITS=null; } }
  $('dimtag').style.display=DIMS_ON?'block':'none';
  status(DIMS_ON?'Dimensions shown for the active object.':'Dimensions hidden.');
});
function updateDimTag(){
  if(!DIMS_ON) return;
  const o = activeObj(), tag=$('dimtag');
  if(!o || !o.d.bbox){ tag.style.display='none'; return; }
  const bb=o.d.bbox, sz=bb.size;
  let txt = sz.map(v=>v.toFixed(2)).join(' × ');
  if(DIMS_UNITS && DIMS_UNITS.per_unit)
    txt += '  ('+sz.map(v=>(v*DIMS_UNITS.per_unit).toFixed(1)).join('×')+' '+DIMS_UNITS.name+')';
  tag.textContent = txt;
  // project the bbox top-centre to screen
  const p = new THREE.Vector3(bb.center[0], bb.max[1], bb.center[2]).project(camera);
  if(p.z>1){ tag.style.display='none'; return; }             // behind the camera
  const r = renderer.domElement.getBoundingClientRect();
  const wr = $('dimtag').parentElement.getBoundingClientRect();
  tag.style.left = ((p.x*0.5+0.5)*r.width + (r.left-wr.left)) + 'px';
  tag.style.top  = ((-p.y*0.5+0.5)*r.height + (r.top-wr.top)) + 'px';
  tag.style.display='block';
}

/* ================================ floor plan import ================================ */
$('fpbuild') && ($('fpbuild').onclick = async ()=>{
  let path;
  try{ path=JSON.parse($('fppath').value); }catch(e){ $('fpbuildmeta').textContent='Bad JSON — need [[x,z],...]'; return; }
  $('fpbuildmeta').textContent='building…';
  try{
    const r=await (await fetch('api/floorplan',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({path, closed:$('fpclosed').checked, thickness:+$('fpth').value, height:+$('fphh').value})})).json();
    if(r.error){ $('fpbuildmeta').textContent=r.error; return; }
    applyResp(r); $('fpbuildmeta').textContent='Built '+(r.walls||[]).length+' walls.'; status('Floor plan built.');
  }catch(e){ $('fpbuildmeta').textContent='failed'; }
});

/* ================================ timeline + keyframes (P2-3, view-layer v1) ================================
   Keys animate three.js GROUP transforms + the spherical camera. The server geometry stays at rest pose:
   playback, scrubbing and webm recording are pure client; GI photos honour the scrubbed CAMERA automatically
   (the photo endpoint reads the live camera), while object pose keys are view-only — stated in the dialog. */
const ANIM = { fps:30, len:90, t:0, playing:false, tracks:new Map(), cam:[] };
let _tlLast = 0;

function tlShow(on){ $('timeline').style.display = on?'flex':'none'; }
$('tlToggle') && ($('tlToggle').onclick = ()=>tlShow($('timeline').style.display==='none'));
$('tlClose') && ($('tlClose').onclick = ()=>{ ANIM.playing=false; $('tlPlay').textContent='▶'; tlShow(false); });
$('tlLen') && ($('tlLen').onchange = ()=>{ ANIM.len=Math.max(10,+$('tlLen').value|0); $('tlScrub').max=ANIM.len; });
$('tlFps') && ($('tlFps').onchange = ()=>{ ANIM.fps=Math.max(6,+$('tlFps').value|0); });
$('tlScrub') && ($('tlScrub').oninput = ()=>{ ANIM.t=+$('tlScrub').value; ANIM.playing=false; $('tlPlay').textContent='▶'; animApply(); });
$('tlPlay') && ($('tlPlay').onclick = ()=>{ ANIM.playing=!ANIM.playing; _tlLast=performance.now(); $('tlPlay').textContent=ANIM.playing?'⏸':'▶'; });
$('tlStop') && ($('tlStop').onclick = ()=>{ ANIM.playing=false; ANIM.t=0; $('tlPlay').textContent='▶'; animApply(); });
$('tlReset') && ($('tlReset').onclick = ()=>{
  ANIM.playing=false; $('tlPlay').textContent='▶';
  for(const [,o] of OBJS){ o.group.position.set(0,0,0); o.group.rotation.set(0,0,0); o.group.scale.set(1,1,1); }
  scheduleRender(); status('Poses reset to rest.');
});

function keysFor(id){ if(!ANIM.tracks.has(id)) ANIM.tracks.set(id,[]); return ANIM.tracks.get(id); }
function putKey(arr, k){ const i=arr.findIndex(e=>e.f===k.f); if(i>=0) arr[i]=k; else { arr.push(k); arr.sort((a,b)=>a.f-b.f); } }
function lerp(a,b,u){ return a+(b-a)*u; }
function lerp3(a,b,u){ return [lerp(a[0],b[0],u),lerp(a[1],b[1],u),lerp(a[2],b[2],u)]; }
function evalKeys(arr, f){
  if(!arr.length) return null;
  if(f<=arr[0].f) return arr[0];
  if(f>=arr[arr.length-1].f) return arr[arr.length-1];
  let i=1; while(arr[i].f<f) i++;
  const a=arr[i-1], b=arr[i], u=(f-a.f)/(b.f-a.f), out={};
  for(const key of ['pos','rot']) if(a[key]&&b[key]) out[key]=lerp3(a[key],b[key],u);
  if(a.scl!==undefined) out.scl=lerp(a.scl,b.scl,u);
  for(const key of ['theta','phi','dist']) if(a[key]!==undefined) out[key]=lerp(a[key],b[key],u);
  if(a.target&&b.target) out.target=lerp3(a.target,b.target,u);
  return out;
}
function animApply(){
  $('tlFrame').textContent = ANIM.t|0; $('tlScrub').value = ANIM.t|0;
  for(const [id,keys] of ANIM.tracks){
    const o=OBJS.get(id); if(!o||!keys.length) continue;
    const k=evalKeys(keys, ANIM.t);
    if(k.pos) o.group.position.set(...k.pos);
    if(k.rot) o.group.rotation.set(k.rot[0]*Math.PI/180, k.rot[1]*Math.PI/180, k.rot[2]*Math.PI/180);
    if(k.scl!==undefined) o.group.scale.setScalar(k.scl);
  }
  if(ANIM.cam.length){
    const k=evalKeys(ANIM.cam, ANIM.t);
    if(k.theta!==undefined){ camTheta=k.theta; camPhi=k.phi; camDist=k.dist; camTarget.set(...k.target); }
  }
  scheduleRender();
}
function animTick(){
  if(!ANIM.playing) return;
  const now=performance.now(), dt=(now-_tlLast)/1000; _tlLast=now;
  ANIM.t += dt*ANIM.fps;
  if(ANIM.t>ANIM.len) ANIM.t=0;                 // loop
  animApply();
}

$('tlKeyObj') && ($('tlKeyObj').onclick = ()=>{
  const o=activeObj(); if(!o){ status('Select an object first.'); return; }
  const g=o.group;
  putKey(keysFor(ACTIVE), {f:ANIM.t|0, pos:[g.position.x,g.position.y,g.position.z],
    rot:[g.rotation.x,g.rotation.y,g.rotation.z].map(r=>r*180/Math.PI), scl:g.scale.x});
  refreshKeyList(); status('Object key @ '+(ANIM.t|0)+'.');
});
$('tlKeyCam') && ($('tlKeyCam').onclick = ()=>{
  putKey(ANIM.cam, {f:ANIM.t|0, theta:camTheta, phi:camPhi, dist:camDist, target:[camTarget.x,camTarget.y,camTarget.z]});
  refreshKeyList(); status('Camera key @ '+(ANIM.t|0)+'.');
});
$('tlEdit') && ($('tlEdit').onclick = ()=>{ refreshKeyList(); openDlg('dlg-keys'); });
$('kApply') && ($('kApply').onclick = ()=>{
  const o=activeObj(); if(!o) return;
  o.group.position.set(+$('kpx').value,+$('kpy').value,+$('kpz').value);
  o.group.rotation.set(+$('krx').value*Math.PI/180,+$('kry').value*Math.PI/180,+$('krz').value*Math.PI/180);
  o.group.scale.setScalar(+$('ksc').value);
  scheduleRender();
});
$('kKey') && ($('kKey').onclick = ()=>{ $('kApply').onclick(); $('tlKeyObj').onclick(); });
$('kSpin') && ($('kSpin').onclick = ()=>{
  ANIM.cam=[{f:0, theta:camTheta, phi:camPhi, dist:camDist, target:[camTarget.x,camTarget.y,camTarget.z]},
            {f:ANIM.len, theta:camTheta+Math.PI*2, phi:camPhi, dist:camDist, target:[camTarget.x,camTarget.y,camTarget.z]}];
  refreshKeyList(); $('kmeta').textContent='Turntable set: one full orbit over '+ANIM.len+' frames.';
});
function refreshKeyList(){
  const el=$('keylist'); if(!el) return;
  let html='';
  const okeys=ACTIVE?keysFor(ACTIVE):[];
  for(const [label,arr] of [['obj',okeys],['cam',ANIM.cam]])
    for(let i=0;i<arr.length;i++)
      html+=`<div class="kv"><span>${label} ◆ f${arr[i].f}</span><span><a href="#" data-jump="${arr[i].f}">go</a> · <a href="#" data-del="${label}:${i}">✕</a></span></div>`;
  el.innerHTML = html || '<div class="note">No keys yet.</div>';
  el.querySelectorAll('[data-jump]').forEach(a=>a.onclick=(e)=>{ e.preventDefault(); ANIM.t=+a.dataset.jump; animApply(); });
  el.querySelectorAll('[data-del]').forEach(a=>a.onclick=(e)=>{
    e.preventDefault(); const [lb,i]=a.dataset.del.split(':');
    (lb==='cam'?ANIM.cam:keysFor(ACTIVE)).splice(+i,1); refreshKeyList(); animApply();
  });
}

/* record one loop to webm via MediaRecorder (browser API — noted as untestable headless) */
let _rec=null;
$('tlRec') && ($('tlRec').onclick = ()=>{
  if(_rec){ _rec.stop(); return; }
  if(!canvas.captureStream || !window.MediaRecorder){ status('Recording needs a browser with MediaRecorder.'); return; }
  const stream=canvas.captureStream(ANIM.fps);
  const chunks=[];
  _rec=new MediaRecorder(stream,{mimeType: MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm'});
  _rec.ondataavailable=e=>{ if(e.data.size) chunks.push(e.data); };
  _rec.onstop=()=>{
    const a=document.createElement('a');
    a.href=URL.createObjectURL(new Blob(chunks,{type:'video/webm'}));
    a.download='polystudio_anim.webm'; a.click();
    _rec=null; $('tlRec').classList.remove('on'); status('Recording saved (.webm).');
  };
  ANIM.t=0; ANIM.playing=true; _tlLast=performance.now(); $('tlPlay').textContent='⏸';
  _rec.start(); $('tlRec').classList.add('on');
  const loopMs=(ANIM.len/ANIM.fps)*1000;
  setTimeout(()=>{ if(_rec) _rec.stop(); ANIM.playing=false; $('tlPlay').textContent='▶'; }, loopMs+150);
  status('Recording one loop ('+Math.round(loopMs/1000)+'s)…');
});

/* ================================ procedural generators (Generate menu) ================================ */
const GEN_SPECS = {
  landscape: {title:'Landscape', note:'A fractal terrain with biome materials. Scatter grass/trees on it afterwards.',
    params:[{id:'biome',label:'Biome',type:'select',options:['temperate','desert','arctic','volcanic']},
            {id:'size',label:'Size',type:'range',min:2,max:12,step:0.5,val:4},
            {id:'relief',label:'Relief',type:'range',min:0.1,max:2,step:0.05,val:0.6},
            {id:'sea_level',label:'Sea level',type:'range',min:0,max:0.8,step:0.05,val:0.35},
            {id:'res',label:'Detail',type:'range',min:40,max:110,step:2,val:72}]},
  planet: {title:'Planet', note:'A displaced sphere with elevation biomes and polar caps.',
    params:[{id:'biome',label:'Biome',type:'select',options:['temperate','desert','arctic','volcanic']},
            {id:'radius',label:'Radius',type:'range',min:0.2,max:2.5,step:0.05,val:0.8},
            {id:'relief',label:'Relief',type:'range',min:0,max:0.4,step:0.02,val:0.1},
            {id:'sea_level',label:'Sea level',type:'range',min:0.1,max:0.9,step:0.05,val:0.45},
            {id:'res',label:'Detail',type:'range',min:28,max:80,step:2,val:56}]},
  ocean: {title:'Ocean / lake', note:'Waves on a new water plane — or on the SELECTED surface (its verts are displaced + painted water).',
    params:[{id:'use_selected',label:'Apply to selected object',type:'check',val:false},
            {id:'size',label:'Size (new plane)',type:'range',min:1,max:12,step:0.5,val:4},
            {id:'amplitude',label:'Wave height',type:'range',min:0,max:0.3,step:0.01,val:0.06},
            {id:'wavelength',label:'Wavelength',type:'range',min:0.2,max:3,step:0.1,val:0.8},
            {id:'harmonics',label:'Chop (harmonics)',type:'range',min:1,max:8,step:1,val:4}]},
  clouds: {title:'Clouds', note:'Puffy blob clouds (mesh). Turn ON fog in the photo panel for atmosphere — the renderer has no true volumetrics.',
    params:[{id:'puffs',label:'Puffs',type:'range',min:2,max:18,step:1,val:7},
            {id:'size',label:'Puff size',type:'range',min:0.15,max:1.5,step:0.05,val:0.5},
            {id:'spread',label:'Spread',type:'range',min:0.5,max:5,step:0.1,val:1.4},
            {id:'softness',label:'Softness',type:'range',min:0.08,max:0.5,step:0.02,val:0.22}]},
  scatter: {title:'Grass · trees · rocks', note:'Scatters onto the up-facing area of the ACTIVE object (select the landscape first). One merged mesh.',
    needsObject:true,
    params:[{id:'what',label:'Kind',type:'select',options:['grass','trees','rocks']},
            {id:'count',label:'Count',type:'range',min:20,max:2000,step:20,val:400},
            {id:'size',label:'Size',type:'range',min:0.02,max:0.6,step:0.01,val:0.1},
            {id:'up_only',label:'Up-facing only',type:'check',val:true}]},
  tree: {title:'Tree', note:'Recursive branching trunk with leaf cards. Higher depth = more branches; every seed differs.',
    params:[{id:'depth',label:'Depth',type:'range',min:2,max:7,step:1,val:5},
            {id:'height',label:'Trunk length',type:'range',min:0.5,max:4,step:0.1,val:1.6},
            {id:'splits',label:'Splits/branch',type:'range',min:1,max:4,step:1,val:2},
            {id:'spread',label:'Spread',type:'range',min:0.2,max:1.3,step:0.05,val:0.6},
            {id:'leaves',label:'Leaves',type:'check',val:true},
            {id:'leaf_size',label:'Leaf size',type:'range',min:0.05,max:0.5,step:0.02,val:0.16}]},
  creature: {title:'Creature', note:'Spore-style: a smooth-welded body + head, tube legs, eyes, and optional arms / tail / antennae / spikes. Every seed is a different beast.',
    params:[{id:'legs',label:'Legs',type:'select',options:['0','2','4','6']},
            {id:'plump',label:'Plumpness',type:'range',min:0.5,max:1.8,step:0.05,val:1.0},
            {id:'arms',label:'Arms',type:'check',val:false},
            {id:'tail',label:'Tail',type:'check',val:false},
            {id:'antennae',label:'Antennae',type:'check',val:false},
            {id:'spikes',label:'Spikes',type:'check',val:true}]},
  star: {title:'Star', note:'An emissive sphere — it genuinely lights the scene in GI photos.',
    params:[{id:'temperature',label:'Colour',type:'select',options:['red','orange','yellow','white','blue']},
            {id:'radius',label:'Radius',type:'range',min:0.1,max:2,step:0.05,val:0.6},
            {id:'emission',label:'Brightness',type:'range',min:1,max:15,step:0.5,val:6}]},
  solar_system: {title:'Solar system', note:'An emissive sun + planets in a row (one gets rings). Orbit them with camera keys on the timeline.',
    params:[{id:'planets',label:'Planets',type:'range',min:1,max:8,step:1,val:4}]},
  env: {title:'Sky & space backdrop', note:'A procedural environment image: it becomes the render sky AND the GI dome light, and shows in the viewport.',
    params:[{id:'preset',label:'Preset',type:'select',options:['day','sunset','night','starfield','nebula','galaxy']},
            {id:'palette',label:'Nebula palette',type:'select',options:['purple','teal','fire']},
            {id:'arms',label:'Galaxy arms',type:'range',min:1,max:5,step:1,val:2},
            {id:'quality',label:'Quality',type:'select',options:['dome','high']}]},
};
let GEN_KIND='landscape';
function genOpen(kind){
  GEN_KIND=kind;
  const spec=GEN_SPECS[kind];
  $('genTitle').textContent='Generate — '+spec.title;
  $('genNote').textContent=spec.note;
  let html='';
  for(const p of spec.params){
    if(p.type==='select')
      html+=`<div class="inline">${p.label} <select id="gp_${p.id}" style="flex:1">${p.options.map(o=>`<option>${o}</option>`).join('')}</select></div>`;
    else if(p.type==='check')
      html+=`<label class="inline"><input type="checkbox" id="gp_${p.id}" ${p.val?'checked':''}> ${p.label}</label>`;
    else
      html+=`<div class="kv">${p.label} <b id="gpv_${p.id}">${p.val}</b></div>`+
            `<input type="range" id="gp_${p.id}" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.val}">`;
  }
  $('genParams').innerHTML=html;
  for(const p of spec.params) if(p.type==='range'){
    const el=$('gp_'+p.id); el.oninput=()=>$('gpv_'+p.id).textContent=el.value;
  }
  // supershape: preset dropdown fills the eight superformula sliders
  if(kind==='supershape' && $('gp_preset')){
    const applyPreset=()=>{
      const pv=SUPERSHAPE_PRESETS[$('gp_preset').value]; if(!pv) return;
      for(const k in pv){ const el=$('gp_'+k); if(el){ el.value=pv[k]; const lbl=$('gpv_'+k); if(lbl) lbl.textContent=pv[k]; } }
    };
    $('gp_preset').onchange=applyPreset; applyPreset();
  }
  $('genMeta').textContent='';
  openDlg('dlg-generate');
}
document.querySelectorAll('.gen-item').forEach(b=>b.onclick=()=>genOpen(b.dataset.gen));
$('genShuffle') && ($('genShuffle').onclick=()=>{ $('genSeed').value=Math.floor(Math.random()*99999); });
$('genGo') && ($('genGo').onclick = async ()=>{
  const spec=GEN_SPECS[GEN_KIND];
  if(spec.needsObject && !ACTIVE){ $('genMeta').textContent='Select a target object first.'; return; }
  const body={seed:+$('genSeed').value};
  for(const p of spec.params){
    const el=$('gp_'+p.id);
    body[p.id]= p.type==='check' ? el.checked : (p.type==='select' ? el.value : +el.value);
  }
  if(GEN_KIND==='creature') body.legs=+body.legs;
  if(GEN_KIND==='scatter') body.object=ACTIVE;
  if(GEN_KIND==='ocean' && body.use_selected){ if(!ACTIVE){ $('genMeta').textContent='Select a surface first.'; return; } body.object=ACTIVE; }
  delete body.use_selected;
  $('genMeta').textContent='generating…';
  try{
    if(GEN_KIND==='env'){
      const r=await (await fetch('api/env',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
      if(r.error){ $('genMeta').textContent=r.error; return; }
      const img=new Image();
      img.onload=()=>{ const tex=new THREE.Texture(img); tex.needsUpdate=true; scene.background=tex; scheduleRender(); };
      img.src='data:image/png;base64,'+r.preview;
      $('genMeta').innerHTML='Backdrop set — it now lights GI photos too. <a class="dl" id="envdl" href="data:image/png;base64,'+r.preview+'" download="polystudio_env.png">download PNG</a>';
      status('Backdrop: '+body.preset+'.');
    }else{
      body.kind=GEN_KIND;
      const r=await (await fetch('api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
      if(r.error){ $('genMeta').textContent=r.error; return; }
      applyResp(r);
      $('genMeta').textContent='Generated.'+(r.note?' '+r.note:'');
      status(spec.title+' generated.');
    }
  }catch(e){ $('genMeta').textContent='failed'; }
});
$('envClear') && ($('envClear').onclick = async ()=>{
  try{ await fetch('api/env',{method:'DELETE'}); }catch(e){}
  scene.background=new THREE.Color(0x10131a); scheduleRender(); status('Backdrop cleared.');
});

/* ================================ geodesic grow / fitprims / lattice ================================ */
$('selGeo') && ($('selGeo').onclick = ()=>{
  if(!ACTIVE){ status('Select an object first.'); return; }
  if(MODE!=='vertex' && MODE!=='face'){ status('Switch to Vertex or Face mode and pick seed vertices first.'); return; }
  if(!selectionVerts().length){ status('Pick at least one seed vertex first.'); return; }
  openDlg('dlg-geodesic');
});
$('geoR') && ($('geoR').oninput=()=>$('geoRval').textContent=(+$('geoR').value).toFixed(2));
$('geogo') && ($('geogo').onclick = async ()=>{
  const seeds=selectionVerts(); if(!seeds.length){ $('geometa').textContent='No seed vertices.'; return; }
  $('geometa').textContent='growing…';
  try{
    const r=await api('select/geodesic', {object:ACTIVE, seeds, radius:+$('geoR').value});
    if(r.error){ $('geometa').textContent=r.error; return; }
    selVerts.clear(); r.indices.forEach(i=>selVerts.add(i));
    softW = new Float32Array(r.weights);
    refreshSelectionVisuals();
    $('geometa').textContent=r.count+' vertices within '+r.radius+' along the surface.';
    status('Selection grown to '+r.count+' vertices (soft weights set).');
  }catch(e){ $('geometa').textContent='failed'; }
});

$('fpk') && ($('fpk').oninput=()=>$('fpkval').textContent=$('fpk').value);
$('fpgo') && ($('fpgo').onclick = async ()=>{
  if(!ACTIVE){ $('fpmeta').textContent='Select an object first.'; return; }
  $('fpmeta').textContent='fitting…';
  try{
    const r=await api('fitprims', {object:ACTIVE, k:+$('fpk').value, adopt:$('fpadopt').checked});
    if(r.error){ $('fpmeta').textContent=r.error; return; }
    const kinds=Object.entries(r.kinds||{}).map(([k,v])=>v+' '+k+(v>1?'s':'')).join(', ');
    let msg=(kinds||'no parts')+' · residual '+(r.residual_relative*100).toFixed(2)+'% of size';
    if(r.adopted===true) msg+=' — tree ADOPTED (exact ops restored).';
    else if(r.adopted===false) msg+=' — not adopted ('+(r.note||'over tolerance')+').';
    $('fpmeta').textContent=msg; status('Fit: '+msg);
  }catch(e){ $('fpmeta').textContent='failed'; }
});

$('latamt') && ($('latamt').oninput=()=>$('latamtval').textContent=(+$('latamt').value).toFixed(2));
$('latgo') && ($('latgo').onclick = async ()=>{
  if(!ACTIVE){ $('latmeta').textContent='Select an object first.'; return; }
  $('latmeta').textContent='deforming…';
  try{
    const r=await api('op', {op:'lattice', object:ACTIVE, preset:$('latpreset').value, amount:+$('latamt').value});
    if(r.error){ $('latmeta').textContent=r.error; return; }
    applyResp(r); $('latmeta').textContent='Deformed.'; status('Lattice deform applied.');
  }catch(e){ $('latmeta').textContent='failed'; }
});

/* new generator specs */
GEN_SPECS.moon = {title:'Moon', note:'A cratered body — bowls with raised rims over gentle fBm.',
  params:[{id:'radius',label:'Radius',type:'range',min:0.1,max:1.5,step:0.05,val:0.4},
          {id:'craters',label:'Craters',type:'range',min:0,max:40,step:1,val:14},
          {id:'res',label:'Detail',type:'range',min:28,max:72,step:2,val:44}]};
GEN_SPECS.asteroids = {title:'Asteroid belt', note:'A ring of squashed rocky bodies (one merged mesh). Pairs with the solar system.',
  params:[{id:'count',label:'Count',type:'range',min:20,max:400,step:10,val:120},
          {id:'radius',label:'Ring radius',type:'range',min:0.5,max:8,step:0.1,val:2.2},
          {id:'width',label:'Ring width',type:'range',min:0.1,max:2,step:0.05,val:0.5},
          {id:'size',label:'Rock size',type:'range',min:0.01,max:0.3,step:0.005,val:0.05}]};
GEN_SPECS.galaxy_field = {title:'Galaxy (geometric)', note:'A real fly-through spiral of emissive star billboards (merged mesh). Pair with a starfield backdrop. Very high counts await GPU instancing.',
  params:[{id:'count',label:'Stars',type:'range',min:200,max:4000,step:100,val:1500},
          {id:'arms',label:'Arms',type:'range',min:1,max:6,step:1,val:3},
          {id:'radius',label:'Radius',type:'range',min:1,max:8,step:0.5,val:3},
          {id:'twist',label:'Arm twist',type:'range',min:0.5,max:5,step:0.1,val:2.6},
          {id:'thickness',label:'Disk thickness',type:'range',min:0.02,max:1,step:0.02,val:0.12},
          {id:'star_size',label:'Star size',type:'range',min:0.008,max:0.08,step:0.004,val:0.02}]};

GEN_SPECS.supershape = {title:'Supershape', note:'The Gielis superformula as a 3-D family — one set of numbers sweeps from spheres to boxes to stars to flowers. Pick a preset, or dial the symmetry (m) and the three exponents on each axis.',
  params:[{id:'preset',label:'Preset',type:'select',options:['sphere','rounded box','star','flower','diatom','twisted']},
          {id:'m1',label:'Symmetry m (lat)',type:'range',min:0,max:14,step:1,val:6},
          {id:'m2',label:'Symmetry m (lon)',type:'range',min:0,max:14,step:1,val:6},
          {id:'n11',label:'Lat exponent 1',type:'range',min:0.1,max:12,step:0.1,val:1},
          {id:'n12',label:'Lat exponent 2',type:'range',min:0.1,max:12,step:0.1,val:1},
          {id:'n13',label:'Lat exponent 3',type:'range',min:0.1,max:12,step:0.1,val:1},
          {id:'n21',label:'Lon exponent 1',type:'range',min:0.1,max:12,step:0.1,val:1},
          {id:'n22',label:'Lon exponent 2',type:'range',min:0.1,max:12,step:0.1,val:1},
          {id:'n23',label:'Lon exponent 3',type:'range',min:0.1,max:12,step:0.1,val:1},
          {id:'scale',label:'Scale',type:'range',min:0.2,max:3,step:0.1,val:0.8},
          {id:'res',label:'Resolution',type:'range',min:32,max:120,step:8,val:72}]};
const SUPERSHAPE_PRESETS = {
  'sphere':      {m1:0,m2:0,n11:1,n12:1,n13:1,n21:1,n22:1,n23:1},
  'rounded box': {m1:4,m2:4,n11:8,n12:8,n13:8,n21:8,n22:8,n23:8},
  'star':        {m1:6,m2:6,n11:0.3,n12:0.3,n13:0.3,n21:0.3,n22:0.3,n23:0.3},
  'flower':      {m1:6,m2:6,n11:1,n12:1,n13:1,n21:0.4,n22:1.7,n23:1.7},
  'diatom':      {m1:8,m2:8,n11:0.5,n12:0.5,n13:8,n21:0.5,n22:0.5,n23:8},
  'twisted':     {m1:3,m2:7,n11:0.6,n12:1.2,n13:1.2,n21:0.6,n22:1.2,n23:1.2}
};

/* ================================ gaussian curvature + developability heatmaps ================================ */
async function paintHeatmap(endpoint, label){
  if(CURV){ // reuse the curvature toggle-off path
    const o=OBJS.get(CURV.oid);
    if(o && o.mesh.geometry.attributes.color.array.length===CURV.orig.length){
      o.mesh.geometry.attributes.color.array.set(CURV.orig); o.mesh.geometry.attributes.color.needsUpdate=true;
    }
    CURV=null; $('gaussbtn').classList.remove('on'); $('devbtn').classList.remove('on'); $('curvbtn').classList.remove('on');
    return false;
  }
  if(!ACTIVE) return false;
  const r = await api_get(endpoint+`&object=${ACTIVE}`);
  if(r.error){ status(r.error); return false; }
  const o=OBJS.get(ACTIVE); if(!o) return false;
  const attr=o.mesh.geometry.attributes.color;
  CURV={oid:ACTIVE, orig:Float32Array.from(attr.array)};
  const s=r.scale||1;
  const isDev = endpoint.indexOf('developable')>=0;
  for(let i=0;i<r.values.length && i<attr.count;i++){
    let t = r.values[i]/s;                         // gaussian: signed; developable: 0..1 hot
    if(isDev){ t=Math.min(1,Math.max(0,t)); attr.setXYZ(i, t, 0.5*(1-t)+0.2, 0.9*(1-t)); }
    else { t=Math.max(-1,Math.min(1,t)); if(t>=0) attr.setXYZ(i,1,1-t*0.8,1-t); else attr.setXYZ(i,1+t,1+t*0.6,1); }
  }
  attr.needsUpdate=true;
  if(r.developable_fraction!==undefined) status(label+' · '+Math.round(r.developable_fraction*100)+'% unrolls flat. '+(r.note||''));
  else status(label+'. '+(r.note||''));
  return true;
}
$('gaussbtn') && ($('gaussbtn').onclick = async ()=>{
  const on=await paintHeatmap('analyze_surface?metric=gaussian','Gaussian curvature');
  $('gaussbtn').classList.toggle('on', on);
});
$('devbtn') && ($('devbtn').onclick = async ()=>{
  const on=await paintHeatmap('analyze_surface?metric=developable','Developability');
  $('devbtn').classList.toggle('on', on);
});

/* ================================ bevel edges / corners (chamfer · fillet) ================================ */
$('bvratio') && ($('bvratio').oninput=()=>$('bvratioval').textContent=(+$('bvratio').value).toFixed(2));
$('bvseg') && ($('bvseg').oninput=()=>$('bvsegval').textContent=$('bvseg').value);
$('bvmode') && ($('bvmode').onchange=()=>{ $('bvsegrow').style.display = $('bvmode').value==='fillet' ? 'block':'none'; });
if($('bvsegrow')) $('bvsegrow').style.display='none';
$('bvgo') && ($('bvgo').onclick = async ()=>{
  if(!ACTIVE){ $('bvmeta').textContent='Select an object first.'; return; }
  const sel = (typeof selectionVerts==='function' && MODE==='vertex') ? selectionVerts() : [];
  const body = {op:'bevel_selection', object:ACTIVE, mode:$('bvmode').value, ratio:+$('bvratio').value};
  if($('bvmode').value==='fillet') body.segments=+$('bvseg').value;
  if(sel.length) body.verts=sel;
  $('bvmeta').textContent = 'beveling '+(sel.length?sel.length+' selected corners':'all corners')+'…';
  try{
    const r=await api('op', body);
    if(r.error){ $('bvmeta').textContent=r.error; return; }
    applyResp(r); $('bvmeta').textContent='Beveled '+(sel.length?sel.length+' corners':'all corners')+'.';
    status('Bevel applied.');
  }catch(e){ $('bvmeta').textContent='failed'; }
});

/* ================================ server-side parameter keyframes (P2-3) ================================ */
let PK_KEYABLE = [];
async function pkRefresh(){
  if(!ACTIVE){ $('pkmeta').textContent='Select an object with an edit history.'; return; }
  try{
    const r = await api_get('anim/keyable?object='+ACTIVE);
    PK_KEYABLE = r.keyable||[];
    const sel=$('pkParam');
    sel.innerHTML='<option value="">— pick a keyable param —</option>'+
      PK_KEYABLE.map((k,i)=>`<option value="${i}">${k.op}.${k.param} (=${k.value})</option>`).join('');
    $('pkmeta').textContent = PK_KEYABLE.length? PK_KEYABLE.length+' keyable parameters.' : 'No keyable params — apply a modifier (flute, taper, lattice…) first.';
  }catch(e){ $('pkmeta').textContent='failed to load params'; }
}
$('pkRefresh') && ($('pkRefresh').onclick = pkRefresh);
$('pkParam') && ($('pkParam').onchange = ()=>{
  const k=PK_KEYABLE[+$('pkParam').value]; if(!k) return;
  $('pkV0').value=k.value; $('pkV1').value=(k.value*2||0.1).toFixed(3);      // sensible defaults from current value
});
async function pkBakeFrame(frame, commit){
  const k=PK_KEYABLE[+$('pkParam').value];
  if(!k){ $('pkmeta').textContent='Pick a parameter first.'; return null; }
  const keys=[[+$('pkF0').value, +$('pkV0').value]];
  if($('pkMidOn') && $('pkMidOn').checked) keys.push([+$('pkFm').value, +$('pkVm').value]);
  keys.push([+$('pkF1').value, +$('pkV1').value]);
  const body={object:ACTIVE, index:k.index, param:k.param, keys, frame,
    easing: $('pkEase') ? $('pkEase').value : 'linear'};
  const r=await (await fetch('api/anim/bake',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(r.error){ $('pkmeta').textContent=r.error; return null; }
  if(commit) applyResp(r);
  return r;
}
$('pkMidOn') && ($('pkMidOn').onchange = ()=>{ const on=$('pkMidOn').checked; $('pkVm').disabled=!on; $('pkFm').disabled=!on; });
$('pkBake') && ($('pkBake').onclick = async ()=>{
  const r=await pkBakeFrame(ANIM.t|0, true);
  if(r) $('pkmeta').textContent='Baked '+PK_KEYABLE[+$('pkParam').value].param+' = '+r.value+' at frame '+(ANIM.t|0)+'. (Geometry updated — export/photo will see it.)';
  pkRefresh();
});
$('pkScrub') && ($('pkScrub').onclick = async ()=>{
  const k=PK_KEYABLE[+$('pkParam').value];
  if(!k){ $('pkmeta').textContent='Pick a parameter first.'; return; }
  $('pkmeta').textContent='sweeping…';
  const f0=+$('pkF0').value, f1=+$('pkF1').value;
  // preview 8 frames across the range, animating in the viewport
  const N=8;
  for(let i=0;i<=N;i++){
    const f=f0+(f1-f0)*i/N;
    const r=await pkBakeFrame(f, true);
    if(!r) return;
    await new Promise(res=>setTimeout(res,90));
  }
  $('pkmeta').textContent='Swept '+k.param+' from '+$('pkV0').value+' to '+$('pkV1').value+'. Object left at end value.';
});
// refresh keyable params whenever the keys dialog opens
(function(){ const kb=document.querySelector('[data-dlg="dlg-keys"]'); if(kb) kb.addEventListener('click', ()=>setTimeout(pkRefresh,60)); })();

/* ================================ animated ocean (H1-4) ================================ */
$('ocT') && ($('ocT').oninput = async ()=>{
  $('ocTval').textContent=(+$('ocT').value).toFixed(1);
  if(!ACTIVE) return;
  try{
    const r=await api('ocean/animate', {object:ACTIVE, time:+$('ocT').value});
    if(r.error){ $('ocmeta').textContent=r.error; return; }
    applyResp(r);
  }catch(e){}
});
$('ocScrub') && ($('ocScrub').onclick = async ()=>{
  if(!ACTIVE){ $('ocmeta').textContent='Select a generated ocean first.'; return; }
  $('ocmeta').textContent='playing swell…';
  for(let i=0;i<=24;i++){
    const t=i*0.5;
    try{
      const r=await api('ocean/animate', {object:ACTIVE, time:t});
      if(r.error){ $('ocmeta').textContent=r.error; return; }
      applyResp(r); $('ocT').value=Math.min(12,t); $('ocTval').textContent=Math.min(12,t).toFixed(1);
      await new Promise(res=>setTimeout(res,70));
    }catch(e){ $('ocmeta').textContent='failed'; return; }
  }
  $('ocmeta').textContent='Swell played. Waves travel — exports and GI photos capture the current frame.';
});

/* ================================ node subgraphs / reusable groups (P3-2) ================================ */
async function refreshGroupList(){
  try{
    const r=await nodeApi({action:'group_list'});
    const sel=$('nodegroupsel'); if(!sel) return;
    sel.innerHTML='<option value="">insert group…</option>'+
      (r.groups||[]).map(g=>`<option value="${g.name}">${g.name} (${g.nodes}n)</option>`).join('');
  }catch(e){}
}
$('nodegroupsave') && ($('nodegroupsave').onclick = async ()=>{
  const ids = NODE.multi.size ? [...NODE.multi] : (NODE.sel?[NODE.sel]:[]);
  if(ids.length<1){ status('Select nodes first (box-select or shift-click), then Save group.'); return; }
  const name = await uiPrompt('Group nodes', 'Name this node group:', 'group');
  if(!name) return;
  const r=await nodeApi({action:'group_save', ids, name});
  if(r.error){ status(r.error); return; }
  await refreshGroupList();
  status(`Saved group "${r.name}": ${r.nodes} nodes, ${r.internal_edges} internal edges, ${r.net_inputs}in/${r.net_outputs}out.`);
});
$('nodegroupsel') && ($('nodegroupsel').onchange = async ()=>{
  const name=$('nodegroupsel').value; if(!name) return;
  const r=await nodeApi({action:'group_insert', name, x:120, y:120});
  $('nodegroupsel').value='';
  if(r.error){ status(r.error); return; }
  nodeRefresh();
  status(`Inserted group "${name}" (${r.ids.length} nodes, ${r.edges} internal edges wired).`);
});

/* ================================ hierarchical parenting (A2-3) ================================ */
$('parentbtn') && ($('parentbtn').onclick = async ()=>{
  if(!ACTIVE){ $('hiermeta').textContent='Select an active object to be the parent.'; return; }
  const kids=[...selObjs].filter(id=>id!==ACTIVE);
  if(!kids.length){ $('hiermeta').textContent='Shift-select one or more other objects, plus the active parent.'; return; }
  let n=0, err='';
  for(const k of kids){
    const r=await api('parent', {action:'set', child:k, parent:ACTIVE});
    if(r.error) err=r.error; else n++;
  }
  $('hiermeta').textContent = n? `Parented ${n} object(s) to "${(OBJS.get(ACTIVE)||{d:{}}).d.name}". Move it to move them all.` : (err||'nothing parented');
  status('Hierarchy updated.');
});
$('unparentbtn') && ($('unparentbtn').onclick = async ()=>{
  if(!ACTIVE) return;
  const r=await api('parent', {action:'clear', child:ACTIVE});
  $('hiermeta').textContent = r.error ? r.error : 'Detached from parent.';
});

/* ================================ creature walk cycle (H1-7 gait) ================================ */
$('wkT') && ($('wkT').oninput = async ()=>{
  $('wkTval').textContent=(+$('wkT').value).toFixed(2);
  if(!ACTIVE) return;
  try{
    const r=await api('creature/walk', {object:ACTIVE, time:+$('wkT').value});
    if(r.error){ $('wkmeta').textContent=r.error; return; }
    applyResp(r);
  }catch(e){}
});
$('wkScrub') && ($('wkScrub').onclick = async ()=>{
  if(!ACTIVE){ $('wkmeta').textContent='Select a generated creature with legs first.'; return; }
  $('wkmeta').textContent='walking…';
  for(let i=0;i<=32;i++){
    const t=i/16;
    try{
      const r=await api('creature/walk', {object:ACTIVE, time:t});
      if(r.error){ $('wkmeta').textContent=r.error; return; }
      applyResp(r); $('wkT').value=Math.min(2,t); $('wkTval').textContent=Math.min(2,t).toFixed(2);
      await new Promise(res=>setTimeout(res,60));
    }catch(e){ $('wkmeta').textContent='failed'; return; }
  }
  $('wkmeta').textContent='Walk played — legs stride, body holds. Exports/photos capture the current pose.';
});

/* ================================ parametric sketcher (F1-2) ================================ */
$('skH') && ($('skH').oninput = ()=>$('skHval').textContent=(+$('skH').value).toFixed(2));
if($('skLc')) $('skLc').parentElement.style.display='none';
$('skTemplate') && ($('skTemplate').onchange = ()=>{
  const t=$('skTemplate').value;
  const la=$('skLa'), lb=$('skLb'), lc=$('skLc');
  if(t==='rect'){ la.textContent='Width'; lb.textContent='Height'; lc.parentElement.style.display='none'; }
  else if(t==='rtri'){ la.textContent='Leg A'; lb.textContent='Leg B'; lc.parentElement.style.display='none'; }
  else if(t==='para'){ la.textContent='Base'; lb.textContent='Height'; lc.textContent='Slant'; lc.parentElement.style.display='flex'; }
  else { la.textContent='Width'; lb.textContent='Height'; lc.textContent='Thickness'; lc.parentElement.style.display='flex'; }
});
function skBuild(){
  const A=+$('skA').value, B=+$('skB').value, C=+$('skC').value, t=$('skTemplate').value;
  if(t==='rect') return {
    points:[[0.03,-0.02],[A*1.05,0.04],[A*0.97,B],[-0.03,B*1.04]],
    constraints:[{type:'fixed',pts:[0],value:[0,0]},{type:'horizontal',pts:[0,1]},{type:'vertical',pts:[0,3]},
      {type:'horizontal',pts:[3,2]},{type:'vertical',pts:[1,2]},{type:'distance',pts:[0,1],value:A},{type:'distance',pts:[0,3],value:B}]};
  if(t==='rtri') return {
    points:[[0.02,-0.02],[A,0.03],[-0.02,B]],
    constraints:[{type:'fixed',pts:[0],value:[0,0]},{type:'horizontal',pts:[0,1]},{type:'vertical',pts:[0,2]},
      {type:'distance',pts:[0,1],value:A},{type:'distance',pts:[0,2],value:B}]};
  if(t==='para') return {
    points:[[0,0],[A,0.03],[A+C,B],[C*1.1,B*0.96]],
    constraints:[{type:'fixed',pts:[0],value:[0,0]},{type:'horizontal',pts:[0,1]},{type:'distance',pts:[0,1],value:A},
      {type:'parallel',pts:[0,1,3,2]},{type:'parallel',pts:[1,2,0,3]},{type:'distance',pts:[0,3],value:Math.hypot(C,B)},{type:'horizontal',pts:[3,2]}]};
  // lshape
  return {
    points:[[0,0],[A,0.02],[A*0.98,C],[C,C*1.02],[C*0.97,B],[-0.02,B*1.01]],
    constraints:[{type:'fixed',pts:[0],value:[0,0]},{type:'horizontal',pts:[0,1]},{type:'distance',pts:[0,1],value:A},
      {type:'vertical',pts:[1,2]},{type:'distance',pts:[1,2],value:C},{type:'horizontal',pts:[2,3]},
      {type:'vertical',pts:[0,5]},{type:'distance',pts:[0,5],value:B},{type:'horizontal',pts:[5,4]},{type:'vertical',pts:[3,4]}]};
}
$('skGo') && ($('skGo').onclick = async ()=>{
  const spec=skBuild();
  $('skmeta').textContent='solving…';
  try{
    const r=await api('sketch/solve', Object.assign(spec, {extrude:true, height:+$('skH').value, name:'Sketch'}));
    if(r.error){ $('skmeta').textContent=r.error; return; }
    if(r.object!==undefined) applyResp(r);
    const conv = r.converged ? 'solved (residual '+(r.max_residual)+')' : 'did NOT fully converge (residual '+r.max_residual+')';
    $('skmeta').textContent = 'Constraints '+conv+(r.extrude_error?(' — extrude failed: '+r.extrude_error):' — extruded.');
    status('Parametric sketch '+conv+'.');
  }catch(e){ $('skmeta').textContent='failed: '+e.message; }
});

/* ================================ camera solve from vanishing points (A0-1) ================================ */
$('pt_camsolve') && ($('pt_camsolve').onclick = ()=>{ openDlg('dlg-camsolve'); });
function parseLines(txt){
  return txt.split('\n').map(s=>s.trim()).filter(Boolean).map(s=>s.split(/[, ]+/).map(Number)).filter(a=>a.length>=4 && a.every(n=>!isNaN(n)));
}
$('csGo') && ($('csGo').onclick = async ()=>{
  const la=parseLines($('csA').value), lb=parseLines($('csB').value);
  if(la.length<2 || lb.length<2){ $('csmeta').textContent='Need at least 2 lines in each family (x1,y1,x2,y2 per row).'; return; }
  $('csmeta').textContent='solving…';
  try{
    const r=await api('camera/solve', {width:+$('csW').value, height:+$('csH').value, lines_a:la, lines_b:lb});
    if(r.error){ $('csmeta').textContent=r.error; return; }
    $('csmeta').innerHTML = `Focal <b>${r.focal_px}px</b> (~${r.focal_35mm_equiv}mm eq), FOV ${r.fov_horizontal_deg}°<br>`+
      `Orthonormal: ${r.orthonormal?'✓':'—'} · ${r.note}<br>`+
      `<button id="csApply">Point viewport camera like this</button>`;
    const btn=$('csApply');
    if(btn) btn.onclick=()=>{
      // apply the suggested orbit to the viewport camera
      if(typeof camTheta!=='undefined' && r.suggested_orbit){
        camTheta = r.suggested_orbit.theta*Math.PI/180;
        camPhi = Math.max(0.05, Math.min(Math.PI-0.05, (90 - r.suggested_orbit.phi)*Math.PI/180));
        if(typeof applyCam==='function') applyCam();
        if(typeof scheduleRender==='function') scheduleRender();
        status('Viewport camera pointed to match the solved photo angle.');
      }
    };
    status('Camera solved: focal '+r.focal_px+'px.');
  }catch(e){ $('csmeta').textContent='failed: '+e.message; }
});

/* ================================ Inspect dialog (measure · mass · section · bounds) ================================ */
async function insRunMeasure(){
  if(!ACTIVE){ $('insMeasure').textContent='Select an object first.'; return; }
  try{
    const m=await (await fetch('api/measure?object='+ACTIVE)).json();
    if(m.error){ $('insMeasure').textContent=m.error; return; }
    let s=`${m.name}\ndims  ${m.dims.join(' × ')}`;
    if(m.dims_units) s+=`  (${m.dims_units.join(' × ')} ${m.units})`;
    s+=`\nvol   ${m.volume}    area ${m.area}\nfaces ${m.faces}    verts ${m.verts}`;
    if(m.watertight===false) s+='\n⚠ open mesh — volume approximate';
    $('insMeasure').textContent=s;
  }catch(e){ $('insMeasure').textContent='measure failed'; }
}
// auto-run measure whenever the Inspect dialog is opened
document.querySelectorAll('[data-dlg="dlg-inspect"]').forEach(b=>b.addEventListener('click', ()=>setTimeout(insRunMeasure,60)));
$('insMassGo') && ($('insMassGo').onclick = async ()=>{
  if(!ACTIVE){ $('insMass').textContent='Select an object first.'; return; }
  $('insMass').textContent='computing…';
  try{
    const r=await (await fetch(`api/mass_properties?object=${ACTIVE}&density=${+$('insDensity').value||1}`)).json();
    if(r.error){ $('insMass').textContent=r.error; return; }
    $('insMass').textContent=
      `vol ${r.volume}   mass ${r.mass}\nCOM  (${r.center_of_mass.map(x=>x.toFixed(4)).join(', ')})\n`+
      `principal moments ${r.principal_moments.map(x=>x.toFixed(4)).join('  ')}`;
  }catch(e){ $('insMass').textContent='failed'; }
});
$('insSecGo') && ($('insSecGo').onclick = async ()=>{
  if(!ACTIVE){ $('insSection').textContent='Select an object first.'; return; }
  $('insSection').textContent='cutting…';
  try{
    const r=await (await fetch(`api/section/measure?object=${ACTIVE}&axis=${$('insAxis').value}&offset=${+$('insOffset').value||0}&res=320`)).json();
    if(r.error){ $('insSection').textContent=r.error; return; }
    $('insSection').textContent = r.area===0 ? 'plane misses the solid at this offset'
      : `area ${r.area}   perimeter ${r.perimeter}   contours ${r.contours}`;
  }catch(e){ $('insSection').textContent='failed'; }
});
$('insBoundsGo') && ($('insBoundsGo').onclick = async ()=>{
  if(!ACTIVE){ $('insBounds').textContent='Select an object first.'; return; }
  $('insBounds').textContent='fitting…';
  try{
    const r=await (await fetch('api/bounding?object='+ACTIVE)).json();
    if(r.error){ $('insBounds').textContent=r.error; return; }
    $('insBounds').textContent=
      `AABB ${r.aabb.dims.map(x=>x.toFixed(3)).join(' × ')}   vol ${r.aabb.volume}\n`+
      `OBB  ${r.obb.dims.map(x=>x.toFixed(3)).join(' × ')}   vol ${r.obb.volume}\n`+
      `oriented box is ${r.tightness_ratio}× tighter`;
  }catch(e){ $('insBounds').textContent='failed'; }
});

/* draft report (numbers only) — shares dlg-draft's pull + threshold controls */
$('draftreport') && ($('draftreport').onclick = async ()=>{
  if(!ACTIVE){ $('draftmeta').textContent='Select an object first.'; return; }
  $('draftmeta').textContent='analysing…';
  try{
    const pull=$('draftpull').value, mind=+$('draftmin').value;
    const r=await (await fetch(`api/draft_report?object=${ACTIVE}&pull=${pull}&min_degrees=${mind}`)).json();
    if(r.error){ $('draftmeta').textContent=r.error; return; }
    $('draftmeta').textContent=
      `min draft ${r.min_draft}°   mean ${r.mean_draft}°\n`+
      `moldable ${(r.moldable_area_fraction*100).toFixed(1)}%   undercut ${(r.undercut_area_fraction*100).toFixed(1)}%   parting ${(r.parting_area_fraction*100).toFixed(1)}%`;
  }catch(e){ $('draftmeta').textContent='failed'; }
});

/* ================================ hydraulic erosion (H1-3) ================================ */
$('tErode') && ($('tErode').onclick = ()=>{ if(!ACTIVE){ status('Select a landscape object first.'); return; } openDlg('dlg-erode'); });
$('erDrop') && ($('erDrop').oninput = ()=>$('erDropVal').textContent=$('erDrop').value);
$('erStr') && ($('erStr').oninput = ()=>$('erStrVal').textContent=(+$('erStr').value).toFixed(1));
$('erGo') && ($('erGo').onclick = async ()=>{
  if(!ACTIVE){ $('erMeta').textContent='Select a landscape object first.'; return; }
  $('erMeta').textContent='eroding… (simulating rainfall)';
  try{
    const r=await (await fetch('api/erode',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({object:ACTIVE, droplets:+$('erDrop').value, strength:+$('erStr').value, seed:+$('erSeed').value})})).json();
    if(r.error){ $('erMeta').textContent=r.error; return; }
    applyResp(r); const e=r.erosion;
    $('erMeta').textContent=`peak ${e.peak_before} → ${e.peak_after}\nrelief ${e.relief_before} → ${e.relief_after}\nmaterial moved ${e.material_moved}`;
    status('Eroded: peaks lowered, drainage carved.');
  }catch(e){ $('erMeta').textContent='failed: '+e.message; }
});

/* ================================ C3-3: docked properties panel (N-panel) ================================ */
function objPanelToggleDocked(){
  const p=$('dlg-object'); if(!p) return;
  const open=p.classList.contains('open');
  if(!open){ p.classList.add('open','docked'); refreshObjList && refreshObjList(); }   // N -> open docked
  else if(p.classList.contains('docked')){ p.classList.remove('open'); }                  // N again -> close
  else { p.classList.add('docked'); }                                                     // was floating -> dock it
  const db=$('objDock'); if(db) db.textContent = p.classList.contains('docked') ? '⇤' : '⇥';
}
$('objDock') && ($('objDock').onclick = (e)=>{
  e.stopPropagation();
  const p=$('dlg-object');
  p.classList.toggle('docked');
  $('objDock').textContent = p.classList.contains('docked') ? '⇤' : '⇥';
  $('objDock').title = p.classList.contains('docked') ? 'Float the panel' : 'Dock to the right edge (N)';
});

/* ================================ browser-storage scene slots (localStorage) ================================ */
const STORAGE_PREFIX = 'polystudio:scene:';
function storageAvailable(){
  try{ const k='__ps_test__'; localStorage.setItem(k,'1'); localStorage.removeItem(k); return true; }
  catch(e){ return false; }
}
function storageList(){
  const out=[];
  for(let i=0;i<localStorage.length;i++){
    const k=localStorage.key(i);
    if(k && k.startsWith(STORAGE_PREFIX)){
      let meta={};
      try{ const o=JSON.parse(localStorage.getItem(k)); meta={objects:(o.objects||[]).length, saved:o._savedAt||''}; }catch(e){}
      out.push({name:k.slice(STORAGE_PREFIX.length), key:k, ...meta});
    }
  }
  return out.sort((a,b)=>(b.saved||'').localeCompare(a.saved||''));
}
function storageBytesUsed(){
  let n=0; for(let i=0;i<localStorage.length;i++){ const k=localStorage.key(i); if(k&&k.startsWith(STORAGE_PREFIX)) n+=(localStorage.getItem(k)||'').length; }
  return n;
}
async function buildSceneJSON(){                                 // same payload as Save-to-disk, incl. animation
  const j = await (await fetch('api/scene/save')).json();
  if(ANIM.tracks.size || ANIM.cam.length){
    const order=[...OBJS.keys()]; const tr={};
    for(const [id,keys] of ANIM.tracks){ const i=order.indexOf(id); if(i>=0&&keys.length) tr[i]=keys; }
    j.animation={fps:ANIM.fps, len:ANIM.len, tracks:tr, cam:ANIM.cam};
  }
  return j;
}
async function loadSceneJSON(scene){
  const r = await (await fetch('api/scene/load', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(scene)})).json();
  if(r.error){ status('Load failed: '+r.error); return false; }
  applyResp(r);
  ANIM.tracks.clear(); ANIM.cam=[];
  if(scene.animation){
    const order=[...OBJS.keys()];
    ANIM.fps=scene.animation.fps||30; ANIM.len=scene.animation.len||90;
    if($('tlFps')){ $('tlFps').value=ANIM.fps; $('tlLen').value=ANIM.len; $('tlScrub').max=ANIM.len; }
    for(const [idx,keys] of Object.entries(scene.animation.tracks||{})){ const id=order[+idx]; if(id) ANIM.tracks.set(id, keys); }
    ANIM.cam=scene.animation.cam||[];
  }
  status('Scene loaded ('+r.loaded+' objects).');
  return true;
}
function renderStorageList(mode){
  const el=$('storageList'); if(!el) return;
  const rows=storageList();
  $('storageQuota') && ($('storageQuota').textContent = rows.length ? `(${(storageBytesUsed()/1024).toFixed(0)} KB used)` : '');
  if(!rows.length){ el.innerHTML='<div class="note">No scenes saved in this browser yet.</div>'; return; }
  el.innerHTML = rows.map(r=>`<div class="objrow" data-key="${r.key}" style="align-items:center">
    <span>${r.name}<span class="note"> · ${r.objects||0} obj${r.saved?' · '+r.saved.slice(0,16).replace('T',' '):''}</span></span>
    <span><button class="stOpen" data-key="${r.key}" style="padding:2px 8px">Open</button>
    <button class="stDel" data-key="${r.key}" title="Delete" style="padding:2px 6px">✕</button></span></div>`).join('');
  el.querySelectorAll('.stOpen').forEach(b=>b.onclick=async ()=>{
    if(OBJS && OBJS.size>0 && !await uiConfirm('Opening a scene replaces everything currently in the viewport. Continue?', 'Open')) return;
    try{ const scene=JSON.parse(localStorage.getItem(b.dataset.key)); if(await loadSceneJSON(scene)){ $('dlg-storage').classList.remove('open'); } }
    catch(e){ $('storageMeta').textContent='Could not read that slot.'; }
  });
  el.querySelectorAll('.stDel').forEach(b=>b.onclick=async ()=>{
    const nm=b.dataset.key.slice(STORAGE_PREFIX.length);
    if(await uiConfirm('Delete saved scene "'+nm+'"?', 'Delete')){ localStorage.removeItem(b.dataset.key); renderStorageList(mode); status('Deleted "'+nm+'".'); }
  });
}
function openStorage(mode){
  if(!storageAvailable()){ status('Browser storage is unavailable (private mode or disabled).'); return; }
  $('storageTitle').textContent = mode==='save' ? 'Save to browser storage' : 'Open from browser storage';
  $('storageMode').textContent = mode==='save'
    ? 'Type a name and Save, or overwrite an existing slot by reusing its name.'
    : 'Pick a saved scene to open.';
  $('storageSaveRow').style.display = mode==='save' ? 'flex' : 'none';
  $('storageMeta').textContent=''; renderStorageList(mode);
  openDlg('dlg-storage');
}
$('scenebrowsersave') && ($('scenebrowsersave').onclick = ()=>openStorage('save'));
$('scenebrowserload') && ($('scenebrowserload').onclick = ()=>openStorage('load'));
$('storageSaveGo') && ($('storageSaveGo').onclick = async ()=>{
  const name=($('storageName').value||'').trim();
  if(!name){ $('storageMeta').textContent='Give the scene a name first.'; return; }
  const key=STORAGE_PREFIX+name;
  if(localStorage.getItem(key) && !await uiConfirm('A scene named "'+name+'" exists. Overwrite it?', 'Overwrite')) return;
  $('storageMeta').textContent='saving…';
  try{
    const j=await buildSceneJSON(); j._savedAt=new Date().toISOString();
    try{ localStorage.setItem(key, JSON.stringify(j)); }
    catch(err){ $('storageMeta').textContent = /quota/i.test(err.message||'') ? 'Browser storage is full — delete a slot or use Save .json.' : 'Save failed: '+err.message; return; }
    $('storageName').value=''; renderStorageList('save');
    $('storageMeta').textContent='Saved "'+name+'" to this browser.'; status('Scene saved to browser ('+j.objects.length+' objects).');
  }catch(e){ $('storageMeta').textContent='Save failed: '+e.message; }
});


/* ================================ engine render (leCore rasteriser) ================================ */
// (the engine rasteriser now runs through rvRenderEngine() in the Render View — A1/A2)


/* ================================ first-impression UI fixes ================================ */
// matbar hide/show persistence + the show-again pill
try{ if(localStorage.getItem('polystudio:matbar')==='hidden') document.body.classList.add('matbar-hidden'); }catch(_){}
$('matbarshow') && ($('matbarshow').onclick = ()=>{ document.body.classList.remove('matbar-hidden');
  try{ localStorage.setItem('polystudio:matbar','shown'); }catch(_){} });

// context-menu "Object properties…": open the panel FOCUSED on the active (right-clicked) object,
// scroll its row into view and flash it so it's unambiguous which object the properties belong to
function openObjProps(){
  refreshObjList(); refreshAttrs();
  openDlg('dlg-object');
  const row = document.querySelector(`#objlist [data-id="${ACTIVE}"]`);
  if(row){
    row.scrollIntoView({block:'nearest'});
    row.style.transition='background 0.15s'; const bg=row.style.background;
    row.style.background='rgba(90,140,220,0.35)';
    setTimeout(()=>{ row.style.background=bg; }, 650);
  }
}


/* ================================ P3-2: collapse selection to a group node (Ctrl+G) ================================ */
// (Ctrl+G groups nodes -- KEYMAP, C4; the body is groupSelectedNodes())


/* ================================ basics pass: new scene + scene-graph view ================================ */
$('scenenew') && ($('scenenew').onclick = async ()=>{
  if(!await uiConfirm('Start a new scene? All unsaved objects, node graphs, imported textures and custom materials will be cleared.', 'New scene')) return;
  try{
    const r = await (await fetch('api/scene/new', {method:'POST'})).json();
    selObjs.clear(); ACTIVE=null; selFaces.clear(); selVerts.clear();
    applyResp(r); refreshObjList(); refreshAttrs(); refreshSelectionVisuals();
    if(NODE.open) nodeRefresh();
    status('New scene.');
  }catch(e){ status('New scene failed: '+e.message); }
});

let SG_ON=false;
$('scenegraphbtn') && ($('scenegraphbtn').onclick = async ()=>{
  SG_ON=!SG_ON;
  const v=$('scenegraphview'); v.classList.toggle('on', SG_ON);
  $('scenegraphbtn').classList.toggle('on', SG_ON);
  if(!SG_ON) return;
  try{
    const d = await (await fetch('api/scene_graph')).json();
    const sw = a => `<span class="swatch" style="background:rgb(${a.map(c=>Math.round(c*255)).join(',')})"></span>`;
    const objCards = d.objects.map(o=>{
      const tex = o.hasTexture?'<div class="sglink">texture ✓</div>':(o.hasFaceColors?'<div class="sglink">face colours ✓</div>':'');
      const par = o.parent?`<div class="sglink">parent: ${o.parent}</div>`:'';
      return `<div class="sgcard" data-oid="${o.id}" id="sgo_${o.id}"><b>${o.name}</b> <span class="sub">· ${o.faces}p / ${o.verts}v</span>${tex}${par}</div>`;
    }).join('');
    const matCards = d.materials.map((m,i)=>
      `<div class="sgcard" id="sgm_${i}">${sw(m.albedo)}${m.name}${m.custom?' <span class="sub">(session)</span>':''}</div>`).join('');
    const texCards = d.textures.map((t,i)=>`<div class="sgcard" id="sgt_${t.object}">🖼 ${t.size[0]}×${t.size[1]}</div>`).join('') || '<div class="sub" style="color:var(--dim)">none</div>';
    const lightCards = d.lights.map(l=>`<div class="sgcard">${l.kind==='environment'?'🌐':'💡'} ${l.role}<div class="sub">${l.kind}</div></div>`).join('');
    v.innerHTML = `<div id="sgwrap"><svg id="sgwires"></svg>`+
                  `<div class="sgcol"><h4>Objects (${d.objects.length})</h4>${objCards}</div>`+
                  `<div class="sgcol" style="margin-left:60px"><h4>Materials in use (${d.materials.length})</h4>${matCards}</div>`+
                  `<div class="sgcol" style="margin-left:60px"><h4>Textures (${d.textures.length})</h4>${texCards}</div>`+
                  `<div class="sgcol" style="margin-left:60px"><h4>Lights</h4>${lightCards}</div></div>`;
    // WIRES: object -> material (tinted by albedo, labelled with face count) and object -> texture (dashed)
    const wrap=$('sgwrap'), svg=$('sgwires');
    const anchor=(el,side)=>{ // point on a card edge, in wrap coordinates
      const x = el.offsetLeft + (side==='r'?el.offsetWidth:0), y = el.offsetTop + el.offsetHeight/2;
      return [x,y];
    };
    const midx=(m)=>d.materials.findIndex(x=>x.name===m);
    let paths='';
    for(const l of d.links){
      const src=$('sgo_'+l.object); if(!src) continue;
      let dst=null, color='#5a6f95', dash='', label='';
      if(l.material!==undefined){ const i=midx(l.material); dst=$('sgm_'+i);
        const a=(d.materials[i]||{}).albedo||[0.5,0.5,0.5];
        color=`rgb(${a.map(c=>Math.round(60+c*180)).join(',')})`; label=l.faces; }
      else if(l.texture!==undefined){ dst=$('sgt_'+l.texture); dash='stroke-dasharray="5,4"'; }
      if(!dst) continue;
      const [x1,y1]=anchor(src,'r'), [x2,y2]=anchor(dst,'l');
      const mx=(x1+x2)/2;
      paths+=`<path d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="${color}" stroke-width="1.6" opacity="0.85" ${dash}/>`;
      if(label) paths+=`<text x="${mx}" y="${(y1+y2)/2-4}" fill="${color}" font-size="9" text-anchor="middle">${label}</text>`;
    }
    svg.setAttribute('width', wrap.scrollWidth); svg.setAttribute('height', wrap.scrollHeight);
    svg.setAttribute('viewBox', `0 0 ${wrap.scrollWidth} ${wrap.scrollHeight}`);
    svg.innerHTML = paths;
    v.querySelectorAll('.sgcard[data-oid]').forEach(el=>{
      el.onclick = ()=>{ const id=el.dataset.oid; selObjs.clear(); selObjs.add(id); ACTIVE=id;
        refreshObjList(); refreshAttrs(); refreshSelectionVisuals(); status('Selected '+id+' from scene view.'); };
    });
  }catch(e){ v.innerHTML = '<div style="color:var(--dim)">scene graph failed: '+e.message+'</div>'; }
});


/* ================================ still-camera progressive resolve ================================ */
// While the camera is parked, keep asking the server for one more accumulated sample; the server anti-
// aliases + de-noises by averaging jittered full-quality frames (SDF bakes are cached, so rounds are cheap),
// reports the inter-round delta, and flags convergence — at which point we STOP: no noise left to remove.
const PROG = {token:0, running:false, ctrl:null};
async function startProgressive(){
  if(PROG.running || !previewWanted()) return;              // B1/B4
  PROG.running = true;
  const myToken = PROG.token;
  try{
    for(let round=0; round<24; round++){
      if(PROG.token!==myToken || photoBusy || RV.busy) break;   // camera moved, scene edited, or a render started
      PROG.ctrl = new AbortController();
      let r;
      try{ r = await fetch(`api/render_progressive?session=${SESSION_ID}&w=760&`+camParams(), {signal:PROG.ctrl.signal}); }
      catch(e){ break; }                                   // aborted mid-flight by a camera move
      PROG.ctrl = null;
      if(PROG.token!==myToken) break;                      // stale by the time it arrived — don't show it
      if(!r.ok) break;
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      if(!rvShow(url,'live')) { URL.revokeObjectURL(url); break; }   // A3: a final render is showing — stop
      const n=r.headers.get('X-Round'), d=parseFloat(r.headers.get('X-Delta')||'1');
      if(r.headers.get('X-Converged')==='1'){
        rvMeta(`resolved — ${n} samples, noise gone (Δ ${(d*255).toFixed(2)}/255)`);
        break;
      }
      rvMeta(`resolving… sample ${n} (Δ ${(d*255).toFixed(2)}/255)`);
    }
  }catch(e){ /* keep the last good frame */ }
  finally{ PROG.running=false; }
}


/* ================================ engine status (leos-core extras awareness) ================================ */
$('enginestatus') && ($('enginestatus').onclick = async ()=>{
  try{
    const [d, pf] = await Promise.all([
      (await fetch('api/engine_status')).json(),
      (await fetch('api/engine_preflight')).json()
    ]);
    const ex = Object.entries(d.extras).map(([k,v])=>`${v?'✓':'—'} ${k}`).join('\n');
    // Preflight: this app can run on a pip-installed engine OR a vendored copy, and they are not
    // necessarily the same version. Say plainly whether the live engine has everything the app calls.
    const pfTxt = pf.ok
      ? `Capabilities: all ${pf.required} required features present.`
      : `Capabilities: ${pf.present}/${pf.required} present — MISSING:\n`
        + pf.missing.map(m=>`  · ${m.breaks}`).join('\n') + `\n\n${pf.hint}`;
    uiReport('Engine status', `Engine: ${d.resolution}\nroot: ${d.engine_root}\n\n${pfTxt}\n\nOptional tiers:\n${ex}\n\n${d.install_hint}`);
  }catch(e){ status('engine status failed: '+e.message); }
});


/* ================================ GPU auto-enable (WebGPU when available) ================================ */
// The client WebGPU sphere-tracer redraws LIVE during camera moves (engine-emitted WGSL per object). If the
// browser has an adapter, turn it on automatically — the checkbox still shows and controls the state, and
// the existing error path falls back to the server preview cleanly.
// DELIBERATELY OPT-IN. This used to auto-enable whenever the browser reported a WebGPU adapter, which
// meant it switched itself on for most Chrome users and put an overlay canvas across the viewport. It is
// an experimental path that cannot be exercised in CI here, so it does not get to turn itself on: the
// checkbox in the Render panel enables it, and the label says the browser supports it.
(async ()=>{
  try{
    if(!navigator.gpu || !$('gpustatus')) return;
    $('gpustatus').textContent = '(WebGPU available — tick to enable)';
  }catch(_){ }
})();


/* ===================== Scatter instances on a surface (engine meshscatter) ===================== */
function scatterFillSelects(){
  const opts = [...OBJS.values()].map(o=>`<option value="${o.d.id}">${o.d.name}</option>`).join('');
  const src=$('scsrc'), tgt=$('sctgt');
  if(!src||!tgt) return;
  const keepS=src.value, keepT=tgt.value;
  src.innerHTML=opts; tgt.innerHTML=opts;
  // sensible defaults: instance = the selected object, surface = something else
  const sel = ACTIVE || (OBJS.size?[...OBJS.keys()][0]:'');
  src.value = keepS && [...OBJS.keys()].includes(keepS) ? keepS : sel;
  const others=[...OBJS.keys()].filter(id=>id!==src.value);
  tgt.value = keepT && others.includes(keepT) ? keepT : (others[0]||sel);
}
$('scjit') && ($('scjit').oninput = e=>{ $('scjitv').textContent = (+e.target.value).toFixed(2); });
$('scalign') && ($('scalign').oninput = e=>{ $('scalignv').textContent = (+e.target.value).toFixed(2); });
$('scclose') && ($('scclose').onclick = $('sccancel').onclick = ()=>closeDlg('dlg-scatter'));
$('scgo') && ($('scgo').onclick = async ()=>{
  const body = {source:$('scsrc').value, target:$('sctgt').value,
                count:$('sccount').value|0, scale:+$('scscale').value,
                scale_jitter:+$('scjit').value, align:+$('scalign').value, seed:$('scseed').value|0};
  if($('scblue').checked) body.radius = Math.max(0.01, (+$('scscale').value)*0.9);   // min spacing ~ instance size
  if(body.source===body.target){ status('Pick two different objects: one to copy, one to scatter it on.'); return; }
  status('Scattering…');
  try{
    const r = await fetch('api/scatter', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error||r.status);
    applyResp(d);
    const s=d.scatter;
    status(`Scattered ${s.placements} copies of ${s.source} on ${s.target} (${s.faces.toLocaleString()} faces).`);
    closeDlg('dlg-scatter');
  }catch(e){ status('Scatter failed: '+e.message); }
});


/* ==================== Shared workspace (.lews) — the bridge to leStudio ==================== */
let LEWS_BLOB = null;
$('lewsbtn') && ($('lewsbtn').onclick = ()=>$('lewsfile').click());
$('lewsfile') && ($('lewsfile').onchange = async ()=>{
  const f = $('lewsfile').files[0]; if(!f) return;
  status('Reading workspace…');
  LEWS_BLOB = await f.arrayBuffer();
  try{
    const r = await fetch('api/workspace/inspect', {method:'POST', body:LEWS_BLOB});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error||r.status);
    $('lwfile').textContent = `${f.name} — written by ${d.app || 'an unknown app'}`;
    // name every section, including ones we cannot open: carried, not silently dropped
    const secs = (d.sections||[]).reduce((m,s)=>{ m[s.kind]=(m[s.kind]||0)+1; return m; }, {});
    $('lwsections').textContent = Object.entries(secs).map(([k,n])=>`${n}× ${k}`).join(' · ');
    $('lwtex').innerHTML = (d.textures||[]).length
      ? d.textures.map(t=>`<div class="sub">🖼 <b>${t.name}</b> — ${t.w}×${t.h}${t.layers>1?`, ${t.layers} layers`:''}</div>`).join('')
      : '<div class="sub" style="color:var(--dim)">none — this workspace has no painted documents</div>';
    $('lwgo').disabled = !(d.textures||[]).length;
    openDlg('dlg-lews');
  }catch(e){ status('Could not read that workspace: '+e.message); }
  $('lewsfile').value='';
});
$('lwclose') && ($('lwclose').onclick = $('lwcancel').onclick = ()=>{ closeDlg('dlg-lews'); LEWS_BLOB=null; });
$('lwgo') && ($('lwgo').onclick = async ()=>{
  if(!LEWS_BLOB) return;
  const target = ($('lwapply').checked && ACTIVE) ? `?object=${encodeURIComponent(ACTIVE)}` : '';
  closeDlg('dlg-lews'); status('Importing workspace…');
  try{
    const r = await fetch('api/workspace/import'+target, {method:'POST', body:LEWS_BLOB});
    const d = await r.json();
    if(!r.ok) throw new Error(d.error||r.status);
    applyResp(d);
    const w = d.workspace||{};
    status(`Imported ${(w.textures||[]).length} texture(s)`
         + (w.applied_to ? ` — applied to the selected object` : '')
         + (w.carried_sections ? ` · ${w.carried_sections} section(s) from the other app will be carried on export` : ''));
  }catch(e){ status('Workspace import failed: '+e.message); }
  LEWS_BLOB=null;
});
$('lewsexp') && ($('lewsexp').onclick = ()=>{
  status('Exporting shared workspace…');
  window.location = 'api/workspace/export';
});
