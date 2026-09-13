// G5: touch gestures, and the ordering hazard they created. The touch module registers its
// listeners AFTER the mouse ones, so it cannot rely on capture-phase ordering to preempt them --
// the mouse handlers bail out on pointerType 'touch' instead. mouse_path_calls must be exactly 1.
//   CHROME=... PUPPETEER_PATH=... node tests/touch_test.js
const P=process.env.PUPPETEER_PATH||'', CH=process.env.CHROME||undefined;
const puppeteer=require(require.resolve('puppeteer',{paths:[P]}));
const PAGE='file://'+require('path').resolve(__dirname,'..','index.html');
// STRUCTURAL CHECK, and an honest note about what the runtime check can and cannot prove.
// Mutation-testing this harness showed that removing the pointerType guard from the mouse handlers
// does NOT produce a double-fire in Chromium: at the event target, capture-phase listeners run before
// bubble-phase ones, so the touch module's stopImmediatePropagation() already wins there. The guard is
// therefore defence against browsers that order at-target listeners by registration instead -- which
// the runtime probe below cannot exercise. So we assert the guard is present in the source as well.
const SRC = require('fs').readFileSync(require('path').resolve(__dirname,'..','app.js'),'utf8');
const guards = (SRC.match(/pointerType===['"]touch['"]\)\s*return/g)||[]).length;
if (guards < 3) {
  console.error(`FAILED: the mouse pointer handlers must bail out on touch (found ${guards} guards, expected >= 3)`);
  process.exit(1);
}
console.log(`source check: ${guards} pointerType guards on the mouse path`);
(async()=>{
 const b=await puppeteer.launch({executablePath:CH,args:['--no-sandbox']});
 const p=await b.newPage();
 await p.setRequestInterception(true);
 p.on('request',r=>r.url().startsWith('file://')?r.continue():r.respond({status:404,body:''}));
 await p.evaluateOnNewDocument(`const mk=()=>new Proxy(function(){},{get:(t,k)=>{if(k==='then')return undefined;if(!t[k])t[k]=mk();return t[k];},set:(t,k,v)=>{t[k]=v;return true},apply:()=>mk(),construct:()=>mk()});window.THREE=mk();window.THREE.Vector3=function(x,y,z){this.x=x||0;this.y=y||0;this.z=z||0;this.set=function(){return this};this.copy=function(){return this};this.addScaledVector=function(){return this};this.setFromMatrixColumn=function(){return this};this.clone=function(){return new window.THREE.Vector3(this.x,this.y,this.z)};this.sub=function(){return this};this.length=function(){return 1};this.normalize=function(){return this};this.applyMatrix4=function(){return this};this.distanceTo=function(){return 1};this.lerp=function(){return this};};`);
 await p.goto(PAGE,{waitUntil:'domcontentloaded'});
 await new Promise(r=>setTimeout(r,900));
 const out=await p.evaluate(async()=>{
  const o={}; const c=document.getElementById('view');
  c.setPointerCapture=()=>{}; c.releasePointerCapture=()=>{};
  const pd=(id,x,y,type='touch')=>c.dispatchEvent(new PointerEvent('pointerdown',
    {pointerId:id,pointerType:type,clientX:x,clientY:y,bubbles:true,cancelable:true,button:0}));
  const pm=(id,x,y,type='touch')=>c.dispatchEvent(new PointerEvent('pointermove',
    {pointerId:id,pointerType:type,clientX:x,clientY:y,bubbles:true,cancelable:true}));
  const pu=(id,x,y,type='touch')=>c.dispatchEvent(new PointerEvent('pointerup',
    {pointerId:id,pointerType:type,clientX:x,clientY:y,bubbles:true,cancelable:true}));

  // one finger drag = orbit
  const th0=camTheta, phi0=camPhi;
  pd(1,300,300); pm(1,360,300); pm(1,380,320); pu(1,380,320);
  o.one_finger_orbits = (camTheta!==th0) && (camPhi!==phi0);

  // two fingers = pinch zoom + pan
  const d0=camDist;
  pd(1,300,300); pd(2,400,300);
  pm(1,260,300); pm(2,440,300);            // spread -> zoom in
  o.pinch_zooms = camDist !== d0;
  const tx=camTarget.x;
  pm(1,300,340); pm(2,480,340);            // slide both -> pan
  pu(1,300,340); pu(2,480,340);
  o.two_finger_pans = true;                 // camTarget is a stub; assert no throw instead

  // a tap must SELECT, not orbit
  let picked=0; const origPick=window.pickScene; window.pickScene=(e)=>{picked++; return null;};
  const th1=camTheta;
  pd(3,200,200); pu(3,200,200);
  o.tap_selects = picked>0 && camTheta===th1;
  window.pickScene=origPick;

  // the MOUSE path must not also run for a touch pointer (this is the ordering hazard)
  let mouseRan=0;
  const origSel=window.selectAtPointer; window.selectAtPointer=(...a)=>{mouseRan++; return origSel(...a);};
  picked=0; window.pickScene=()=>{picked++; return null;};
  pd(4,210,210); pu(4,210,210);
  o.mouse_path_calls = mouseRan;            // exactly 1 = only the touch tap, not a duplicate
  window.selectAtPointer=origSel; window.pickScene=origPick;

  // and a real mouse still works
  const th2=camTheta;
  c.dispatchEvent(new PointerEvent('pointerdown',{pointerId:9,pointerType:'mouse',button:0,altKey:true,clientX:100,clientY:100,bubbles:true}));
  c.dispatchEvent(new PointerEvent('pointermove',{pointerId:9,pointerType:'mouse',clientX:160,clientY:100,bubbles:true}));
  c.dispatchEvent(new PointerEvent('pointerup',{pointerId:9,pointerType:'mouse',clientX:160,clientY:100,bubbles:true}));
  o.mouse_orbit_still_works = camTheta!==th2;
  return o;
 });
 console.log(JSON.stringify(out,null,1));
 await b.close();
 
 // A harness that cannot fail is worse than no harness. Assert, then set the exit code.
 const fails = (()=>{
   const f=[];
   for(const k of ['one_finger_orbits','pinch_zooms','tap_selects','mouse_orbit_still_works'])
     if(!out[k]) f.push(k);
   if(out.mouse_path_calls!==1) f.push('mouse path double-fired on touch ('+out.mouse_path_calls+' calls)');
   return f;
 })();
 if (fails.length) { console.error('\nFAILED: ' + fails.join(', ')); process.exit(1); }
 console.log('\nPASS');

})();
