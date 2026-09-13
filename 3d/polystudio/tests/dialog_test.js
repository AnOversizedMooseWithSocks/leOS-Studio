// D1/D2/G4: dialog placement and modal focus containment.
//   d1_no_overlap        - dialogs cascade instead of stacking in one corner
//   d2_onscreen          - none can be pushed off the viewport
//   transform_dlg.left   - deliberately centred dialogs (left:50% + translateX) stay centred;
//                          the cascade once read '50%' as 50px and jammed them against the edge
//   modal_tab_prevented  - aria-modal="true" must actually contain Tab, or it is a lie to AT
//   CHROME=... PUPPETEER_PATH=... node tests/dialog_test.js
const P=process.env.PUPPETEER_PATH||'', CH=process.env.CHROME||undefined;
const puppeteer=require(require.resolve('puppeteer',{paths:[P]}));
const PAGE='file://'+require('path').resolve(__dirname,'..','index.html');
(async()=>{
 const b=await puppeteer.launch({executablePath:CH,args:['--no-sandbox']});
 const p=await b.newPage(); await p.setViewport({width:1280,height:800});
 await p.setRequestInterception(true);
 p.on('request',r=>r.url().startsWith('file://')?r.continue():r.respond({status:404,body:''}));
 await p.evaluateOnNewDocument(`const mk=()=>new Proxy(function(){},{get:(t,k)=>{if(k==='then')return undefined;if(!t[k])t[k]=mk();return t[k];},set:(t,k,v)=>{t[k]=v;return true},apply:()=>mk(),construct:()=>mk()});window.THREE=mk();window.THREE.Vector3=function(x,y,z){this.x=x||0;this.y=y||0;this.z=z||0;this.set=function(){return this};this.copy=function(){return this};this.addScaledVector=function(){return this};this.setFromMatrixColumn=function(){return this};this.clone=function(){return new window.THREE.Vector3(this.x,this.y,this.z)};this.sub=function(){return this};this.length=function(){return 1};this.normalize=function(){return this};this.applyMatrix4=function(){return this};this.distanceTo=function(){return 1};this.lerp=function(){return this};};`);
 await p.goto(PAGE,{waitUntil:'domcontentloaded'});
 await new Promise(r=>setTimeout(r,900));
 const o=await p.evaluate(async()=>{
  const out={};
  const ids=['dlg-render','dlg-camera','dlg-lighting','dlg-shortcuts','dlg-material'];
  ids.forEach(openDlg);
  const rects=ids.map(id=>{const r=document.getElementById(id).getBoundingClientRect();return {id,x:Math.round(r.left),y:Math.round(r.top)};});
  out.cascade = rects;
  // D1: no two dialogs may land on exactly the same spot
  const seen=new Set(); out.d1_no_overlap = rects.every(r=>{const k=r.x+','+r.y; if(seen.has(k))return false; seen.add(k); return true;});
  // D2: all must be on screen
  out.d2_onscreen = ids.every(id=>{const r=document.getElementById(id).getBoundingClientRect();
    return r.left>=0 && r.top>=36 && r.left < innerWidth-40 && r.top < innerHeight-40;});
  // transform-positioned dialogs must not be double-shifted by the cascade
  openDlg('dlg-import');
  const im=document.getElementById('dlg-import');
  out.transform_dlg = {left:Math.round(im.getBoundingClientRect().left), transform:getComputedStyle(im).transform!=='none'};
  ids.concat(['dlg-import']).forEach(closeDlg);

  // aria-modal honesty: can Tab escape the modal?
  const before=document.querySelectorAll('.dlg.open').length;
  const pr=uiConfirm('trap test?');
  const modal=document.getElementById('uiask');
  out.modal_aria = modal.querySelector('[aria-modal="true"]')!==null;
  const focusables=[...modal.querySelectorAll('button,input,select,textarea,[tabindex]')].filter(e=>e.offsetParent!==null);
  out.modal_focusables = focusables.length;
  out.modal_focus_inside = modal.contains(document.activeElement);
  // simulate Tab from the LAST focusable: without a trap, focus leaves the dialog
  focusables[focusables.length-1].focus();
  const ev=new KeyboardEvent('keydown',{key:'Tab',bubbles:true,cancelable:true});
  modal.dispatchEvent(ev);
  out.modal_tab_prevented = ev.defaultPrevented;     // a trap must preventDefault and wrap
  document.querySelector('#uiask .cancel').click(); await pr;
  return out;
 });
 console.log(JSON.stringify(o,null,1));
 await b.close();
 
 // A harness that cannot fail is worse than no harness. Assert, then set the exit code.
 const fails = (()=>{
   const f=[];
   for(const k of ['d1_no_overlap','d2_onscreen','modal_aria','modal_focus_inside','modal_tab_prevented'])
     if(!o[k]) f.push(k);
   if(!(o.transform_dlg.left>100)) f.push('centred dialog jammed to the edge (left='+o.transform_dlg.left+')');
   return f;
 })();
 if (fails.length) { console.error('\nFAILED: ' + fails.join(', ')); process.exit(1); }
 console.log('\nPASS');

})();
