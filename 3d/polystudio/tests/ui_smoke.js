// Poly Studio UI regression harness. Loads index.html in headless Chromium with a stub THREE
// and asserts the behaviours the 1.1.0 UX pass fixed. It needs no server: /api/* is mocked.
//   CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules node tests/ui_smoke.js
const PAGE = 'file://' + require('path').resolve(__dirname, '..', 'index.html');
const path=process.env.PUPPETEER_PATH||'';
const puppeteer=require(require.resolve('puppeteer',{paths:[path]}));
const fs=require('fs');


// A Proxy-based THREE stub: every property access returns another callable/constructible proxy, so
// app.js's scene-graph wiring runs to completion without a real WebGL context. We are smoke-testing
// OUR code paths (element wiring, the Render View, dialogs), not three.js.
const STUB = `
window.__errs=[];
const mk = () => new Proxy(function(){}, {
  get:(t,k)=>{ if(k==='then') return undefined;
               if(k===Symbol.toPrimitive) return ()=>0;
               if(!t[k]) t[k]=mk(); return t[k]; },
  set:(t,k,v)=>{ t[k]=v; return true; },
  apply:()=>mk(), construct:()=>mk()
});
window.THREE = mk();
window.THREE.Vector3 = function(x,y,z){ this.x=x||0; this.y=y||0; this.z=z||0;
  this.set=function(){return this}; this.copy=function(){return this};
  this.addScaledVector=function(){return this}; this.setFromMatrixColumn=function(){return this};
  this.clone=function(){return new window.THREE.Vector3(this.x,this.y,this.z)};
  this.sub=function(){return this}; this.length=function(){return 1};
  this.normalize=function(){return this}; this.applyMatrix4=function(){return this};
  this.distanceTo=function(){return 1}; this.lerp=function(){return this}; };
`;
(async()=>{
  const browser=await puppeteer.launch({executablePath:process.env.CHROME||undefined, args:['--no-sandbox','--disable-gpu','--disable-dev-shm-usage']});
  const page=await browser.newPage();
  const errs=[], logs=[];
  page.on('pageerror', e=>errs.push('PAGEERROR: '+e.message));
  page.on('console', m=>{ if(m.type()==='error') logs.push('CONSOLE: '+m.text()); });
  await page.setRequestInterception(true);
  page.on('request', req=>{
    const u=req.url();
    if(u.startsWith('file://')) return req.continue();
    if(u.includes('/api/')) return req.respond({status:200,contentType:'application/json',body:'{}'});
    return req.respond({status:404,body:''});                 // CDN three.js: blocked, stub takes over
  });
  await page.evaluateOnNewDocument(STUB);
  await page.goto(PAGE,{waitUntil:'domcontentloaded'});
  await new Promise(r=>setTimeout(r,1500));
  const probe = await page.evaluate(async ()=>{
    const out={};
    const has=id=>!!document.getElementById(id);
    out.rvExists = has('dlg-renderview') && has('rvstage') && has('rvimg');
    out.rvFns = ['rvShow','rvRender','rvCancel','previewWanted','rvZoomFit','rvPush','rvWipe']
                  .filter(f=>typeof window[f]!=='function');
    out.guards = ['isTypingTarget','bareKey'].filter(f=>typeof window[f]!=='function');
    // A3: with a final render showing, a live preview write must be REFUSED
    try{
      RV.slot="live";
      const ok1 = rvShow('data:image/gif;base64,R0lGODlhAQABAAAAACw=','final');
      const ok2 = rvShow('data:image/gif;base64,R0lGODlhAQABAAAAACw=','live');
      out.a3 = (ok1===true && ok2===false);
    }catch(e){ out.a3='threw: '+e.message; }
    // B1: preview must be refused while the render view is shut
    try{ document.getElementById('dlg-renderview').classList.remove('open');
         out.b1 = (previewWanted()===false); }catch(e){ out.b1='threw: '+e.message; }
    // G3: the command palette must exist, open, and rank a real menu item first
    try{
      openCommandPalette();
      const pal=document.getElementById('cmdpal');
      const inp=document.getElementById('cpinput');
      inp.value='bevel'; inp.dispatchEvent(new Event('input'));
      const rows=[...document.querySelectorAll('#cplist .cprow')].map(r=>r.textContent);
      out.g3 = pal.classList.contains('open') && rows.length>0 && /bevel/i.test(rows[0]);
      out.g3rows = rows.slice(0,3);
      document.getElementById('cmdpal').classList.remove('open');
    }catch(e){ out.g3='threw: '+e.message; }
    // G4: dialogs carry a role and an accessible name; icon-only toolstrip buttons are labelled
    try{
      const dlgs=[...document.querySelectorAll('.dlg')];
      out.g4_roles = dlgs.every(d=>d.getAttribute('role')==='dialog' && d.getAttribute('aria-labelledby'));
      const tb=[...document.querySelectorAll('#toolstrip button, .rvbar button')];
      out.g4_labels = tb.length>0 && tb.every(b=>(b.getAttribute('aria-label')||'').length>0);
    }catch(e){ out.g4='threw: '+e.message; }
    // G5: the viewport must opt out of browser touch gestures
    try{ out.g5 = getComputedStyle(document.getElementById('view')).touchAction==='none'; }
    catch(e){ out.g5='threw: '+e.message; }
    // F1: redo must be wired
    out.f1 = !!document.getElementById('redobtn');
    // B5: the hand-rolled ZIP writer must produce a real archive (PK header, CRC, EOCD)
    try{
      const enc=new TextEncoder();
      const blob=zipStore([{name:'a.txt', bytes:enc.encode('hello')},
                           {name:'b/c.png', bytes:new Uint8Array([1,2,3,4,5])}]);
      out.b5_size = blob.size;
      out.b5 = blob.type==='application/zip' && blob.size>100;
      out.b5_crc = crc32(enc.encode('hello'))>>>0;          // known value: 0x3610a686
      out.b5_crc_ok = out.b5_crc === 0x3610a686;
    }catch(e){ out.b5='threw: '+e.message; }
    // G6: job chips render with a working cancel
    try{
      let cancelled=false;
      const j=JOBS.start('Test job', ()=>{ cancelled=true; });
      j.progress(50,'halfway');
      const chip=document.querySelector('#jobs .jobchip');
      chip.querySelector('.jx').click();
      out.g6 = !!chip && /halfway/.test(chip.textContent) && cancelled===true;
      j.done('done');
    }catch(e){ out.g6='threw: '+e.message; }
    // D4: uiConfirm/uiPrompt/uiReport exist and the modal opens
    try{
      const p=uiConfirm('test?');
      const open=document.getElementById('uiask').classList.contains('open');
      document.querySelector('#uiask .cancel').click();
      out.d4 = open && (await p)===false;
    }catch(e){ out.d4='threw: '+e.message; }
    // B3: history must hold blob URLs that survive later display swaps (the old code revoked them)
    try{
      const PNG='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
      RV.hist.length=0; RV.keep.clear();
      const u1=rvBlobFromB64(PNG), u2=rvBlobFromB64(PNG);
      out.b3_isblob = u1.startsWith('blob:');
      RV.slot='live'; rvShow(u1,'final');
      rvPush({src:u1, label:'first', short:'1'});
      RV.slot='live'; rvShow(u2,'final');           // swapping display must NOT free the history entry
      const held = await fetch(u1).then(r=>r.ok).catch(()=>false);
      out.b3_history_survives = held===true && RV.keep.has(u1);
      // evicting past the cap must free it
      RV.histCap = 1;
      const u3=rvBlobFromB64(PNG);
      rvPush({src:u3, label:'second', short:'2'});
      const gone = await fetch(u1).then(()=>false).catch(()=>true);
      out.b3_evict_frees = gone===true && !RV.keep.has(u1);
      RV.histCap = 12;
    }catch(e){ out.b3='threw: '+e.message; }
    // B3 fallout: renders became blob URLs, and /api/upscale decodes base64. Posting img.src sent
    // the literal string "blob:null/<uuid>" and the 2x button failed with "could not read image".
    try{
      out.b3_upscale_converts = typeof rvImageAsDataURL === 'function';
      const PNG1='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==';
      RV.slot='live'; rvShow(rvBlobFromB64(PNG1),'final');
      const asData = await rvImageAsDataURL();
      out.b3_upscale_dataurl = asData.startsWith('data:image');
    }catch(e){ out.b3_upscale_dataurl='threw: '+e.message; }
    // C1: a keydown in a TEXTAREA must be treated as typing
    try{
      const ta=document.createElement('textarea'); document.body.appendChild(ta);
      out.c1 = isTypingTarget({target:ta})===true && bareKey({ctrlKey:true})===false;
    }catch(e){ out.c1='threw: '+e.message; }
    return out;
  });
  console.log(JSON.stringify(probe,null,1));
  console.log('--- page errors ---'); errs.forEach(e=>console.log(e));
  console.log('--- console errors ---'); logs.slice(0,12).forEach(e=>console.log(e));
  await browser.close();
  const fatal = errs.filter(e=>!/Failed to (load|fetch)|net::|three/i.test(e));
  process.exit(fatal.length ? 1 : 0);
})();
