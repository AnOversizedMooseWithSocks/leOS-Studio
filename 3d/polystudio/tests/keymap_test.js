// Poly Studio UI regression harness. Loads index.html in headless Chromium with a stub THREE
// and asserts the behaviours the 1.1.0 UX pass fixed. It needs no server: /api/* is mocked.
//   CHROME=/path/to/chrome PUPPETEER_PATH=/path/to/node_modules node tests/keymap_test.js
const PAGE = 'file://' + require('path').resolve(__dirname, '..', 'index.html');
const P=process.env.PUPPETEER_PATH||'';
const puppeteer=require(require.resolve('puppeteer',{paths:[P]}));
(async()=>{
 const b=await puppeteer.launch({executablePath:process.env.CHROME||undefined,args:['--no-sandbox']});
 const p=await b.newPage();
 const errs=[]; p.on('pageerror',e=>errs.push(e.message));
 await p.setRequestInterception(true);
 p.on('request',r=>r.url().startsWith('file://')?r.continue():r.respond({status:404,body:''}));
 await p.evaluateOnNewDocument(`const mk=()=>new Proxy(function(){},{get:(t,k)=>{if(k==='then')return undefined;if(!t[k])t[k]=mk();return t[k];},set:(t,k,v)=>{t[k]=v;return true},apply:()=>mk(),construct:()=>mk()});window.THREE=mk();window.THREE.Vector3=function(x,y,z){this.x=x||0;this.y=y||0;this.z=z||0;this.set=function(){return this};this.copy=function(){return this};this.addScaledVector=function(){return this};this.setFromMatrixColumn=function(){return this};this.clone=function(){return new window.THREE.Vector3(this.x,this.y,this.z)};this.sub=function(){return this};this.length=function(){return 1};this.normalize=function(){return this};this.applyMatrix4=function(){return this};this.distanceTo=function(){return 1};this.lerp=function(){return this};};`);
 await p.goto(PAGE,{waitUntil:'domcontentloaded'});
 await new Promise(r=>setTimeout(r,900));
 const res = await p.evaluate(async ()=>{
  const out={};
  const fire=(key,mods={})=>{
    const e=new KeyboardEvent('keydown',Object.assign({key,bubbles:true,cancelable:true},mods));
    document.dispatchEvent(e); return e;
  };
  // spy on the actions the table calls
  const calls=[];
  const wrap=(name)=>{ const orig=window[name]; window[name]=(...a)=>{ calls.push(name+':'+a[0]); return orig&&orig(...a); }; };
  ['setMode','setTool','setDisplay','selAll','selNone','selInvert','frameSelected','openDlg','toggleDlg','rvLoadHist','perfRender'].forEach(wrap);

  // THE BUG C4 FIXES: Alt+1 used to fire setDisplay AND setMode('object')
  calls.length=0; fire('1',{altKey:true});
  out.alt1_calls = calls.slice();
  out.alt1_no_double = calls.some(c=>c==='setDisplay:textured') && !calls.some(c=>c.startsWith('setMode'));

  calls.length=0; fire('1');                       out.bare1 = calls.slice();
  calls.length=0; fire('3');                       out.bare3 = calls.slice();
  calls.length=0; fire('w');                       out.w     = calls.slice();
  calls.length=0; fire('c');                       out.c     = calls.slice();
  calls.length=0; fire('c',{ctrlKey:true});        out.ctrlC = calls.slice();   // must be EMPTY
  calls.length=0; fire('x',{ctrlKey:true});        out.ctrlX = calls.slice();   // must be EMPTY
  calls.length=0; fire('f');                       out.f     = calls.slice();
  calls.length=0; fire('f',{ctrlKey:true});        out.ctrlF = calls.slice();   // must be EMPTY
  calls.length=0; fire('a');                       out.a     = calls.slice();
  calls.length=0; fire('a',{altKey:true});         out.altA  = calls.slice();
  calls.length=0; fire('P',{shiftKey:true});       out.shiftP= calls.slice();
  calls.length=0; fire('p');                       out.p     = calls.slice();

  // typing guard: none of these may fire from inside a textarea
  const ta=document.createElement('textarea'); document.body.appendChild(ta); ta.focus();
  calls.length=0;
  const ev=new KeyboardEvent('keydown',{key:'1',bubbles:true,cancelable:true}); ta.dispatchEvent(ev);
  const ev2=new KeyboardEvent('keydown',{key:'Backspace',bubbles:true,cancelable:true}); ta.dispatchEvent(ev2);
  out.typing_blocked = calls.length===0 && !ev2.defaultPrevented;
  ta.remove();

  // preventDefault where the browser owns the key
  out.prevent_ctrlK = fire('k',{ctrlKey:true}).defaultPrevented;
  out.prevent_altA  = fire('a',{altKey:true}).defaultPrevented;
  out.no_prevent_w  = !fire('w').defaultPrevented;
  document.getElementById('cmdpal').classList.remove('open');

  // REGRESSIONS FOUND IN SELF-AUDIT (all shipped broken in the first 1.1.0 cut):
  // '?' is Shift+/ and '~' is Shift+`, so a table entry written bare must still match a Shift event
  calls.length=0; fire('?',{shiftKey:true});  out.shifted_qmark = calls.slice();
  calls.length=0; fire('~',{shiftKey:true});  out.shifted_tilde_ran = !!document.getElementById('perfhud');
  // macOS emits an accented char for Alt+letter and Alt+digit; e.code must rescue the binding
  calls.length=0; fire('\u00e5',{altKey:true, code:'KeyA'});   out.mac_altA = calls.slice();
  calls.length=0; fire('\u00a1',{altKey:true, code:'Digit1'}); out.mac_alt1 = calls.slice();
  // a conditional binding must beat an unconditional one on the same key
  document.getElementById('dlg-renderview').classList.add('open');
  document.getElementById('rvhist').classList.add('show');
  RV.hist=[{src:'x',label:'a',short:'a'},{src:'y',label:'b',short:'b'}];
  calls.length=0; fire('1'); out.digit_prefers_render_history = calls.slice();
  document.getElementById('rvhist').classList.remove('show');
  calls.length=0; fire('1'); out.digit_falls_back_to_mode = calls.slice();
  document.getElementById('dlg-renderview').classList.remove('open');

  // the shortcuts dialog is generated from the same table
  const body=document.querySelector('#dlg-shortcuts .dlgbody');
  out.doc_rows = body.querySelectorAll('.scrow').length;
  out.doc_has_ctrlK = /Ctrl\+K/.test(body.textContent);
  out.doc_has_redo  = /Redo/.test(body.textContent);
  out.doc_no_loading = !/loading/.test(body.textContent);
  // every documented binding that has a run() must be dispatchable
  out.orphans = KEYMAP.filter(e=>e.run && !(e.combo||e.combos)).map(e=>e.label);
  out.n_entries = KEYMAP.length; out.n_combos = KEYMAP_INDEX.size;
  return out;
 });
 console.log(JSON.stringify(res,null,1));
 console.log('page errors:', errs.length?errs:'none');
 await b.close();
 
 // A harness that cannot fail is worse than no harness. Assert, then set the exit code.
 const fails = (()=>{
   const f=[];
   const truthy=['alt1_no_double','typing_blocked','prevent_ctrlK','prevent_altA','no_prevent_w',
                 'doc_has_ctrlK','doc_has_redo','doc_no_loading','shifted_tilde_ran'];
   for(const k of truthy) if(!res[k]) f.push(k);
   const empty=['ctrlC','ctrlX','ctrlF'];                    // the browser must keep these
   for(const k of empty) if((res[k]||[]).length) f.push(k+' stole a browser shortcut');
   const nonEmpty=['bare1','bare3','w','c','f','a','altA','shifted_qmark','mac_altA','mac_alt1',
                   'digit_prefers_render_history','digit_falls_back_to_mode'];
   for(const k of nonEmpty) if(!(res[k]||[]).length) f.push(k+' did not fire');
   if(!/rvLoadHist/.test(String(res.digit_prefers_render_history))) f.push('digit ignored render history');
   if(!/setMode/.test(String(res.digit_falls_back_to_mode))) f.push('digit did not fall back to mode');
   if((res.orphans||[]).length) f.push('undispatchable entries: '+res.orphans.join('/'));
   if(!(res.doc_rows>20)) f.push('shortcuts dialog looks empty');
   return f;
 })();
 if (fails.length) { console.error('\nFAILED: ' + fails.join(', ')); process.exit(1); }
 console.log('\nPASS');

})();
