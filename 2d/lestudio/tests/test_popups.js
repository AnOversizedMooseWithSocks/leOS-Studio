// Drives the SHIPPED placePopup() out of index.html and checks the invariant
// the colour-picker bug violated: a popup we draw must end up wholly inside the
// window, whatever size it is and wherever its anchor sits.
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
const HTML = fs.readFileSync(process.env.LS_HTML || path.join(__dirname, '../src/lestudio/static/index.html'), 'utf8');
const JS = HTML.split('<script>')[1].split('</script>')[0];

const m = JS.match(/function placePopup\(el, anchor, gap\)\{[\s\S]*?\n\}/);
if (!m) { console.log('FAIL: placePopup not found'); process.exit(1); }

let pass = 0, fail = 0;
const bad = [];
const PAD = 8;

function run(vw, vh, aw, ah, ax, ay, pw, ph) {
  const style = {};
  const el = {
    style,
    getBoundingClientRect() {
      // honour a maxHeight cap the way a real box would
      const capH = parseFloat(style.maxHeight), capW = parseFloat(style.maxWidth);
      return { width: isNaN(capW) ? pw : Math.min(pw, capW),
               height: isNaN(capH) ? ph : Math.min(ph, capH) };
    },
  };
  const anchor = { left: ax, top: ay, right: ax + aw, bottom: ay + ah };
  const ctx = vm.createContext({ window: { innerWidth: vw, innerHeight: vh } });
  vm.runInContext(m[0] + '\nthis.placePopup=placePopup;', ctx);
  const r = ctx.placePopup(el, anchor);
  const left = parseFloat(style.left), top = parseFloat(style.top);
  const h = r.height, w = r.width;

  const problems = [];
  if (!(top >= PAD - 0.5)) problems.push('top ' + top + ' above the window');
  if (!(top + h <= vh - PAD + 0.5)) problems.push('bottom ' + (top + h) + ' past ' + (vh - PAD));
  if (!(left >= PAD - 0.5)) problems.push('left ' + left + ' off the left edge');
  if (!(left + w <= vw - PAD + 0.5)) problems.push('right ' + (left + w) + ' past ' + (vw - PAD));
  if (problems.length) {
    fail++;
    if (bad.length < 6) bad.push(`vw=${vw} vh=${vh} anchor=(${ax},${ay},${aw}x${ah}) popup=${pw}x${ph}: ` + problems.join('; '));
  } else pass++;
}

// a grid that covers: roomy windows, short windows, anchors at the very top and
// the very bottom, and popups larger than the window in both axes
const VW = [1920, 1280, 900, 420, 260];
const VH = [1200, 900, 700, 400, 200, 120];
const POP = [[236, 300], [236, 640], [400, 1400], [120, 40], [900, 200]];
for (const vw of VW) for (const vh of VH) for (const [pw, ph] of POP) {
  for (const ay of [0, Math.round(vh * 0.5), vh - 40, vh - 4]) run(vw, vh, 60, 26, 10, ay, pw, ph);
  for (const ax of [0, Math.round(vw * 0.5), vw - 60, vw - 8]) run(vw, vh, 60, 26, ax, Math.round(vh * 0.6), pw, ph);
}

// the reported case: a colour swatch low in a tall sidebar
run(1512, 860, 60, 26, 1180, 700, 236, 300);
// and the one that must still prefer "below the anchor" when there is room
(function preferBelow() {
  const style = {};
  const el = { style, getBoundingClientRect: () => ({ width: 236, height: 300 }) };
  const ctx = vm.createContext({ window: { innerWidth: 1512, innerHeight: 900 } });
  vm.runInContext(m[0] + '\nthis.placePopup=placePopup;', ctx);
  ctx.placePopup(el, { left: 100, top: 100, right: 160, bottom: 126 });
  const ok = Math.abs(parseFloat(style.top) - 132) < 1.5;
  ok ? pass++ : (fail++, bad.push('with room below, the popup should sit under the anchor, got top=' + style.top));
})();

console.log(`placePopup: ${pass} placements inside the window, ${fail} outside`);
bad.forEach(b => console.log('  FAIL ' + b));
if (fail) process.exitCode = 1;

// ---- colour conversions must round-trip: a picker that shifts the colour on
// open is worse than one that is badly placed.
(function colourMaths(){
  const grab = re => { const g = JS.match(re); if(!g) throw new Error('missing '+re); return g[0]; };
  const ctx = vm.createContext({});
  vm.runInContext([
    grab(/function cpHex2Rgb\(h\)\{[\s\S]*?\n\}/),
    grab(/function cpRgb2Hex\(r,g,b\)\{[\s\S]*?\n\}/),
    grab(/function cpRgb2Hsv\(r,g,b\)\{[\s\S]*?\n\}/),
    grab(/function cpHsv2Rgb\(h,s,v\)\{[\s\S]*?\n\}/),
  ].join('\n') + '\nthis.h2r=cpHex2Rgb;this.r2h=cpRgb2Hex;this.r2v=cpRgb2Hsv;this.v2r=cpHsv2Rgb;', ctx);

  let bad = 0, n = 0;
  for (let r = 0; r < 256; r += 7) for (let g = 0; g < 256; g += 11) for (let b = 0; b < 256; b += 13) {
    n++;
    const hsv = ctx.r2v(r, g, b);
    const back = ctx.v2r(hsv[0], hsv[1], hsv[2]).map(Math.round);
    if (Math.abs(back[0]-r) > 1 || Math.abs(back[1]-g) > 1 || Math.abs(back[2]-b) > 1) {
      if (bad < 4) console.log(`  FAIL rgb(${r},${g},${b}) -> hsv -> rgb(${back})`);
      bad++;
    }
    const hx = ctx.r2h(r, g, b);
    const rt = ctx.h2r(hx);
    if (rt[0]!==r || rt[1]!==g || rt[2]!==b) { if (bad<8) console.log('  FAIL hex round-trip '+hx); bad++; }
  }
  // the specific colour from the report
  const p = ctx.r2v(180, 41, 255), q = ctx.v2r(p[0], p[1], p[2]).map(Math.round);
  if (q[0]!==180 || q[1]!==41 || q[2]!==255) { console.log('  FAIL reported purple: '+q); bad++; }
  // malformed input must be rejected, not silently turned into black
  ['', '#', 'nope', '#12345', '#gggggg', null].forEach(v => {
    if (ctx.h2r(v) !== null) { console.log('  FAIL cpHex2Rgb accepted '+JSON.stringify(v)); bad++; }
  });
  // short form works
  if (ctx.r2h(...ctx.h2r('#f0a')) !== '#ff00aa') { console.log('  FAIL 3-digit hex'); bad++; }

  console.log(`colour maths: ${n} rgb triples round-tripped, ${bad} wrong`);
  if (bad) process.exitCode = 1;
})();

// ---- no top-level function may be declared twice. Function declarations
// hoist and the LAST one wins, so a name collision silently replaces working
// code. Adding the picker's own hex2rgb() (0-255 ints) shadowed the existing
// hex2rgb() (0-1 floats) that every paint, fill and text call uses -- colours
// would have gone to the server 255x too large, with no error anywhere.
(function noShadowedFunctions(){
  const names = JS.match(/^\s*function\s+[A-Za-z_$][\w$]*\s*\(/gm)
    .map(s => s.trim().replace(/^function\s+/, '').replace(/\s*\($/, ''));
  const seen = {}, dupes = [];
  names.forEach(n => { if (seen[n]) { if (!dupes.includes(n)) dupes.push(n); } else seen[n] = 1; });
  console.log(`function names: ${names.length} declared, ${dupes.length} shadowed`);
  dupes.forEach(d => console.log('  FAIL ' + d + ' is declared more than once; the later one wins'));
  if (dupes.length) process.exitCode = 1;
})();

// ---- drive the picker end to end against the DOM shim -----------------------
(function pickerBehaviour(){
  const { makeDocument, Element } = require('./domshim.js');
  const cp = HTML.slice(HTML.indexOf('<div id="cpPop"'), HTML.indexOf('<div id="modalBack"'));
  const doc = makeDocument(cp + '<div id="host"><input type="color" id="bColor" value="#2b7bff"></div>');
  const $ = id => doc.getElementById(id);

  // give the parts a geometry the shim cannot compute
  $('cpPop')._rect = { left:0, top:0, width:236, height:300 };
  $('cpSV')._rect  = { left:0, top:0, width:216, height:132, right:216, bottom:132 };
  $('cpHue')._rect = { left:0, top:0, width:216, height:14, right:216, bottom:14 };
  $('bColor')._rect= { left:1180, top:700, right:1240, bottom:726, width:60, height:26 };

  const grab = re => { const g = JS.match(re); if(!g) throw new Error('missing '+re); return g[0]; };
  const src = [
    grab(/function placePopup\(el, anchor, gap\)\{[\s\S]*?\n\}/),
    grab(/function cpHex2Rgb\(h\)\{[\s\S]*?\n\}/),
    grab(/function cpRgb2Hex\(r,g,b\)\{[\s\S]*?\n\}/),
    grab(/function cpRgb2Hsv\(r,g,b\)\{[\s\S]*?\n\}/),
    grab(/function cpHsv2Rgb\(h,s,v\)\{[\s\S]*?\n\}/),
    grab(/const CP = \{target:null[\s\S]*?\n\}\)\(\);/),
  ].join('\n');
  const ctx = vm.createContext({
    document: doc, console, Event: class { constructor(t,o){ this.type=t; this.bubbles=!!(o&&o.bubbles); } },
    window: { innerWidth: 1512, innerHeight: 860, addEventListener(){} },
    recentCols: ['#ff0000', '#00ff00'],
  });
  vm.runInContext(src + '\nthis.CP=CP;', ctx);

  let bad = 0;
  const t = (name, cond, extra) => { if (cond) console.log('  ok   ' + name);
    else { bad++; console.log('  FAIL ' + name + (extra ? ' -> ' + extra : '')); } };

  // the native popup must be cancelled and ours opened in its place
  const ev = doc.fire($('bColor'), 'click');
  t('clicking a colour input cancels the browser picker', ev.defaultPrevented);
  t('our picker opens', $('cpPop').classList.contains('open'));
  t('it targets the input that was clicked', ctx.CP.target === $('bColor'));
  t('it loads the input\'s current colour', $('cpHex').value === '#2b7bff', $('cpHex').value);

  // the reported geometry: swatch at y=700 in an 860px window, popup 300 tall.
  // Below would end at 1032; it must flip above instead.
  const top = parseFloat($('cpPop').style.top);
  t('the picker stays on screen', top >= 8 && top + 300 <= 852, 'top=' + top);
  t('it flips above the swatch rather than off the bottom', top < 700, 'top=' + top);

  // dragging the hue bar edits the target and fires input, then change
  const fired = [];
  $('bColor').addEventListener('input',  () => fired.push('input'));
  $('bColor').addEventListener('change', () => fired.push('change'));
  const before = $('bColor').value;
  doc.fire($('cpHue'), 'pointerdown', { clientX: 108, clientY: 7 });
  t('dragging hue changes the colour', $('bColor').value !== before, $('bColor').value);
  t('an input event fires while dragging', fired.includes('input'));
  t('no change event until the drag ends', !fired.includes('change'));
  doc.fire($('cpHue'), 'pointerup', { clientX: 108, clientY: 7 });
  t('change fires on release', fired.includes('change'));

  // typing a hex updates the target; nonsense is ignored rather than applied
  const good = $('bColor').value;
  $('cpHex').value = '#b429ff';
  ctx.document.getElementById('cpHex').oninput({ target: $('cpHex') });
  t('typing a valid hex applies it', $('bColor').value === '#b429ff', $('bColor').value);
  $('cpHex').value = 'zzz';
  ctx.document.getElementById('cpHex').oninput({ target: $('cpHex') });
  t('typing nonsense leaves the colour alone', $('bColor').value === '#b429ff', $('bColor').value);

  // clicking outside closes it
  doc.fire($('host'), 'click');
  t('clicking outside closes the picker', !$('cpPop').classList.contains('open'));
  t('the target is released', ctx.CP.target === null);

  console.log(`picker behaviour: ${bad} failed`);
  if (bad) process.exitCode = 1;
})();

// ---- a close with no edit must not fire a spurious change ------------------
(function noSpuriousChange(){
  const { makeDocument } = require('./domshim.js');
  const cp = HTML.slice(HTML.indexOf('<div id="cpPop"'), HTML.indexOf('<div id="modalBack"'));
  const doc = makeDocument(cp + '<div id="host"><input type="color" id="bColor" value="#2b7bff"></div>');
  const $ = id => doc.getElementById(id);
  $('cpPop')._rect = { left:0, top:0, width:236, height:300 };
  $('bColor')._rect = { left:100, top:100, right:160, bottom:126, width:60, height:26 };
  const grab = re => JS.match(re)[0];
  const ctx = vm.createContext({
    document: doc, console, Event: class { constructor(t,o){ this.type=t; this.bubbles=!!(o&&o.bubbles); } },
    window: { innerWidth: 1512, innerHeight: 860, addEventListener(){} }, recentCols: [],
  });
  vm.runInContext([
    grab(/function placePopup\(el, anchor, gap\)\{[\s\S]*?\n\}/),
    grab(/function cpHex2Rgb\(h\)\{[\s\S]*?\n\}/), grab(/function cpRgb2Hex\(r,g,b\)\{[\s\S]*?\n\}/),
    grab(/function cpRgb2Hsv\(r,g,b\)\{[\s\S]*?\n\}/), grab(/function cpHsv2Rgb\(h,s,v\)\{[\s\S]*?\n\}/),
    grab(/const CP = \{target:null[\s\S]*?\n\}\)\(\);/),
  ].join('\n'), ctx);

  const fired = [];
  $('bColor').addEventListener('change', () => fired.push('change'));
  doc.fire($('bColor'), 'click');          // open
  doc.fire($('host'), 'click');            // close without touching anything
  const ok = fired.length === 0;
  console.log('open-then-close with no edit: ' + (ok ? 'no spurious change  ok' : 'FAIL fired ' + fired.length));
  if (!ok) process.exitCode = 1;
})();

// ---- soft-edit preview maths: the shipped softWeights / pulledGhost must
// mirror the server (weights) and behave like a rope (pull) -----------------
(function softEditPreview(){
  const grab = re => { const g = JS.match(re); if(!g) throw new Error('missing '+re); return g[0]; };
  let bad = 0;
  const t = (name, cond, extra) => { if (cond) console.log('  ok   ' + name);
    else { bad++; console.log('  FAIL ' + name + (extra ? ' -> ' + extra : '')); } };

  const pts = []; for (let i = 0; i < 50; i++) pts.push([20 + i * 4, 70]);
  const ctx = vm.createContext({
    Math,
    ssPaths: { K1: pts.map(p => [p[0], p[1]]) },
    ssStrokes: [{ id: 'K1', rigged: false, pins: [] }],
    ptSel: [['K1', 25]],
    $: id => ({ ssFall: { value: '60' }, ssStr: { value: '100' } }[id]),
  });
  vm.runInContext(grab(/function softWeights\(\)\{[\s\S]*?\n\}/) + '\nthis.softWeights=softWeights;', ctx);
  const W = ctx.softWeights().K1;
  t('grabbed joint weight is 1', Math.abs(W[25] - 1) < 1e-9, W[25]);
  t('immediate neighbour follows most of the way', W[24] > 0.9, W[24]);
  t('weights fade smoothly outward', W[20] > W[18] && W[18] > W[16]);
  t('outside the falloff nothing moves', W[5] === undefined && W[45] === undefined);
  // strength scales everything
  const ctx2 = vm.createContext(Object.assign({}, {
    Math, ssPaths: ctx.ssPaths, ssStrokes: ctx.ssStrokes, ptSel: ctx.ptSel,
    $: id => ({ ssFall: { value: '60' }, ssStr: { value: '50' } }[id]),
  }));
  vm.runInContext(grab(/function softWeights\(\)\{[\s\S]*?\n\}/) + '\nthis.softWeights=softWeights;', ctx2);
  t('strength halves the grabbed weight', Math.abs(ctx2.softWeights().K1[25] - 0.5) < 1e-9);

  // the rope preview: pull the far end, lengths hold, the pin holds
  const rope = []; for (let i = 0; i < 10; i++) rope.push([20 + i * 20, 70]);
  const ctx3 = vm.createContext({
    Math,
    ssPaths: { K2: rope },
    ssStrokes: [{ id: 'K2', rigged: true, pins: [0] }],
  });
  vm.runInContext(grab(/function pulledGhost\(sid,i,tx,ty\)\{[\s\S]*?\n\}/) + '\nthis.pulledGhost=pulledGhost;', ctx3);
  // reachable target: pin at (20,70), rope length 180, target 152 away
  const g = ctx3.pulledGhost('K2', 9, 150, 150);
  t('pulled end reaches the cursor', Math.abs(g[9][0] - 150) < 1e-9 && Math.abs(g[9][1] - 150) < 1e-9);
  t('the pin holds', Math.abs(g[0][0] - 20) < 0.8 && Math.abs(g[0][1] - 70) < 0.8,
    JSON.stringify(g[0]));
  let worst = 0;
  for (let i = 1; i < g.length; i++)
    worst = Math.max(worst, Math.abs(Math.hypot(g[i][0] - g[i-1][0], g[i][1] - g[i-1][1]) - 20));
  t('segment lengths stay rope-like (worst < 1px of 20)', worst < 1, worst.toFixed(2));
  t('midpoints actually moved toward the pull', g[5][1] > 71, g[5][1]);
  // UNREACHABLE target (245 away on a 180 rope): the physics cannot satisfy
  // both ends, so the right answer is EVEN stretch, never one sheared segment.
  // The first draft of this test asserted <3px here and "failed" -- the maths
  // was right and the expectation was wrong.
  const g2 = ctx3.pulledGhost('K2', 9, 260, 20);
  const lens = [];
  for (let i = 1; i < g2.length; i++)
    lens.push(Math.hypot(g2[i][0] - g2[i-1][0], g2[i][1] - g2[i-1][1]));
  const spread = Math.max(...lens) - Math.min(...lens);
  t('over-stretch distributes evenly across segments', spread < 6,
    'spread=' + spread.toFixed(2) + ' lens=' + lens.map(v=>v.toFixed(0)).join(','));

  console.log(`soft-edit preview: ${bad} failed`);
  if (bad) process.exitCode = 1;
})();

// ---- QuickShape fitter: the four shape classes -----------------------------
(function quickShapeFits(){
  const g = JS.match(/function quickShape\(pts\)\{[\s\S]*?\n\}/);
  if(!g){ console.log('  FAIL quickShape missing'); process.exitCode=1; return; }
  const ctx = vm.createContext({Math});
  vm.runInContext(g[0]+'\nthis.quickShape=quickShape;', ctx);
  const qs = ctx.quickShape;
  let bad = 0;
  const t=(n,c)=>{ if(c)console.log('  ok   '+n); else {bad++; console.log('  FAIL '+n);} };
  const line=[]; for(let i=0;i<40;i++)line.push([50+i*6+Math.sin(i)*2, 100+i*1.5+Math.cos(i*1.3)*2]);
  t('wobbly line snaps to line', (qs(line)||{}).kind==='line');
  const circ=[]; for(let i=0;i<=50;i++){const a=i/50*Math.PI*2;
    circ.push([200+Math.cos(a)*(60+Math.sin(i*2)*2.5), 200+Math.sin(a)*(60+Math.cos(i*3)*2.5)]);}
  const rc=qs(circ);
  t('wobbly circle snaps to circle', (rc||{}).kind==='circle');
  t('circle radius recovered', rc&&Math.abs(Math.hypot(rc.pts[0][0]-200,rc.pts[0][1]-200)-60)<3);
  const arc=[]; for(let i=0;i<=30;i++){const a=i/30*Math.PI/2;
    arc.push([300+Math.cos(a)*80+Math.sin(i)*1.5, 300+Math.sin(a)*80]);}
  t('quarter arc stays an arc, not a full circle', (qs(arc)||{}).kind==='arc');
  const scr=[]; for(let i=0;i<60;i++)scr.push([100+Math.sin(i*0.9)*70+i*2, 100+Math.cos(i*1.7)*55]);
  t('a scribble stays freehand', qs(scr)===null);
  t('tiny strokes ignored', qs([[0,0],[4,4],[8,8]])===null);
  console.log(`quickshape: ${bad} failed`);
  if (bad) process.exitCode = 1;
})();
