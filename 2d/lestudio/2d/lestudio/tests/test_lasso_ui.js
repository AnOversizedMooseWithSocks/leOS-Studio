// R74 step 01: drives the SHIPPED lasso code out of index.html against the
// DOM shim. What is pinned is the GESTURE contract, because that is what a
// person actually does: a freehand drag collects points and posts one poly
// selection on release, a flick clears instead, modifiers override the mode
// dropdown for that drag only, and the polygon lasso accumulates corners that
// Backspace takes back, Enter and the first-corner click close, and Esc
// abandons -- without the half-drawn shape reaching the "clear the layer"
// Delete binding.
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
const { makeDocument, Element } = require('./domshim.js');

const HTML = fs.readFileSync(process.env.LS_HTML || path.join(__dirname, '../src/lestudio/static/index.html'), 'utf8');
const JS = HTML.split('<script>')[1].split('</script>')[0];
let fails = 0, passes = 0;
const ok = (name, cond, extra) => {
  if (cond) { passes++; console.log('  ok   ' + name); }
  else { fails++; console.log('  FAIL ' + name + (extra ? '  -> ' + extra : '')); }
};

const doc = makeDocument('<select id="selMode"><option value="new" selected>new</option></select><input id="selFeather" value="0">');
doc.getElementById('selMode').value = 'new';
function grab(re, label) { const m = JS.match(re); if (!m) throw new Error('could not find ' + label); return m[0]; }

// the shipped pieces, lifted whole
const src = [
  grab(/function selModFromEvent\(e\)\{[\s\S]*?\n\}/, 'selModFromEvent'),
  grab(/const SHIFTTOOLKEY=\{[\s\S]*?\};/, 'SHIFTTOOLKEY').replace(/^ +const/, 'var'),
].join('\n');

const posted = [];
const ctx = vm.createContext({
  document: doc, console, setTimeout, Element,
  $: id => doc.getElementById(id),
  tool: 'lasso', activeSel: null,
});
vm.runInContext(src + '\nthis.selModFromEvent=selModFromEvent;this.SHIFTTOOLKEY=SHIFTTOOLKEY;', ctx);

console.log('\nselection modifier keys');
ok('no modifier keeps the dropdown', ctx.selModFromEvent({}) === null);
ok('shift adds', ctx.selModFromEvent({ shiftKey: true }) === 'add');
ok('alt subtracts', ctx.selModFromEvent({ altKey: true }) === 'subtract');
ok('shift+alt intersects', ctx.selModFromEvent({ shiftKey: true, altKey: true }) === 'intersect');

console.log('\nShift+letter cycles within a family');
ok('shift+L off the lasso reaches the polygon lasso', ctx.SHIFTTOOLKEY.l === 'polylasso', ctx.SHIFTTOOLKEY.l);
ok('shift+M off rect reaches the ellipse', ctx.SHIFTTOOLKEY.m === 'ellipse', ctx.SHIFTTOOLKEY.m);

// the gesture code lives inside the big pointer handlers, so re-create the
// exact branches here from the source text and run them: this catches a typo
// or a renamed variable, which is what these shim tests are for.
console.log('\nthe gesture branches exist in the shipped source and name live things');
const need = [
  [/if\(tool==='lasso'\)\{ lasso=\{pts:\[\[x,y\]\],mod:selModFromEvent\(e\)\}/, 'pointerdown opens a freehand lasso with the modifier'],
  [/if\(tool==='polylasso'\)\{/, 'pointerdown handles the polygon lasso'],
  [/finishPolyLasso\(true\); return;\n {6}\}/, 'clicking the first corner closes the polygon'],
  [/if\(lasso\)\{\n {4}const \[lx,ly\]=canvasXY\(e\)/, 'pointermove extends the freehand path'],
  [/if\(Math\.hypot\(lx-last\[0\],ly-last\[1\]\)>=2\) lasso\.pts\.push/, 'the path is thinned to ~2px'],
  [/if\(polyLasso\)\{ polyLasso\.hover=canvasXY\(e\)/, 'pointermove tracks the rubber band'],
  [/await runSelect\('poly',\{points:l\.pts\.map/, 'release posts a poly selection'],
  [/else deselect\(\); +\/\/ a flick with a lasso means "clear it"/, 'a flick clears instead'],
  [/async function finishPolyLasso\(keep\)\{/, 'finishPolyLasso exists'],
  [/vc\.addEventListener\('dblclick',e=>\{ if\(polyLasso\)/, 'double-click closes the polygon'],
  [/if\(polyLasso\)\{\n {4}if\(e\.key==='Enter'\)\{e\.preventDefault\(\);finishPolyLasso\(true\);return;\}/, 'Enter closes the polygon'],
  [/if\(e\.key==='Backspace'\)\{\n {6}e\.preventDefault\(\);\n {6}polyLasso\.pts\.pop\(\);/, 'Backspace takes back a corner'],
  [/if\(e\.key==='Escape'\)\{e\.preventDefault\(\);finishPolyLasso\(false\);return;\}/, 'Esc abandons the polygon'],
  [/if\(polyLasso&&t!=='polylasso'\)\{ polyLasso=null;/, 'switching tool abandons a half-drawn shape'],
  [/let lasso=null, polyLasso=null;/, 'both states are declared'],
  [/stroke=null; sent=0; smoothPt=null; marquee=null; lasso=null; polyLasso=null;/, 'both states are reset with the rest'],
];
for (const [re, label] of need) ok(label, re.test(JS));

// --- and now RUN them. The branches above are lifted out of the pointer
// handlers verbatim and executed, so this proves the flow works rather than
// that the text is present: a renamed variable or a wrong argument fails here.
console.log('\nthe gestures, executed');
const gsrc = [
  grab(/async function finishPolyLasso\(keep\)\{[\s\S]*?\n\}/, 'finishPolyLasso'),
  'let lasso=null, polyLasso=null;',
  // pointerdown branches, verbatim from the shipped handler
  'function down(e){ const [x,y]=canvasXY(e);\n'
    + grab(/ +if\(tool==='lasso'\)\{ lasso=\{pts:\[\[x,y\]\],mod:selModFromEvent\(e\)\}; vc\.setPointerCapture\(e\.pointerId\); \}/, 'down lasso') + '\n'
    + grab(/ +if\(tool==='polylasso'\)\{\n[\s\S]*?\n {4}\}/, 'down polylasso') + '\n}',
  // pointermove branches
  'function move(e){\n'
    + grab(/ +if\(lasso\)\{\n[\s\S]*?\n {2}\}/, 'move lasso') + '\n'
    + grab(/ +if\(polyLasso\)\{ polyLasso\.hover=canvasXY\(e\); repaintCanvas\(\); drawMarquee\(\); return; \}/, 'move polylasso') + '\n}',
  // pointerup branch
  'async function up(){\n' + grab(/ +if\(lasso\)\{\n {4}const l=lasso; lasso=null;[\s\S]*?\n {2}\}/, 'up lasso') + '\n}',
  // the polygon key branch
  'async function key(e){\n' + grab(/ +if\(polyLasso\)\{\n {4}if\(e\.key==='Enter'\)[\s\S]*?\n {2}\}/, 'poly keys') + '\n}',
].join('\n');
const g = vm.createContext({
  document: doc, console, setTimeout, Math, Object,
  $: id => doc.getElementById(id),
  tool: 'lasso', activeSel: null, vctx: null,
  canvasXY: e => [e.x, e.y],
  vc: { setPointerCapture(){} },
  repaintCanvas(){}, drawMarquee(){}, setStatus(){},
  deselect: () => posted.push(['deselect']),
  selModFromEvent: ctx.selModFromEvent,
  runSelect: async (t, p, m) => { posted.push([t, p, m]); },
});
vm.runInContext(gsrc + '\nthis.down=down;this.move=move;this.up=up;this.key=key;this.state=()=>({lasso,polyLasso});', g);

(async () => {
  // freehand: press, drag round a box, release -> one poly selection, add mode
  posted.length = 0;
  g.down({ x: 10, y: 10, shiftKey: true, pointerId: 1 });
  [[60,10],[60,60],[10,60],[10,12]].forEach(([x,y]) => g.move({ x, y }));
  ok('the drag collected the corners', g.state().lasso.pts.length === 5, JSON.stringify(g.state().lasso.pts));
  await g.up();
  ok('release posted one poly selection', posted.length === 1 && posted[0][0] === 'poly', JSON.stringify(posted));
  ok('with the points rounded', posted[0][1].points[0].every(Number.isInteger));
  ok('and shift meant add', posted[0][2] === 'add', posted[0][2]);
  ok('the lasso state is cleared', g.state().lasso === null);

  // sub-pixel jitter is thinned away
  posted.length = 0;
  g.down({ x: 0, y: 0, pointerId: 1 });
  for (let i = 0; i < 10; i++) g.move({ x: i * 0.5, y: 0 });
  ok('jitter under 2px adds no points', g.state().lasso.pts.length <= 3, g.state().lasso.pts.length);
  await g.up();
  ok('a flick clears instead of selecting', posted.length === 1 && posted[0][0] === 'deselect', JSON.stringify(posted));

  // polygon: corners, Backspace, close on the first point
  g.tool = 'polylasso';
  posted.length = 0;
  [[20,20],[80,20],[80,80],[20,80]].forEach(([x,y]) => g.down({ x, y, altKey: true }));
  ok('four corners collected', g.state().polyLasso.pts.length === 4);
  await g.key({ key: 'Backspace', preventDefault(){} });
  ok('Backspace took one back', g.state().polyLasso.pts.length === 3);
  g.down({ x: 22, y: 21 });                      // within 8px of the first
  await new Promise(r => setTimeout(r, 5));
  ok('clicking the first corner closed it', posted.length === 1 && posted[0][0] === 'poly', JSON.stringify(posted));
  ok('alt meant subtract', posted[0][2] === 'subtract', posted[0][2]);
  ok('the polygon state is cleared', g.state().polyLasso === null);

  // Enter closes, Esc abandons
  posted.length = 0;
  [[5,5],[50,5],[50,50]].forEach(([x,y]) => g.down({ x, y }));
  await g.key({ key: 'Enter', preventDefault(){} });
  ok('Enter closed it', posted.length === 1 && posted[0][0] === 'poly');
  posted.length = 0;
  [[5,5],[50,5],[50,50]].forEach(([x,y]) => g.down({ x, y }));
  await g.key({ key: 'Escape', preventDefault(){} });
  ok('Esc abandoned it with no selection', posted.length === 0 && g.state().polyLasso === null);
  // two corners is not a shape
  posted.length = 0;
  [[5,5],[50,5]].forEach(([x,y]) => g.down({ x, y }));
  await g.key({ key: 'Enter', preventDefault(){} });
  ok('two corners close to nothing', posted.length === 0);

  console.log('\n' + passes + ' passed, ' + fails + ' failed');
  process.exit(fails ? 1 : 0);
})();

console.log('\nthe polygon keys run BEFORE the destructive ones');
const iPoly = JS.indexOf("if(polyLasso){\n    if(e.key==='Enter')");
const iDel = JS.indexOf("if((e.key==='Delete'||e.key==='Backspace')&&mode==='nodes'");
ok('Backspace on a corner cannot reach the clear-layer binding', iPoly > 0 && iDel > iPoly, iPoly + ' vs ' + iDel);

console.log('\nthe overlay draws the lasso, not just the marquee');
ok('drawMarquee handles a lasso in progress', /function drawMarquee\(\)\{\n {2}if\(lasso\|\|polyLasso\)\{/.test(JS));
ok('the closing edge is drawn faint', /vctx\.globalAlpha=0\.45/.test(JS));
ok('polygon corners get handles', /pts\.forEach\(\(p,i\)=>\{vctx\.beginPath\(\);vctx\.arc\(p\[0\],p\[1\],i===0\?4:2\.5/.test(JS));

console.log('\nrunSelect takes the override and the marquee passes it too');
ok('runSelect has a modeOverride parameter', /async function runSelect\(tool,params,modeOverride\)\{/.test(JS));
ok('the override wins over the dropdown', /const mode=modeOverride\|\|\$\('selMode'\)\.value;/.test(JS));
ok('a marquee drag carries its modifier', /await runSelect\(tool,\{x0:Math\.round\(m\.x0\)[\s\S]{0,120}\},m\.mod\);/.test(JS));
