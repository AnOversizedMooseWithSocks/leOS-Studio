// R74 step 02: drives the SHIPPED gradient-drag code against the DOM shim.
// The branches are lifted out of the pointer handlers and RUN, so a renamed
// variable or a wrong argument fails here rather than passing a text match.
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
function grab(re, label) { const m = JS.match(re); if (!m) throw new Error('could not find ' + label); return m[0]; }

const slice = (o, c) => HTML.slice(HTML.indexOf(o), HTML.indexOf(c));
const doc = makeDocument(slice('<div id="gradHud"', '<div id="creatureHud"')
  + '<select id="bSel"><option value="" selected></option></select><input id="bSelInv" type="checkbox">');
['grKind:linear', 'grFrom:#000000', 'grTo:#ffffff', 'grOp:100', 'grDither:0', 'bSel:'].forEach(p => {
  const [id, v] = p.split(':'); doc.getElementById(id).value = v;
});

const posted = [];
const src = [
  grab(/function hex2rgb\(h\)\{[^\n]*\n/, 'hex2rgb'),
  grab(/async function doGradient\(g\)\{[\s\S]*?\n\}/, 'doGradient'),
  grab(/\(function wireGradient\(\)\{[\s\S]*?\n\}\)\(\);/, 'wireGradient'),
  'let gradDrag=null;',
  'function down(e){ const [x,y]=canvasXY(e);\n'
    + grab(/ +if\(tool==='gradient'\)\{ if\(!sel\)return toast\('Pick a layer first'\); gradDrag=\{x0:x,y0:y,x1:x,y1:y\}; vc\.setPointerCapture\(e\.pointerId\); \}/, 'down gradient') + '\n}',
  'function move(e){\n' + grab(/ +if\(gradDrag\)\{\n {4}let \[gx2,gy2\]=canvasXY\(e\);[\s\S]*?\n {2}\}/, 'move gradient') + '\n}',
  'async function up(){\n' + grab(/ +if\(gradDrag\)\{\n {4}const g=gradDrag; gradDrag=null; repaintCanvas\(\);[\s\S]*?\n {2}\}/, 'up gradient') + '\n}',
].join('\n');

const ctx = vm.createContext({
  document: doc, console, setTimeout, Math, Element, JSON,
  $: id => doc.getElementById(id),
  tool: 'gradient', sel: 'L2', J: {},
  canvasXY: e => [e.x, e.y],
  vc: { setPointerCapture() {} },
  repaintCanvas() {}, drawMarquee() {}, refresh() {},
  setStatus: m => posted.push(['status', m]),
  toast: m => posted.push(['toast', m]),
  api: async (url, opt) => { posted.push([url, JSON.parse(opt.body)]); return { ok: true }; },
});
vm.runInContext(src + '\nthis.down=down;this.move=move;this.up=up;this.state=()=>gradDrag;this.doGradient=doGradient;', ctx);

(async () => {
  console.log('\na drag posts one gradient with the geometry it drew');
  posted.length = 0;
  ctx.down({ x: 10, y: 50, pointerId: 1 });
  ctx.move({ x: 100, y: 50 });
  ctx.move({ x: 190, y: 50 });
  ok('the drag tracks both ends', ctx.state().x0 === 10 && ctx.state().x1 === 190, JSON.stringify(ctx.state()));
  await ctx.up();
  const g = posted.find(p => p[0] === '/api/gradient');
  ok('one gradient posted', !!g && posted.filter(p => p[0] === '/api/gradient').length === 1);
  ok('with the drag geometry', g[1].x0 === 10 && g[1].y0 === 50 && g[1].x1 === 190 && g[1].y1 === 50, JSON.stringify(g[1]));
  ok('and the panel settings', g[1].kind === 'linear' && g[1].opacity === 1 && g[1].to_transparent === false, JSON.stringify(g[1]));
  ok('colours as rgb triples', Array.isArray(g[1].color) && g[1].color.length === 3);
  ok('the drag state is cleared', ctx.state() === null);

  console.log('\nShift snaps the direction to 45 degrees');
  ctx.down({ x: 0, y: 0, pointerId: 1 });
  ctx.move({ x: 100, y: 10, shiftKey: true });
  const s = ctx.state();
  ok('a shallow drag snapped to horizontal', Math.abs(s.y1 - s.y0) < 0.001, JSON.stringify(s));
  ctx.move({ x: 100, y: 90, shiftKey: true });
  const s2 = ctx.state();
  ok('a near-diagonal snapped to 45', Math.abs(Math.abs(s2.x1 - s2.x0) - Math.abs(s2.y1 - s2.y0)) < 0.001, JSON.stringify(s2));
  ctx.move({ x: 100, y: 90 });
  ok('without shift it follows the pointer', ctx.state().y1 === 90);
  await ctx.up();

  console.log('\na click that is not a drag does nothing');
  posted.length = 0;
  ctx.down({ x: 40, y: 40, pointerId: 1 });
  ctx.move({ x: 41, y: 41 });
  await ctx.up();
  ok('no gradient posted for a 1px drag', !posted.some(p => p[0] === '/api/gradient'), JSON.stringify(posted));
  ok('and it says what to do instead', posted.some(p => p[0] === 'status' && /drag from one end/.test(p[1])));

  console.log('\nthe panel controls do what they say');
  doc.getElementById('grKind').value = 'radial';
  doc.getElementById('grAlpha').checked = true;
  doc.fire(doc.getElementById('grAlpha'), 'change');
  ok('to-transparent disables the second swatch', doc.getElementById('grTo').disabled === true);
  ok('and labels it unused', /unused/.test(doc.getElementById('grToLbl').textContent), doc.getElementById('grToLbl').textContent);
  posted.length = 0;
  await ctx.doGradient({ x0: 1, y0: 2, x1: 3, y1: 4 });
  const g2 = posted.find(p => p[0] === '/api/gradient');
  ok('the kind reaches the server', g2[1].kind === 'radial');
  ok('so does to-transparent', g2[1].to_transparent === true);
  const from = doc.getElementById('grFrom').value, to = doc.getElementById('grTo').value;
  doc.fire(doc.getElementById('grSwap'), 'click');
  ok('swap exchanges the ends', doc.getElementById('grFrom').value === to && doc.getElementById('grTo').value === from);

  console.log('\nno layer, no gradient');
  ctx.sel = null;
  posted.length = 0;
  ctx.down({ x: 5, y: 5, pointerId: 1 });
  ok('it asks for a layer instead of opening a drag', ctx.state() === null && posted.some(p => p[0] === 'toast'), JSON.stringify(posted));

  console.log('\nthe preview draws the ramp line');
  const dm = JS.slice(JS.indexOf('function drawMarquee(){'));
  const body = dm.slice(0, dm.indexOf('\nfunction '));
  ok('drawMarquee previews the gradient', /if\(gradDrag\)\{/.test(body));
  ok('with a handle at each end', /vctx\.arc\(gradDrag\.x0,gradDrag\.y0,5/.test(body) && /vctx\.arc\(gradDrag\.x1,gradDrag\.y1,5/.test(body));
  ok('the far handle shows transparent when fading out', /grAlpha'\)\.checked\?'rgba\(0,0,0,0\)'/.test(body));
  ok('gradDrag is reset with the other drags', /marquee=null; lasso=null; polyLasso=null; gradDrag=null;/.test(JS));

  console.log('\n' + passes + ' passed, ' + fails + ' failed');
  process.exit(fails ? 1 : 0);
})();
