// R74 step 03: the shape tools, driven against the DOM shim.
//
// Two things are proved by RUNNING the shipped code, not by matching its
// text: the Shift-drag straight line (whose preview call was a phantom
// function of mine that `node --check` happily accepted -- the whole reason
// this file exists), and the Stroke outline / Fill buttons in the Select
// panel, which must read the Brush panel like every other tool that paints.
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

const doc = makeDocument(
  '<input id="bColor" value="#ff0000"><input id="bSize" value="20"><input id="bOp" value="90">'
  + '<input id="bHard" value="70"><input id="bLoad" value="60"><input id="bLive" type="checkbox">'
  + '<input id="bSmooth" value="0"><select id="bRail"><option value="" selected></option></select>'
  + '<select id="bMedia"><option value="water" selected>water</option></select>'
  + '<button id="selStroke"></button><button id="selFill"></button>'
  + '<input id="selStrokeIn" type="checkbox">');
doc.getElementById('bMedia').value = 'water';
['bColor:#ff0000', 'bSize:20', 'bOp:90', 'bHard:70', 'bLoad:60', 'bSmooth:0'].forEach(p => {
  const [id, v] = p.split(':'); doc.getElementById(id).value = v;
});

const calls = [];
const drew = [];
const src = [
  grab(/function hex2rgb\(h\)\{[^\n]*\n/, 'hex2rgb'),
  grab(/function brushBody\(\)\{[^\n]*\n[^\n]*\n/, 'brushBody'),
  'let stroke=null, smoothPt=null;',
  // the Shift-line branch, lifted verbatim out of the pointermove handler
  'function move(e){ let p=canvasXY(e); p=[p[0],p[1],penPressure(e)];\n'
    + grab(/ +if\(e\.shiftKey&&\(tool==='brush'\|\|tool==='erase'\)&&stroke\.length\)\{[\s\S]*?\n {2}\}/, 'shift line')
    + '\n  stroke.push(p); return "freehand"; }',
  grab(/\$\('selStroke'\)\.onclick=async\(\)=>\{[\s\S]*?\n\};/, 'selStroke'),
  grab(/\$\('selFill'\)\.onclick=async\(\)=>\{[\s\S]*?\n\};/, 'selFill'),
].join('\n');

const ctx = vm.createContext({
  document: doc, console, setTimeout, Math, Element, JSON,
  $: id => doc.getElementById(id),
  tool: 'brush', sel: 'L3', activeSel: 'S1', J: {},
  canvasXY: e => [e.x, e.y], penPressure: () => 1,
  drawLocalStroke: () => drew.push('local'),
  previewDot: () => drew.push('dot'),
  repaintCanvas() {}, refresh() {}, splineById: () => null,
  setStatus: m => calls.push(['status', m]),
  toast: m => calls.push(['toast', m]),
  api: async (url, opt) => {
    calls.push([url, JSON.parse(opt.body)]);
    return url === '/api/stroke_selection' ? { ok: true, rings: 1 } : { ok: true, pixels: 10 };
  },
});
vm.runInContext(src + '\nthis.move=move;this.setStroke=s=>{stroke=s;};this.getStroke=()=>stroke;', ctx);

(async () => {
  console.log('\nShift draws a straight line');
  ctx.setStroke([[10, 10, 1]]);
  drew.length = 0;
  ctx.move({ x: 50, y: 30, shiftKey: true });
  let s = ctx.getStroke();
  ok('the stroke keeps exactly two points', s.length === 2, JSON.stringify(s));
  ok('the first is the origin', s[0][0] === 10 && s[0][1] === 10);
  ok('the second is the pointer', s[1][0] === 50 && s[1][1] === 30);
  ok('the preview was actually drawn', drew.length === 1, JSON.stringify(drew));
  ctx.move({ x: 90, y: 12, shiftKey: true });
  s = ctx.getStroke();
  ok('dragging further replaces the end, never appends', s.length === 2 && s[1][0] === 90, s.length);

  console.log('\nShift+Alt locks the angle to 45');
  ctx.setStroke([[0, 0, 1]]);
  ctx.move({ x: 100, y: 10, shiftKey: true, altKey: true });
  let e1 = ctx.getStroke()[1];
  ok('a shallow drag snaps to horizontal', Math.abs(e1[1]) < 1e-9, JSON.stringify(e1));
  ctx.setStroke([[0, 0, 1]]);
  ctx.move({ x: 100, y: 90, shiftKey: true, altKey: true });
  e1 = ctx.getStroke()[1];
  ok('a near-diagonal snaps to 45', Math.abs(Math.abs(e1[0]) - Math.abs(e1[1])) < 1e-6, JSON.stringify(e1));

  console.log('\nwithout Shift it is an ordinary freehand stroke');
  ctx.setStroke([[0, 0, 1]]);
  ctx.move({ x: 5, y: 5 });
  ctx.move({ x: 9, y: 9 });
  ok('points accumulate', ctx.getStroke().length === 3, ctx.getStroke().length);

  console.log('\nShift is a BRUSH gesture only');
  // with another tool active the branch must not fire at all: the marquee
  // has its own Shift (square/circle) and the gradient has its own (45 deg),
  // so a straight-line rewrite of the point list there would be a bug.
  ctx.tool = 'rect';
  ctx.setStroke([[0, 0, 1]]);
  const via = ctx.move({ x: 40, y: 40, shiftKey: true });
  ok('the branch is skipped for a selection tool', via === 'freehand', String(via));
  ok('and the point was appended, not substituted', ctx.getStroke().length === 2 && ctx.getStroke()[1][0] === 40);
  ctx.tool = 'erase';
  ctx.setStroke([[0, 0, 1], [3, 3, 1]]);
  const viaE = ctx.move({ x: 60, y: 20, shiftKey: true });
  ok('but the eraser gets it too', viaE === undefined && ctx.getStroke().length === 2 && ctx.getStroke()[1][0] === 60);
  ctx.tool = 'brush';

  console.log('\nStroke outline reads the Brush panel');
  calls.length = 0;
  await doc.getElementById('selStroke')._onclick();
  const st = calls.find(c => c[0] === '/api/stroke_selection');
  ok('it posts the stroke route', !!st);
  ok('with the active layer and selection', st[1].layer === 'L3' && st[1].selection === 'S1');
  ok('the brush colour', JSON.stringify(st[1].color) === JSON.stringify([1, 0, 0]), JSON.stringify(st[1].color));
  ok('the brush width', st[1].radius === 12, st[1].radius);
  ok('the brush opacity and hardness', Math.abs(st[1].opacity - 0.9) < 1e-9 && Math.abs(st[1].hardness - 0.7) < 1e-9);
  ok('and the brush MEDIA -- a watercolour rectangle in one click', st[1].media === 'water', JSON.stringify(st[1]));
  ok('inside is off by default', st[1].inside === false);
  doc.getElementById('selStrokeIn').checked = true;
  calls.length = 0;
  await doc.getElementById('selStroke')._onclick();
  ok('the inside toggle reaches the server', calls.find(c => c[0] === '/api/stroke_selection')[1].inside === true);

  console.log('\nFill reads it too');
  calls.length = 0;
  await doc.getElementById('selFill')._onclick();
  const fl = calls.find(c => c[0] === '/api/fill_selection');
  ok('it posts the fill route', !!fl);
  ok('with the colour and opacity', JSON.stringify(fl[1].color) === JSON.stringify([1, 0, 0]) && Math.abs(fl[1].opacity - 0.9) < 1e-9);

  console.log('\nneither works without a layer');
  ctx.sel = null;
  calls.length = 0;
  await doc.getElementById('selStroke')._onclick();
  await doc.getElementById('selFill')._onclick();
  ok('both ask for a layer instead of posting', !calls.some(c => String(c[0]).startsWith('/api')) && calls.filter(c => c[0] === 'toast').length === 2, JSON.stringify(calls));

  console.log('\n' + passes + ' passed, ' + fails + ' failed');
  process.exit(fails ? 1 : 0);
})();
