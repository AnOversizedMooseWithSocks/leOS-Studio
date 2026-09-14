// R73: drives the SHIPPED creature-brush panel JavaScript out of index.html
// against the tiny DOM shim. node --check passes on phantom function calls,
// so this proves the panel's contract by running it: presets load the want
// sliders and light one chip, a hand edit unlights it, the readouts follow
// the sliders, the rules the server receives are what the sliders say, a
// random roll stays in range, and "again" / "retry" repeat the last release
// (retry after an undo, keep-seed with the same seed) without a new click.
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

const slice = (open, close) => HTML.slice(HTML.indexOf(open), HTML.indexOf(close));
const _slice0 = slice;
const markup = slice('<div id="creatureHud"', '<div id="hatchHud"')
  + '<input id="bColor" value="#ff0000"><input id="bOp" value="90"><input id="bHard" value="70">'
  + '<select id="bSel"><option value=""></option></select><input id="bSelInv" type="checkbox">'
  + '<select id="bMedia"><option value="water" selected>water</option></select><input id="bLoad" value="60">';
const doc = makeDocument(markup);
doc.getElementById('bMedia').value = 'water';

function grab(re, label) { const m = JS.match(re); if (!m) throw new Error('could not find ' + label); return m[0]; }
const src = [
  grab(/const CR_PRESETS=\{[\s\S]*?\n\};/, 'CR_PRESETS'),
  grab(/function hex2rgb\(h\)\{[^\n]*\n/, 'hex2rgb'),
  grab(/function rgb2hex\(c\)\{[^\n]*\n/, 'rgb2hex'),
  grab(/function creatureRules\(\)\{[\s\S]*?\n\}/, 'creatureRules'),
  grab(/function creatureReadouts\(\)\{[\s\S]*?\n\}/, 'creatureReadouts'),
  grab(/function creatureShowRules\(r\)\{[\s\S]*?\n\}/, 'creatureShowRules'),
  grab(/function creaturePresetMark\(name\)\{[\s\S]*?\n\}/, 'creaturePresetMark'),
  grab(/function creaturePreset\(name\)\{[\s\S]*?\n\}/, 'creaturePreset'),
  grab(/function creatureRandomise\(\)\{[\s\S]*?\n\}/, 'creatureRandomise'),
  grab(/var CR_LAST=null;/, 'CR_LAST'),
  grab(/function creatureLastLine\(r,preset\)\{[\s\S]*?\n\}/, 'creatureLastLine'),
  grab(/function genCommon\(\)\{[\s\S]*?\n\}/, 'genCommon'),
  grab(/function brushBody\(\)\{[^\n]*\n[^\n]*\n/, 'brushBody'),
  grab(/async function doCreature\(x,y,path,seed\)\{[\s\S]*?\n\}/, 'doCreature'),
  grab(/async function creatureAgain\(retry\)\{[\s\S]*?\n\}/, 'creatureAgain'),
  grab(/\(function wireCreature\(\)\{[\s\S]*?\n\}\)\(\);/, 'wireCreature'),
].join('\n');

const calls = [];
const ctx = vm.createContext({
  document: doc, console, setTimeout, clearTimeout, Element,
  $: id => doc.getElementById(id),
  sel: 'L2', J: {}, sampleMode: () => 'layer',
  setStatus: () => {}, toast: m => calls.push(['toast', m]), refresh: () => {},
  api: async (url, opt) => { const b = opt && opt.body ? JSON.parse(opt.body) : {}; calls.push([url, b]);
    return url === '/api/creature' ? { ok: true, strokes: b.creatures, seconds: b.seconds, seed: b.seed === undefined ? 4242 : b.seed,
      rules: Object.assign({}, b.rules, b.random_rules ? { lines: 0.25, self: 0.5, others: -0.5, light: 0.75, color: -0.25, field: 0.1, wander: 0.3, target: [0.2, 0.4, 0.6] } : {}) } : { ok: true }; },
});
vm.runInContext(src + '\nthis.creaturePreset=creaturePreset;this.creatureRules=creatureRules;this.creatureRandomise=creatureRandomise;this.doCreature=doCreature;this.creatureAgain=creatureAgain;this.CR_PRESETS=CR_PRESETS;', ctx);
const v = id => doc.getElementById(id).value;
const lit = () => doc.querySelectorAll('#crPresets .crp').filter(b => b.classList.contains('active')).map(b => b.dataset.preset);
const out = id => doc.getElementById(id).parentNode.querySelector('output').textContent;

(async () => {
  console.log('\npresets load the sliders and light one chip');
  doc.fire(doc.querySelector('[data-preset="knotter"]'), 'click');
  ok('knotter sets own-trail to seek', v('crSelf') === '90', v('crSelf'));
  ok('knotter lights only its chip', JSON.stringify(lit()) === '["knotter"]', JSON.stringify(lit()));
  ok('readout follows the slider', out('crSelf') === '90', out('crSelf'));
  ok('knotter sets a creature count', v('crCount') === '4');
  doc.fire(doc.querySelector('[data-preset="mazer"]'), 'click');
  ok('maze runner turns solid on', doc.getElementById('crSolid').checked === true);
  ok('maze runner unlights knotter', JSON.stringify(lit()) === '["mazer"]');
  doc.fire(doc.querySelector('[data-preset="surprise"]'), 'click');
  ok('surprise turns random-each-run on', doc.getElementById('crRandom').checked === true);
  doc.fire(doc.querySelector('[data-preset="explorer"]'), 'click');
  ok('explorer turns random back off', doc.getElementById('crRandom').checked === false);

  console.log('\nevery preset is a complete rule set');
  for (const [name, p] of Object.entries(ctx.CR_PRESETS)) {
    if (p.random) continue;
    const need = ['lines', 'self', 'others', 'light', 'color', 'fieldw', 'field', 'wander', 'solid'];
    ok(name + ' names every want', need.every(k => k in p), need.filter(k => !(k in p)).join());
    ok(name + ' keeps wants in -100..100', ['lines', 'self', 'others', 'light', 'color'].every(k => p[k] >= -100 && p[k] <= 100));
  }

  console.log('\na hand edit makes it custom');
  doc.fire(doc.querySelector('[data-preset="moth"]'), 'click');
  const L = doc.getElementById('crLight'); L.value = '40'; doc.fire(L, 'input');
  ok('no chip stays lit after an edit', lit().length === 0, JSON.stringify(lit()));
  ok('readout follows the edit', out('crLight') === '40', out('crLight'));

  console.log('\nthe rules the server gets are what the sliders say');
  const r = ctx.creatureRules();
  ok('light reads 0.4', Math.abs(r.light - 0.4) < 1e-9, r.light);
  ok('own trail reads the moth value', Math.abs(r.self - (-0.4)) < 1e-9, r.self);
  ok('target is an rgb triple', Array.isArray(r.target) && r.target.length === 3);

  console.log('\nrandom roll stays in range');
  for (let i = 0; i < 20; i++) {
    ctx.creatureRandomise();
    const q = ctx.creatureRules();
    if (!['lines', 'self', 'others', 'light', 'color'].every(k => q[k] >= -1 && q[k] <= 1) || q.wander < 0.05 || q.wander > 0.35 || q.field < 0 || q.field > 1) { ok('roll ' + i + ' in range', false, JSON.stringify(q)); break; }
  }
  ok('twenty rolls stayed in range', true);
  ok('a roll is custom', lit().length === 0);

  console.log('\nrelease, again, retry');
  const dis = id => { const e = doc.getElementById(id); return e.disabled || e.getAttribute('disabled') !== null; };
  ok('again/retry start disabled', dis('crAgain') && dis('crRetry'));
  calls.length = 0;
  await ctx.doCreature(100, 80, [[90, 70], [110, 90]]);
  let c = calls.find(x => x[0] === '/api/creature');
  ok('the release posts the panel', !!c && c[1].x === 100 && c[1].path.length === 2 && c[1].media === 'water' && Math.abs(c[1].load - 0.6) < 1e-9, JSON.stringify(c && c[1]).slice(0, 200));
  ok('again/retry enabled after a release', !doc.getElementById('crAgain').disabled && !doc.getElementById('crRetry').disabled);
  ok('the last-run line names the seed', /seed 4242/.test(doc.getElementById('crLast').textContent), doc.getElementById('crLast').textContent);
  calls.length = 0;
  doc.fire(doc.getElementById('crAgain'), 'click'); await new Promise(r => setTimeout(r, 5));
  c = calls.find(x => x[0] === '/api/creature');
  ok('again repeats the spot and the drag', !!c && c[1].x === 100 && c[1].path.length === 2);
  ok('again does not undo', !calls.some(x => x[0] === '/api/undo'));
  ok('again takes a fresh seed', c[1].seed === undefined);
  calls.length = 0;
  doc.getElementById('crKeepSeed').checked = true;
  doc.fire(doc.getElementById('crRetry'), 'click'); await new Promise(r => setTimeout(r, 5));
  ok('retry undoes first', calls[0] && calls[0][0] === '/api/undo', JSON.stringify(calls.map(x => x[0])));
  c = calls.find(x => x[0] === '/api/creature');
  ok('retry with keep-seed sends the same seed', !!c && c[1].seed === 4242, JSON.stringify(c && c[1].seed));

  console.log('\nrandom rules come back into the sliders');
  doc.getElementById('crRandom').checked = true;
  await ctx.doCreature(50, 50, null);
  ok('the drawn light want lands in the slider', v('crLight') === '75', v('crLight'));
  ok('the drawn target lands in the swatch', v('crTarget') === '#336699', v('crTarget'));
  ok('the readout shows it', out('crLight') === '75');

  // R74d: the panel must actually APPEAR. The creature button was the
  // fourth in a collapsed group whose visible face is Scribble, and its
  // advertised key (Shift+L) reached the polygon lasso first -- so there was
  // no way to open this panel at all. Drive the shipped setTool body: the
  // dock mapping and the show/hide sweep decide whether the panel is seen.
  console.log('\npicking the creature shows its panel');
  const hudDoc = makeDocument(
    slice('<div id="creatureHud"', '<div id="hatchHud"')
    + '<div id="gradHud"></div><div id="hatchHud"></div><div id="textileHud"></div>'
    + '<div id="scribHud"></div><div id="trHud"></div><div id="fillHud"></div>'
    + '<div id="textHud"></div><div id="npHud"></div><div id="fxHud"></div>'
    + '<div id="stampHud"></div><div id="strokeSelPanel"></div>'
    + '<div id="toolDockEmpty"></div>');
  const dockSrc = grab(/const dockTool=\{[^\n]*\n[\s\S]{0,400}?\}\);/, 'dock sweep');
  const h = vm.createContext({ document: hudDoc, console, $: id => hudDoc.getElementById(id) });
  for (const t of ['creature', 'gradient', 'scribble']) {
    // fresh scope each time: the lifted body declares `const dockTool`
    vm.runInContext('(function(t){' + dockSrc
      + ';this.shown=["creatureHud","gradHud","scribHud","hatchHud","textileHud"]'
      + '.filter(id=>$(id).style.display&&$(id).style.display!=="none");'
      + '}).call(this,' + JSON.stringify(t) + ');', h);
    const want = { creature: 'creatureHud', gradient: 'gradHud', scribble: 'scribHud' }[t];
    ok(t + ' shows exactly its own panel',
       h.shown.length === 1 && h.shown[0] === want, JSON.stringify(h.shown));
  }
  ok('the creature panel still has its presets after the move',
     hudDoc.querySelectorAll('#crPresets .crp').length === 8,
     hudDoc.querySelectorAll('#crPresets .crp').length);

  console.log('\n' + passes + ' passed, ' + fails + ' failed');
  process.exit(fails ? 1 : 0);
})();
