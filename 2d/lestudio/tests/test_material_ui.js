// Live E2E for the material controls: runs the SHIPPED handlers from
// index.html against the DOM shim -- proves the functions exist and behave,
// which node --check cannot (phantom calls pass it).
//
// Media and materials share ONE select (mutually exclusive bodies, one
// choice): a separate Material row measured 24px taller and broke the strict
// no-scroll contract for the Brush half at 1440x900. These tests pin the
// single-select contract: 'mat:'-prefixed values, payload splitting, and the
// colour chip.
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
const { makeDocument } = require('./domshim.js');
const HTML = fs.readFileSync(path.join(__dirname, '../src/lestudio/static/index.html'), 'utf8');
const JS = HTML.split('<script>')[1].split('</script>')[0];
let fails = 0;
const ok = (name, cond, extra) => {
  if (cond) console.log('  ok   ' + name);
  else { fails++; console.log('  FAIL ' + name + (extra ? ' -> ' + extra : '')); }
};
const slice = (open, close) => HTML.slice(HTML.indexOf(open), HTML.indexOf(close));
const markup = slice('<div id="side">', '<div id="shortcutsBack"');
const doc = makeDocument(markup);
function grab(re, label) { const m = JS.match(re); if (!m) throw new Error('missing ' + label); return m[0]; }
const src = [
  grab(/function brushBody[\s\S]*?\n\}/, 'brushBody'),
  grab(/function matByName[\s\S]*?\n\}/, 'matByName'),
  grab(/function syncMatChip[\s\S]*?\n\}/, 'syncMatChip'),
  grab(/\$\('bMedia'\)\.onchange=[\s\S]*?syncMatChip\(\);[^\n]*\};/, 'bMedia handler'),
  grab(/function syncMaterialMenu[\s\S]*?\n\}/, 'syncMaterialMenu'),
  grab(/\$\('bMatCol'\)\.onclick=[\s\S]*?dispatchEvent\(new Event\('change'\)\); \};/, 'chip click'),
].join('\n');
const applyStudio = () => {};   // setups are exercised by the python suite
const state = { materials: [
  { name: 'gold', rough: 0.28, metal: 1, color: [1, 0.78, 0.34] },
  { name: 'brushed_steel', rough: 0.45, metal: 1, color: [0.75, 0.77, 0.8] },
  { name: 'chalk', rough: 1, metal: 0, color: null },
] };
// the shim has no <select> model; graft the DOM surface the shipped code
// touches (querySelector for the optgroup guard works via the shim tree,
// appendChild is native shim, Option/optgroup created through document)
const sel = doc.getElementById('bMedia');
class Option { constructor(text, value) { this.text = text; this.value = value;
  this.tagName = 'OPTION'; this.attrs = { value }; this.children = []; } }
const realCreate = doc.createElement ? doc.createElement.bind(doc) : null;
doc.createElement = tag => {
  const { Element } = require('./domshim.js');
  return new Element(tag);
};
// stubs for things the handler calls that the python suite exercises
const ctx = vm.createContext({ document: doc, console, state, Option,
  applyStudio: () => {}, refreshDock: () => {},
  Event: class { constructor(t){this.type=t;} },
  $: id => doc.getElementById(id) });
vm.runInContext(src + '\nthis.syncMaterialMenu=syncMaterialMenu;this.syncMatChip=syncMatChip;this.brushBody=brushBody;', ctx);

console.log('material controls (single-select design)');
ctx.syncMaterialMenu();
const grp = sel.children.find(c => c.tagName === 'OPTGROUP' && c.getAttribute('label') === 'Materials');
ok('Materials optgroup appended', !!grp);
ok('all materials present', grp.children.length === 3, grp && grp.children.length);
ok('values wear the mat: prefix', grp.children.every(o => String(o.value).startsWith('mat:')));
ok('underscore prettified for display', grp.children.some(o => o.text === 'brushed steel'));
ctx.syncMaterialMenu();
ok('re-population guarded (no duplicate group)',
   sel.children.filter(c => c.tagName === 'OPTGROUP' && c.getAttribute('label') === 'Materials').length === 1);

// payload splitting: the one select yields media XOR material
sel.value = 'oil';
ok('media value -> media payload', JSON.stringify(ctx.brushBody()) === '{"media":"oil"}',
   JSON.stringify(ctx.brushBody()));
sel.value = 'mat:gold';
ok('material value -> material payload', JSON.stringify(ctx.brushBody()) === '{"material":"gold"}',
   JSON.stringify(ctx.brushBody()));
sel.value = '';
ok('empty -> neither', ctx.brushBody().media === undefined && ctx.brushBody().material === undefined);

// Load row follows any body; chip follows material colour
const med = doc.getElementById('bMedia');
sel.value = 'mat:gold'; med.onchange({ target: sel });
ok('Load row shown for a material', doc.getElementById('bLoadRow').style.display === 'flex');
ok('colour chip shown for gold', doc.getElementById('bMatCol').style.display === '');
ok('chip hex is gold', doc.getElementById('bMatCol').getAttribute('data-hex') === '#ffc757',
   doc.getElementById('bMatCol').getAttribute('data-hex'));
sel.value = 'mat:chalk'; med.onchange({ target: sel });
ok('chip hidden for colourless material', doc.getElementById('bMatCol').style.display === 'none');
sel.value = 'oil'; med.onchange({ target: sel });
ok('chip hidden for plain media', doc.getElementById('bMatCol').style.display === 'none');
ok('Load row shown for media too', doc.getElementById('bLoadRow').style.display === 'flex');
sel.value = ''; med.onchange({ target: sel });
ok('Load row hidden when empty', doc.getElementById('bLoadRow').style.display === 'none');

// chip click sets the brush colour and fires the rememberColor path
sel.value = 'mat:gold'; med.onchange({ target: sel });
let changed = false;
doc.getElementById('bColor').dispatchEvent = () => { changed = true; };
doc.getElementById('bMatCol').onclick();
ok('chip sets brush colour', doc.getElementById('bColor').value === '#ffc757',
   doc.getElementById('bColor').value);
ok('chip fires change (rememberColor path)', changed);
console.log(fails ? fails + ' failed' : 'all passed');
process.exit(fails ? 1 : 0);
