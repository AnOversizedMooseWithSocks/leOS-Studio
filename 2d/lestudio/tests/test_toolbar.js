// Drives the SHIPPED toolbar / side-tab JavaScript out of index.html against a
// tiny DOM. The point is to prove behaviour, not to re-read the source.
'use strict';
const fs = require('fs'), vm = require('vm'), path = require('path');
const { makeDocument } = require('./domshim.js');

const HTML = fs.readFileSync(process.env.LS_HTML || path.join(__dirname, '../src/lestudio/static/index.html'), 'utf8');
const JS = HTML.split('<script>')[1].split('</script>')[0];

let fails = 0, passes = 0;
const ok = (name, cond, extra) => {
  if (cond) { passes++; console.log('  ok   ' + name); }
  else { fails++; console.log('  FAIL ' + name + (extra ? '  -> ' + extra : '')); }
};

// --- pull only the markup we exercise, so the shim stays small
const slice = (open, close) => HTML.slice(HTML.indexOf(open), HTML.indexOf(close));
const markup = slice('<div id="toolbar">', '<div id="stage">') + slice('<div id="side">', '<div id="shortcutsBack"');
const doc = makeDocument(markup);

// --- lift the pieces of the shipped script we want to run
function grab(re, label) { const m = JS.match(re); if (!m) throw new Error('could not find ' + label); return m[0]; }
const src = [
  grab(/const TABGROUP=\{[^}]*\};/, 'TABGROUP'),
  grab(/const tabState=\{[^}]*\};/, 'tabState'),
  grab(/function sideTab\(name\)\{[\s\S]*?\n\}/, 'sideTab'),
  grab(/document\.querySelectorAll\('\.sidetabs button'\)[^\n]*\n/, 'sidetab wiring'),
  grab(/const TOOLBTN=\{[^}]*\};/, 'TOOLBTN'),
  grab(/function closeToolGroups[\s\S]*?\n\}/, 'closeToolGroups'),
  grab(/function openToolGroup[\s\S]*?\n\}/, 'openToolGroup'),
  grab(/function syncToolRail[\s\S]*?\n\}/, 'syncToolRail'),
  grab(/Object\.entries\(TOOLBTN\)\.forEach[\s\S]*?\n\}\);/, 'tool button wiring'),
  grab(/\(function toolGroups\(\)\{[\s\S]*?\n\}\)\(\);/, 'toolGroups IIFE'),
].join('\n');

const picked = [];
const ctx = vm.createContext({
  document: doc, console, setTimeout, clearTimeout, Element: require('./domshim.js').Element,
  $: id => doc.getElementById(id),
  // stand-in for the real setTool: record it, then do the one thing the real
  // setTool does to the rail (which is the contract under test)
  setTool: t => { picked.push(t); ctx.syncToolRail(t); },
});
vm.runInContext(src + '\nthis.syncToolRail=syncToolRail;this.sideTab=sideTab;this.TOOLBTN=TOOLBTN;', ctx);

// ============================ 1. every tool button is live ==================
console.log('\ntool buttons respond to clicks');
const TOOLS = ctx.TOOLBTN;
for (const [tool, id] of Object.entries(TOOLS)) {
  const b = doc.getElementById(id);
  if (!b) { ok(id + ' exists', false); continue; }
  picked.length = 0;
  b.classList.add('cur');            // simulate it being the visible one
  doc.fire(b, 'click');
  ok(`${id} -> setTool('${tool}')`, picked[0] === tool, 'got ' + JSON.stringify(picked));
}

// ============================ 2. rail stays in sync =========================
console.log('\nrail reflects the active tool');
ctx.syncToolRail('brush');           // known starting state for the paint group
ctx.syncToolRail('wand');
const selGrp = doc.querySelector('.tgrp[data-grp="select"]');
ok('wand becomes the visible button in its group',
  doc.getElementById('tWand').classList.contains('cur'));
ok('only one .cur per group',
  selGrp.querySelectorAll('button').filter(b => b.classList.contains('cur')).length === 1);
ctx.syncToolRail('obj');
ok('switching within a group moves .cur',
  doc.getElementById('tObj').classList.contains('cur') && !doc.getElementById('tWand').classList.contains('cur'));
ok('other groups are untouched', doc.getElementById('tBrush').classList.contains('cur'));

// ============================ 3. flyout open / close ========================
console.log('\nflyout behaviour');
const paint = doc.querySelector('.tgrp[data-grp="paint"]');
const caret = paint.querySelector('.tgc');
doc.fire(caret, 'pointerdown');
ok('caret opens the group', paint.classList.contains('open'));
ok('opening one group closes others', !selGrp.classList.contains('open'));

doc.fire(selGrp.querySelector('.tgc'), 'pointerdown');
ok('opening another group closes the first', !paint.classList.contains('open') && selGrp.classList.contains('open'));

// clicking a member of an open group selects it and closes the flyout
picked.length = 0;
const eBtn = doc.getElementById('tEllipse');
doc.fire(eBtn, 'click');
ok('clicking a flyout member selects that tool', picked[0] === 'ellipse', JSON.stringify(picked));
ok('selecting closes the flyout', !selGrp.classList.contains('open'));

// pointerdown outside any group closes everything
doc.fire(paint.querySelector('.tgc'), 'pointerdown');
ok('reopened for the outside-click test', paint.classList.contains('open'));
doc.fire(doc.getElementById('side'), 'pointerdown');
ok('pointerdown outside closes the flyout', !paint.classList.contains('open'));

// pointerdown INSIDE a group must not close it
doc.fire(paint.querySelector('.tgc'), 'pointerdown');
doc.fire(doc.getElementById('tSmudge'), 'pointerdown');
ok('pointerdown inside the flyout keeps it open', paint.classList.contains('open'));
doc.fire(doc.getElementById('side'), 'pointerdown');

// solo groups never open
const solo = doc.querySelector('.tgrp[data-grp="path"]');
ok('single-tool group is marked solo', solo.classList.contains('solo'));
ctx.openToolGroup(solo);
ok('solo group refuses to open', !solo.classList.contains('open'));

// ============================ 4. long-press ================================
console.log('\nclick-and-hold');
(async () => {
  picked.length = 0;
  const b = doc.getElementById('tBrush');
  doc.fire(b, 'pointerdown');
  await new Promise(r => setTimeout(r, 400));
  ok('holding 350ms opens the group', paint.classList.contains('open'));
  doc.fire(b, 'pointerup');
  doc.fire(b, 'click');                       // the click a real hold produces
  ok('the hold does not also fire the tool', picked.length === 0, JSON.stringify(picked));

  // a short press must still select normally
  paint.classList.remove('open');
  picked.length = 0;
  doc.fire(b, 'pointerdown'); doc.fire(b, 'pointerup'); doc.fire(b, 'click');
  ok('a short click still selects the tool', picked[0] === 'brush', JSON.stringify(picked));
  ok('a short click does not open the flyout', !paint.classList.contains('open'));

  // ========================== 5. side tabs ================================
  console.log('\nside panel tabs');
  const TAB = { layers: 'A', select: 'A', masks: 'A', splines: 'A', brush: 'B', tool: 'B', node: 'B' };
  for (const [tab, g] of Object.entries(TAB)) {
    ctx.sideTab(tab);
    const shown = doc.querySelectorAll('#sidebody' + g + ' .sect').filter(s => s.classList.contains('on'));
    ok(`'${tab}' tab reveals its section`, shown.some(s => s.dataset.tab === tab),
      'visible: ' + JSON.stringify(shown.map(s => s.dataset.tab)));
    const btn = doc.querySelector('#sidetabs' + g + ' button[data-st="' + tab + '"]');
    ok(`'${tab}' tab button highlights`, !!btn && btn.classList.contains('on'));
  }
  // the reported bug: Brush tab came up blank
  ctx.sideTab('brush');
  const bs = doc.getElementById('brushSect');
  ok('Brush section is inside group B', !!bs && bs.closest('#sidebodyB') !== null);
  ok('Brush section is visible when the tab is on', bs.classList.contains('on'));
  ok('Brush controls are present', bs.querySelectorAll('input').length >= 10,
    'inputs=' + bs.querySelectorAll('input').length);
  // and no section is stranded outside the group its tab drives
  ['layers', 'select', 'masks', 'splines'].forEach(t => {
    const s = doc.querySelectorAll('#sidebodyA .sect').filter(x => x.dataset.tab === t);
    ok(`'${t}' section lives in group A`, s.length > 0);
  });

  console.log(`\n${passes} passed, ${fails} failed`);
  process.exit(fails ? 1 : 0);
})();
