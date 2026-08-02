// A very small DOM good enough to drive leStudio's toolbar / side-tab code in
// Node. Not a browser -- just enough of the surface those functions touch, so
// we can exercise the shipped source instead of eyeballing it.
'use strict';
const VOID = new Set(['input', 'img', 'br', 'hr', 'meta', 'link', 'source', 'area', 'col']);

class ClassList {
  constructor(el) { this.el = el; }
  get _s() { return (this.el.attrs.class || '').split(/\s+/).filter(Boolean); }
  _set(a) { this.el.attrs.class = a.join(' '); }
  contains(c) { return this._s.includes(c); }
  add(...cs) { const s = this._s; cs.forEach(c => { if (!s.includes(c)) s.push(c); }); this._set(s); }
  remove(...cs) { this._set(this._s.filter(x => !cs.includes(x))); }
  toggle(c, on) { const has = this.contains(c); const want = on === undefined ? !has : !!on; want ? this.add(c) : this.remove(c); return want; }
  get value() { return this.el.attrs.class || ''; }
}

class Element {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.attrs = {}; this.children = []; this.parentNode = null;
    this.classList = new ClassList(this); this.style = {}; this._lis = { cap: {}, bub: {} };
    this.textContent = '';
  }
  get id() { return this.attrs.id || ''; }
  set id(v) { this.attrs.id = v; }
  get dataset() {
    const d = {};
    for (const k in this.attrs) if (k.startsWith('data-')) d[k.slice(5).replace(/-(\w)/g, (m, c) => c.toUpperCase())] = this.attrs[k];
    return d;
  }
  getAttribute(n) { return this.attrs[n] === undefined ? null : this.attrs[n]; }
  setAttribute(n, v) { this.attrs[n] = String(v); }
  appendChild(c) { c.parentNode = this; this.children.push(c); return c; }
  get descendants() { const out = []; const walk = n => n.children.forEach(c => { out.push(c); walk(c); }); walk(this); return out; }
  matches(sel) { return sel.split(',').some(s => matchCompound(this, s.trim().split(/\s+/).pop())) && matchSel(this, sel); }
  closest(sel) { let n = this; while (n) { if (n.matches && n.matches(sel)) return n; n = n.parentNode; } return null; }
  querySelectorAll(sel) { return this.descendants.filter(e => matchSel(e, sel)); }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
  addEventListener(t, fn, cap) { const b = cap ? this._lis.cap : this._lis.bub; (b[t] = b[t] || []).push(fn); }
  // no layout engine here, so tests set _rect on the elements they care about
  getBoundingClientRect() { return Object.assign({ left: 0, top: 0, right: 0, bottom: 0, width: 0, height: 0 }, this._rect || {}); }
  setPointerCapture() {}
  releasePointerCapture() {}
  contains(n) { for (let x = n; x; x = x.parentNode) if (x === this) return true; return false; }
  get value() { return this.attrs.value === undefined ? '' : this.attrs.value; }
  set value(v) { this.attrs.value = String(v); }
  dispatchEvent(ev) { return this.dispatch(ev.type, ev); }
  set innerHTML(v) { if (v === '') this.children = []; }
  get innerHTML() { return ''; }
  set onclick(fn) { this._onclick = fn; }
  get onclick() { return this._onclick; }
  // event dispatch with a real capture-then-bubble path
  dispatch(type, ev) {
    ev = Object.assign({ type, target: this, defaultPrevented: false, _stop: false }, ev || {});
    ev.preventDefault = () => { ev.defaultPrevented = true; };
    ev.stopPropagation = () => { ev._stop = true; };
    const path = []; for (let n = this; n; n = n.parentNode) path.push(n);
    for (let i = path.length - 1; i >= 0 && !ev._stop; i--) (path[i]._lis.cap[type] || []).forEach(f => !ev._stop && f.call(path[i], ev));
    for (let i = 0; i < path.length && !ev._stop; i++) {
      (path[i]._lis.bub[type] || []).forEach(f => !ev._stop && f.call(path[i], ev));
      if (type === 'click' && path[i]._onclick && !ev.defaultPrevented && !ev._stop) path[i]._onclick.call(path[i], ev);
    }
    return ev;
  }
}

// --- selector matching: #id .cls tag [a="v"], compounds, descendant combinator
function matchCompound(el, part) {
  if (!part) return false;
  const toks = part.match(/^[a-zA-Z]+|#[\w-]+|\.[\w-]+|\[[^\]]+\]/g) || [];
  if (!toks.length) return false;
  return toks.every(t => {
    if (t[0] === '#') return el.id === t.slice(1);
    if (t[0] === '.') return el.classList.contains(t.slice(1));
    if (t[0] === '[') { const m = t.slice(1, -1).match(/^([\w-]+)(?:=["']?([^"']*)["']?)?$/); if (!m) return false; return m[2] === undefined ? el.getAttribute(m[1]) !== null : el.getAttribute(m[1]) === m[2]; }
    return el.tagName === t.toUpperCase();
  });
}
function matchSel(el, sel) {
  return sel.split(',').some(one => {
    const parts = one.trim().split(/\s+/);
    if (!matchCompound(el, parts.pop())) return false;
    let n = el.parentNode;
    for (let i = parts.length - 1; i >= 0; i--) {
      let ok = false;
      while (n) { if (matchCompound(n, parts[i])) { ok = true; n = n.parentNode; break; } n = n.parentNode; }
      if (!ok) return false;
    }
    return true;
  });
}

// --- tolerant HTML parser (enough for this markup)
function parse(html) {
  const root = new Element('div'); let cur = root;
  const re = /<!--[\s\S]*?-->|<\/([a-zA-Z][\w-]*)\s*>|<([a-zA-Z][\w-]*)((?:\s+[\w-]+(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'>]+))?)*)\s*(\/?)>|([^<]+)/g;
  let m;
  while ((m = re.exec(html))) {
    if (m[0].startsWith('<!--')) continue;
    if (m[1]) { if (cur.parentNode) cur = cur.parentNode; continue; }
    if (m[2]) {
      const el = new Element(m[2]);
      const ar = /([\w-]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'>]+)))?/g; let a;
      while ((a = ar.exec(m[3] || ''))) el.attrs[a[1]] = a[2] !== undefined ? a[2] : (a[3] !== undefined ? a[3] : (a[4] !== undefined ? a[4] : ''));
      cur.appendChild(el);
      if (!VOID.has(m[2].toLowerCase()) && !m[4]) cur = el;
    } else if (m[5] && m[5].trim()) cur.textContent += m[5].trim();
  }
  return root;
}

function makeDocument(html) {
  const body = parse(html);
  const doc = {
    body,
    getElementById(id) { return body.descendants.find(e => e.id === id) || null; },
    querySelectorAll(s) { return body.querySelectorAll(s); },
    querySelector(s) { return body.querySelector(s); },
    createElement(t) { return new Element(t); },
    addEventListener(t, fn) { (doc._lis[t] = doc._lis[t] || []).push(fn); },
    _lis: {},
    // dispatch on an element, then let document-level listeners see it
    fire(el, type, ev) {
      const e = el.dispatch(type, ev);
      if (!e._stop) (doc._lis[type] || []).forEach(f => f(e));
      return e;
    },
  };
  return doc;
}
module.exports = { makeDocument, Element };
