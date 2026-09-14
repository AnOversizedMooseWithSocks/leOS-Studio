// B5: the turntable writes its own .zip (STORE method, hand-rolled CRC32) because the client has no
// zip library. This dumps one from the browser to /tmp so Python's zipfile can verify the CRCs.
//   CHROME=... PUPPETEER_PATH=... node tests/zip_roundtrip.js && python3 -c "import zipfile;print(zipfile.ZipFile('/tmp/ps_turntable_test.zip').testzip())"
const P = process.env.PUPPETEER_PATH || '';
const puppeteer = require(require.resolve('puppeteer', {paths: [P]}));
const fs = require('fs');
const PAGE = 'file://' + require('path').resolve(__dirname, '..', 'index.html');
(async () => {
  const b = await puppeteer.launch({executablePath: process.env.CHROME || undefined, args: ['--no-sandbox']});
  const p = await b.newPage();
  await p.setRequestInterception(true);
  p.on('request', r => r.url().startsWith('file://') ? r.continue() : r.respond({status: 404, body: ''}));
  await p.evaluateOnNewDocument(`const mk=()=>new Proxy(function(){},{get:(t,k)=>{if(k==='then')return undefined;if(!t[k])t[k]=mk();return t[k];},set:(t,k,v)=>{t[k]=v;return true},apply:()=>mk(),construct:()=>mk()});window.THREE=mk();`);
  await p.goto(PAGE, {waitUntil: 'domcontentloaded'});
  await new Promise(r => setTimeout(r, 800));
  const b64 = await p.evaluate(async () => {
    const enc = new TextEncoder();
    const blob = zipStore([
      {name: 'turntable_000.png', bytes: enc.encode('frame zero payload')},
      {name: 'turntable_001.png', bytes: new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0, 1, 2, 3])}]);
    const buf = new Uint8Array(await blob.arrayBuffer());
    let s = ''; for (const x of buf) s += String.fromCharCode(x);
    return btoa(s);
  });
  fs.writeFileSync('/tmp/ps_turntable_test.zip', Buffer.from(b64, 'base64'));
  console.log('wrote /tmp/ps_turntable_test.zip (' + Buffer.from(b64, 'base64').length + ' bytes)');
  await b.close();
})();
