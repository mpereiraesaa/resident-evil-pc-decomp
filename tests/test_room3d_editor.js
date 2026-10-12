// Mathematical parity and validation for the offline retouch editor.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const template = fs.readFileSync(path.join(__dirname, '../tools/room3d_editor.html'), 'utf8');
const script = template.match(/<script>([\s\S]*?)<\/script>/)[1];
const context = vm.createContext({});
vm.runInContext(script, context);
const evaluate = code => vm.runInContext(code, context);
const near = (actual, expected) => actual.forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-8));
near(evaluate('movePoint([2,0,0],{scale:[2,1,1],yaw:90,translate:[1,-2,3]},[0,0,0])'), [1,-2,7]);
near(evaluate('movePoint([2,0,0],{},[1,1,1])'), [2,0,0]);
evaluate('var camera={basis:[[1,0,0],[0,1,0],[0,0,1]],origin:[0,0,0],fov:100}');
assert.equal(evaluate('projectEdge([0,0,10],[0,0,100],camera)'), null);
const clipped = evaluate('projectEdge([0,0,100],[100,0,300],camera)');
near(clipped[0], [185,120]);
near(clipped[1], [160+100/3,120]);
for (const value of ['[]', '{missing:{}}', '{lamp:{scale:[0,1,1]}}', '{lamp:{yaw:Infinity}}', '{lamp:{hide:1}}']) {
  assert.throws(() => evaluate(`validateEdits(${value},{lamp:{}})`));
}
assert.equal(evaluate('validateEdits({lamp:{hide:true}},{lamp:{}}).lamp.hide'), true);
console.log('Editor transform, near clipping and validation checks passed');

// Optional real-browser regression test against an already generated editor.
// node tests/test_room3d_editor.js --browser build/room3d-retouch-editor-205.html
if (process.argv[2] === '--browser') {
  const os = require('node:os');
  const cp = require('node:child_process');
  const { pathToFileURL } = require('node:url');
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'room3d-editor-test-'));
  try {
    const source = fs.readFileSync(process.argv[3], 'utf8');
    const probe = `<script>
window.addEventListener('load',()=>{try{
const group=document.getElementById('group'),requested=${JSON.stringify(process.argv[4] || null)};
if(requested){group.value=requested;if(group.value!==requested)throw Error('Requested group missing');}
else group.selectedIndex=Math.min(1,group.options.length-1);
group.dispatchEvent(new Event('change'));
const name=group.value,x=document.querySelector('input[aria-label="Posición X"]');x.value='312.5';x.dispatchEvent(new Event('input'));
if(JSON.parse(document.getElementById('json').textContent)[name].translate[0]!==312.5)throw Error('Translation not saved');
const camera=document.getElementById('camera');camera.selectedIndex=Math.min(1,camera.options.length-1);camera.dispatchEvent(new Event('change'));
const hide=document.getElementById('hide');hide.checked=true;hide.dispatchEvent(new Event('change'));
if(!JSON.parse(document.getElementById('json').textContent)[name].hide)throw Error('Hide not saved');
document.getElementById('reset').click();
if(Object.keys(JSON.parse(document.getElementById('json').textContent)).length)throw Error('Reset failed');
document.body.dataset.test='passed';
}catch(e){document.body.dataset.test='failed: '+e.message;}});
</script>`;
    const file = path.join(temp, 'test.html');
    fs.writeFileSync(file, source.replace('</html>', probe + '</html>'));
    const result = cp.spawnSync('google-chrome', ['--headless', '--disable-gpu', '--no-sandbox',
      '--user-data-dir=' + path.join(temp, 'profile'), '--virtual-time-budget=3000', '--dump-dom',
      pathToFileURL(file).href], { encoding: 'utf8', timeout: 30000, maxBuffer: 32 * 1024 * 1024 });
    assert.equal(result.status, 0, result.stderr);
    assert.ok(result.stdout.includes('data-test="passed"'), result.stdout.match(/data-test="[^"]*"/)?.[0] || 'Browser test did not run');
    console.log('Chrome controls: decimal translation, camera switch, hiding and reset passed');
  } finally {
    fs.rmSync(temp, { recursive: true, force: true });
  }
}
