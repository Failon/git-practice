const vm=require('vm'),fs=require('fs'),assert=require('assert/strict');
const html=fs.readFileSync(__dirname+'/index.html','utf8');
const elems=Object.fromEntries([...html.matchAll(/id="([^"]+)"/g)].map(([,id])=>[id,{value:'',replaceChildren(){},reportValidity(){return true}}]));
elems.field.getContext=()=>new Proxy({},{get:()=>()=>{}});
elems.showcase.value='4';elems.policy.value='id';elems.seed.value='15';
const c=vm.createContext({crypto:require('node:crypto').webcrypto,document:{getElementById:id=>elems[id]},Option:class{},clearInterval(){},setInterval(){return 1},performance:{now:()=>0},cancelAnimationFrame(){},requestAnimationFrame(fn){fn(200);return 1}});
for(const f of ['core','demo','ui'])vm.runInContext(fs.readFileSync(__dirname+'/'+f+'.js','utf8'),c);
const initialMap=vm.runInContext('map.rows.join("")',c);
elems.fresh.onclick();
assert.notEqual(vm.runInContext('map.rows.join("")',c),initialMap);
elems.seed.value='15';
elems.gen.onclick();elems.compare.onclick();assert.match(elems.comparison.textContent,/29 такт., 14 ожид.*27 такт., 9 ожид/);
elems.start.onclick();elems.stop.onclick();elems.step.onclick();const saved=elems.stats.textContent;
elems.replay.onclick();assert.match(elems.stats.textContent,/Такт 0/);elems.history.value=2;elems.history.oninput();assert.equal(elems.stats.textContent,saved);
elems.policy.value='waiting';elems.policy.onchange();for(let i=0;i<27;i++)elems.step.onclick();assert.match(elems.stats.textContent,/Такт 27 · Прибыли 5 \/ 5/);
const finishedMap=vm.runInContext('map.rows.join("")',c);
elems.start.onclick();elems.stop.onclick();
assert.notEqual(vm.runInContext('map.rows.join("")',c),finishedMap);
const pausedMap=vm.runInContext('map.rows.join("")',c);
elems.start.onclick();elems.stop.onclick();
assert.equal(vm.runInContext('map.rows.join("")',c),pausedMap);
for(const [show,count] of [[1,5],[2,1],[3,3],[4,5]]){
  elems.showcase.value=String(show);elems.openShow.onclick();
  assert.ok(elems.stats.textContent.endsWith(' / '+count));
  assert.ok(elems.guide.textContent.length);
}
elems.conflict.onclick();
assert.match(elems.stats.textContent,/Такт 2/);
assert.match(elems.log.textContent,/Дрон 5 ждёт .*2,8 выбрана дроном 4/);
async function checkImport(){
  const input={files:[{text:async()=>fs.readFileSync(__dirname+'/maps/unreachable.json','utf8')}]};
  await elems.file.onchange({target:input});
  assert.match(elems.log.textContent,/Пути нет/);
  input.files=[{text:async()=>'bad json'}];
  await elems.file.onchange({target:input});
  assert.match(elems.log.textContent,/Ошибка карты/);
  console.log('Showcases, conflict replay, controls, import and error handling passed.');
}
checkImport().catch(err=>{console.error(err);process.exitCode=1});
