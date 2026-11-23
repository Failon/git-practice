// Paired evaluation of frozen policies. Run with node escape/evaluate.js.
const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm");
const game=require("./game");
const weights=vm.runInNewContext(fs.readFileSync(path.join(__dirname,"weights.js"),"utf8")+";EscapeWeights");
const names=["initial","stage1","stage2","stage3","distance"];
const rows=[];
for(let seed=200000;seed<201000;seed++){
  const row={seed};
  for(const name of names){
    const env=game.episode(seed),rng=game.random(seed^32051);
    while(!env.done){
      const options=env.options();
      const action=game.choose(options,name==="distance"?name:weights[name],rng);
      env.step(options[action].p);
    }
    row[name]=env.tick-Number(env.caught);
  }
  rows.push(row);
}
const mean=x=>x.reduce((a,b)=>a+b,0)/x.length;
function paired(a,b){
  const differences=rows.map(row=>row[a]-row[b]);
  const rng=game.random(30317),samples=[];
  for(let i=0;i<5000;i++){
    let sum=0;
    for(let j=0;j<rows.length;j++)sum+=differences[Math.floor(rng()*rows.length)];
    samples.push(sum/rows.length);
  }
  samples.sort((x,y)=>x-y);
  return {mean:mean(differences),ci95:[samples[124],samples[4874]],wins:differences.filter(x=>x>0).length,ties:differences.filter(x=>x===0).length};
}
const scores=Object.fromEntries(names.map(k=>[k,{mean:mean(rows.map(r=>r[k])),full:rows.filter(r=>r[k]===200).length}]));
const report={seeds:[200000,200999],maps:rows.length,scores,beforeAfter:paired("stage3","initial"),againstDistance:paired("stage3","distance")};
fs.writeFileSync(path.join(__dirname,"paired-results.csv"),["seed,"+names.join(","),...rows.map(r=>[r.seed,...names.map(k=>r[k])].join(","))].join("\n")+"\n");
fs.writeFileSync(path.join(__dirname,"evaluation.json"),JSON.stringify(report,null,2)+"\n");
console.log(report);
