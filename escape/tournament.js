const fs=require("node:fs"),path=require("node:path"),vm=require("node:vm"),assert=require("node:assert/strict");
global.EscapeGame=require("./game");
const strategies=require("./strategies"),corner=require("./corner");
const weights=vm.runInNewContext(fs.readFileSync(path.join(__dirname,"weights.js"),"utf8")+";EscapeWeights");
const entrants=["initial","distance","stage1","stage2","stage3","argmax","risk","risk-ppo","graph","graph-space","graph-all"];
function series(first,count){
  const stats=Object.fromEntries(entrants.map(n=>[n,{mean:0,full:0}]));
  for(let seed=first;seed<first+count;seed++){
    for(const n of entrants){
      const env=EscapeGame.episode(seed),rng=EscapeGame.random(seed^32051);
      while(!env.done){const opts=env.options();env.step(opts[strategies.action(env,opts,n,rng,weights)].p);}
      stats[n].mean+=env.tick-Number(env.caught);stats[n].full+=Number(!env.caught);
    }
  }
  for(const s of Object.values(stats))s.mean/=count;
  return stats;
}
const selection=series(300000,500);
const winner=entrants.reduce((a,b)=>selection[a].mean>=selection[b].mean?a:b);
const test=series(400000,1000);
const cornerResults={};
for(const policy of ["random","nearest-corner","one-step","dp"]){
  let sum=0,perfect=0;
  for(let seed=400000;seed<401000;seed++){
    const board=corner.build(seed),rng=EscapeGame.random(seed);let p=0,t=0;
    for(;t<32;t++){
      const moves=board.moves[p];let next;
      if(policy==="dp")next=board.choice[t][p];
      else if(policy==="random")next=moves[Math.floor(rng()*moves.length)];
      else if(policy==="nearest-corner")next=moves.reduce((a,b)=>EscapeGame.distance(b,80)<EscapeGame.distance(a,80)?b:a);
      else {const safe=moves.filter(q=>board.inside(q,t+1));if(!safe.length)break;next=safe[Math.floor(rng()*safe.length)];}
      if(!moves.includes(next)||!board.inside(next,t+1))break;
      p=next;
    }
    if(policy==="dp"){assert.equal(t,32);assert.equal(p,80);}
    sum+=t;perfect+=Number(t===32);
  }
  cornerResults[policy]={mean:sum/1000,perfect};
}
const report={criterion:"mean safe moves",selectionSeeds:[300000,300499],testSeeds:[400000,400999],selection,winner,test,cornerResults};
fs.writeFileSync(path.join(__dirname,"tournament.json"),JSON.stringify(report,null,2)+"\n");console.log(report);
