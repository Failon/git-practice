"use strict";
// Node.js only. Episodic PPO-Clip, linear softmax actor and linear critic.
const fs=require("node:fs"),path=require("node:path"),game=require("./game");
const zero=()=>Array(5).fill(0),actor=zero(),critic=zero(),rng=game.random(9451);
const snapshots={initial:actor.slice()},stages=[];
let seed=1,steps=0;
const dot=(a,b)=>a.reduce((sum,v,i)=>sum+v*b[i],0);
for(let guards=1;guards<=3;guards++){
  const begin=steps;
  for(let batch=0;batch<48;batch++){
    const data=[];
    for(let ep=0;ep<24;ep++){
      const env=game.episode(seed++,guards),trajectory=[];
      while(!env.done){
        const opts=env.options(),prob=game.probabilities(opts,actor),a=game.sample(prob,rng);
        const here=opts[0].features,v=[1,here[1],here[2],here[3],env.tick/200];
        const value=dot(v,critic),reward=env.step(opts[a].p);
        trajectory.push({opts,a,v,value,reward,old:prob[a]});steps++;
      }
      let nextValue=0,advantage=0;
      for(let t=trajectory.length-1;t>=0;t--){
        const row=trajectory[t];
        advantage=row.reward+.98*nextValue-row.value+.98*.95*advantage;
        row.target=row.value+advantage;row.adv=advantage;nextValue=row.value;
      }
      data.push(...trajectory);
    }
    const mean=data.reduce((a,r)=>a+r.adv,0)/data.length;
    const sd=Math.sqrt(data.reduce((a,r)=>a+(r.adv-mean)**2,0)/data.length)+1e-8;
    for(const row of data)row.adv=(row.adv-mean)/sd;
    for(let epoch=0;epoch<3;epoch++){
      for(let i=data.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[data[i],data[j]]=[data[j],data[i]];}
      for(let start=0;start<data.length;start+=64){
        const group=data.slice(start,start+64),grad=zero(),dv=zero();
        for(const r of group){
          const prob=game.probabilities(r.opts,actor),ratio=prob[r.a]/r.old;
          if(!((r.adv>0&&ratio>1.2)||(r.adv<0&&ratio<.8))){
            for(let k=0;k<5;k++){
              const avg=prob.reduce((s,p,i)=>s+p*r.opts[i].features[k],0);
              grad[k]+=ratio*r.adv*(r.opts[r.a].features[k]-avg);
            }
          }
          const error=r.target-dot(critic,r.v);
          for(let k=0;k<5;k++)dv[k]+=error*r.v[k];
        }
        for(let k=0;k<5;k++){actor[k]+=.12*grad[k]/group.length;critic[k]+=.025*dv[k]/group.length;}
      }
    }
  }
  snapshots['stage'+guards]=actor.slice();stages.push({guards,episodes:1152,steps:steps-begin});
  console.log(stages.at(-1),actor);
}
const results={};
for(const [name,policy] of Object.entries({...snapshots,distance:'distance'})){
  const ticks=[],rounds=500;let survived=0;
  for(let i=0;i<rounds;i++){
    const key=100000+i,env=game.episode(key),sample=game.random(key^32051);
    while(!env.done){const options=env.options();env.step(options[game.choose(options,policy,sample)].p);}
    ticks.push(env.tick-Number(env.caught));if(!env.caught)survived++;
  }
  results[name]={meanSurvival:ticks.reduce((a,b)=>a+b)/rounds,fullRounds:survived,rounds};
}
const result={trainingSeeds:[1,seed-1],evaluationSeeds:[100000,100499],steps,stages,results};
fs.writeFileSync(path.join(__dirname,'weights.js'),'const EscapeWeights='+JSON.stringify(snapshots)+';\n');
fs.writeFileSync(path.join(__dirname,'metrics.json'),JSON.stringify(result,null,2)+'\n');
console.log(results);
