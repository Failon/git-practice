"use strict";
const EscapeGame=(()=>{
  function random(seed){
    let state=seed>>>0;
    return ()=>{state=(state+0x6D2B79F5)>>>0;let v=Math.imul(state^(state>>>15),1|state);
      v^=v+Math.imul(v^(v>>>7),61|v);return ((v^(v>>>14))>>>0)/4294967296;};
  }
  const size=9, horizon=200;
  const index=(x,y)=>y*size+x;
  function build(seed){
    const rng=random(seed);
    for(;;){
      const open=Array.from({length:size*size},()=>rng()>.24);
      const adjacent=open.map((free,p)=>{
        if(!free)return [];
        const x=p%size,y=Math.floor(p/size);
        return [[x+1,y],[x-1,y],[x,y+1],[x,y-1]].filter(([a,b])=>a>=0&&a<size&&b>=0&&b<size&&open[index(a,b)]).map(([a,b])=>index(a,b));
      });
      const first=open.indexOf(true);if(first<0)continue;
      const component=new Set([first]);
      for(const p of component)for(const n of adjacent[p])component.add(n);
      if(component.size<42)continue;
      const cells=[...component];
      for(let i=cells.length-1;i>0;i--){const j=Math.floor(rng()*(i+1));[cells[i],cells[j]]=[cells[j],cells[i]];}
      return {adjacent,open:open.map((_,i)=>component.has(i)),starts:cells.slice(0,4)};
    }
  }
  const distance=(a,b)=>Math.abs(a%size-b%size)+Math.abs(Math.floor(a/size)-Math.floor(b/size));
  function episode(seed,count=3){
    const board=build(seed),rng=random(seed^0x5ddc10);
    const state={board,player:board.starts[0],guards:board.starts.slice(1,1+count),tick:0,done:false,caught:false};
    state.options=()=>[state.player,...board.adjacent[state.player]].map(p=>{
      const nearest=Math.min(...state.guards.map(g=>distance(p,g)));
      const nearby=state.guards.filter(g=>board.adjacent[g].includes(p)).length;
      return {p,nearest,features:[Number(nearest===0),nearby/3,Math.min(nearest,5)/5,board.adjacent[p].length/4,Number(p===state.player)]};
    });
    state.step=p=>{
      if(state.done)throw Error("Раунд завершён");
      if(p!==state.player&&!board.adjacent[state.player].includes(p))throw Error("Недоступная клетка");
      const occupied=state.guards.includes(p);
      state.guards=state.guards.map(g=>{const ways=board.adjacent[g];return ways[Math.floor(rng()*ways.length)];});
      state.player=p;state.tick++;
      state.caught=occupied||state.guards.includes(p);
      state.done=state.caught||state.tick===horizon;
      return state.caught?0:1;
    };
    return state;
  }
  const dot=(a,b)=>a.reduce((sum,x,i)=>sum+x*b[i],0);
  function probabilities(options,w){
    const logits=options.map(o=>dot(o.features,w)),peak=Math.max(...logits);
    const exps=logits.map(x=>Math.exp(x-peak)),total=exps.reduce((a,b)=>a+b,0);
    return exps.map(x=>x/total);
  }
  function sample(probs,rng){let u=rng();for(let i=0;i<probs.length-1;i++){u-=probs[i];if(u<=0)return i;}return probs.length-1;}
  function choose(options,policy,rng){
    if(policy==="distance"){
      const score=options.map(o=>o.nearest);
      return score.indexOf(Math.max(...score));
    }
    return sample(probabilities(options,policy),rng);
  }
  return {episode,random,probabilities,sample,choose,horizon,size,distance};
})();
if(typeof module!=="undefined")module.exports=EscapeGame;
