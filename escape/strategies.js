"use strict";
const EscapeStrategies=(()=>{
  const distances=new WeakMap();
  function table(board){
    if(distances.has(board))return distances.get(board);
    const matrix=board.adjacent.map((_,start)=>{
      const dist=new Int16Array(board.open.length).fill(999),queue=[start];dist[start]=0;
      for(let i=0;i<queue.length;i++)for(const n of board.adjacent[queue[i]])if(dist[n]===999){dist[n]=dist[queue[i]]+1;queue.push(n);}
      return dist;
    });
    distances.set(board,matrix);return matrix;
  }
  function action(env,opts,name,rng,weights){
    if(name==="distance"||weights[name])return EscapeGame.choose(opts,name==="distance"?name:weights[name],rng);
    const matrix=name.includes("graph")?table(env.board):null;
    const scores=opts.map(o=>{
      const risk=EscapeGame.captureRisk(env.board,env.guards,o.p);
      const logit=o.features.reduce((sum,f,i)=>sum+f*weights.stage3[i],0);
      if(name==="risk")return -risk;
      if(name==="argmax")return logit;
      if(name==="risk-ppo")return -1000*risk+logit;
      const d=env.guards.map(g=>matrix[o.p][g]);
      if(name==="graph")return Math.min(...d);
      if(name==="graph-space")return Math.min(...d)+.1*env.board.adjacent[o.p].length;
      if(name==="graph-all")return -d.reduce((sum,v)=>sum+1/(1+v)**2,0);
      throw Error("Unknown policy "+name);
    });
    return scores.indexOf(Math.max(...scores));
  }
  return {action};
})();
if(typeof module!=="undefined")module.exports=EscapeStrategies;
