"use strict";
// The three stage cues depend only on the scene clock.
const CornerScene=(()=>{
  const size=9,last=8,end=33;
  const x=p=>p%size,y=p=>Math.floor(p/size);
  function build(seed){
    const rng=EscapeGame.random(seed),open=Array.from({length:81},()=>rng()>.26);
    // Carve a staircase: each pair of steps reaches the next surviving square.
    for(let k=0;k<last;k++){
      open[10*k]=true;open[10*k+(rng()<.5?1:9)]=true;
    }
    open[80]=true;
    const moves=open.map((free,p)=>free?[p,p+1,p+9,p-1,p-9].filter(q=>open[q]&&Math.abs(x(p)-x(q))+Math.abs(y(p)-y(q))<=1):[]);
    const inside=(p,t)=>Math.min(x(p),y(p))>=Math.min(last,Math.floor(t/4));
    // Backward induction: maximize remaining safe moves. Among equal paths
    // prefer approaching the corner gradually instead of waiting there early.
    const value=Array.from({length:end+1},()=>new Int16Array(81));
    const choice=Array.from({length:end},()=>new Int16Array(81).fill(-1));
    for(let t=end-2;t>=0;t--){
      for(let p=0;p<81;p++){
        if(!open[p]||!inside(p,t))continue;
        let bestTie=-Infinity;
        for(const q of moves[p]){
          if(!inside(q,t+1))continue;
          const score=1+value[t+1][q];
          const tie=-Math.abs(x(q)+y(q)-(t+1)/2)+(q!==p?.1:0);
          if(score>value[t][p]||(score===value[t][p]&&tie>bestTie)){
            value[t][p]=score;choice[t][p]=q;bestTie=tie;
          }
        }
      }
    }
    return {open,moves,choice,maxSafe:value[0][0],inside};
  }
  return {build,end};
})();
if(typeof module!=="undefined")module.exports=CornerScene;
