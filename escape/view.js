"use strict";
const el=id=>document.getElementById(id),ctx=el("board").getContext("2d");
let env,rng,timer;
function pause(){clearInterval(timer);timer=null;}
function draw(){
  env.board.open.forEach((open,p)=>{
    ctx.fillStyle=open?"#20393d":"#647f83";
    ctx.fillRect(p%9*50+1,Math.floor(p/9)*50+1,48,48);
  });
  function token(p,color,radius,dx=0){
    ctx.fillStyle=color;ctx.beginPath();ctx.arc(p%9*50+25+dx,Math.floor(p/9)*50+25,radius,0,Math.PI*2);ctx.fill();
  }
  const groups=new Map();
  for(const g of env.guards)groups.set(g,(groups.get(g)||0)+1);
  for(const [p,n] of groups)for(let i=0;i<n;i++)token(p,"#ffad73",n===1?14:7,(i-(n-1)/2)*13);
  token(env.player,env.caught?"#eb6583":"#a6efc6",10);
  const safe=env.tick-Number(env.caught);
  el("status").textContent="Пережито ходов: "+safe+" / 200"+(env.caught?" · Пойман":env.done?" · Успешный побег":"");
}
function reset(){
  pause();
  if(!el("seed").value||!el("seed").reportValidity())return;
  const key=Number(el("seed").value);
  env=EscapeGame.episode(key);rng=EscapeGame.random(key^32051);draw();
}
function step(){
  if(env.done){pause();return;}
  const name=el("policy").value,opts=env.options();
  const action=EscapeGame.choose(opts,name==="distance"?name:EscapeWeights[name],rng);
  env.step(opts[action].p);draw();if(env.done)pause();
}
function fresh(){el("seed").value=crypto.getRandomValues(new Uint32Array(1))[0];reset();}
el("start").onclick=()=>{if(!timer&&!env.done){timer=setInterval(step,100);step();}};
el("step").onclick=()=>{pause();step();};el("pause").onclick=pause;
el("seed").onchange=reset;el("policy").onchange=reset;el("reset").onclick=reset;el("fresh").onclick=fresh;
fresh();
