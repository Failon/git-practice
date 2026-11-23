"use strict";
const el=id=>document.getElementById(id),ctx=el("board").getContext("2d");
let env,rng,timer,nextRound;
function pause(){clearInterval(timer);clearTimeout(nextRound);timer=null;}
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
  if(el("risk").checked&&!env.done){
    ctx.textAlign="center";ctx.font="bold 12px system-ui";
    for(const {p} of env.options()){
      const pct=Math.round(100*EscapeGame.captureRisk(env.board,env.guards,p));
      ctx.fillStyle=pct?"#ffd395":"#d4fff1";
      ctx.fillText(pct+"%",p%9*50+25,Math.floor(p/9)*50+46);
    }
  }
  const safe=env.tick-Number(env.caught);
  el("status").textContent="Пережито ходов: "+safe+" / 200"+(env.caught?" · Пойман":env.done?" · Успешный побег":"");
}
function reset(){
  pause();
  if(!el("seed").value||!el("seed").reportValidity())return;
  const key=Number(el("seed").value);
  const manual=el("policy").value==="manual";
  el("pad").hidden=!manual;
  for(const id of ["start","pause","step"])el(id).hidden=manual;
  env=EscapeGame.episode(key);rng=EscapeGame.random(key^32051);draw();if(manual)el("board").focus({preventScroll:true});
}
function step(){
  if(env.done){pause();return;}
  const name=el("policy").value,opts=env.options();
  const action=EscapeStrategies.action(env,opts,name,rng,EscapeWeights);
  env.step(opts[action].p);draw();
  if(env.done){
    const continuous=Boolean(timer);
    pause();
    if(continuous)nextRound=setTimeout(fresh,1800);
  }
}
function launch(){
  reset();
  if(el("policy").value!=="manual" && el("seed").value && el("seed").validity.valid)el("start").onclick();
}
function fresh(){el("seed").value=crypto.getRandomValues(new Uint32Array(1))[0];launch();}
el("start").onclick=()=>{if(!timer&&!env.done){timer=setInterval(step,100);step();}};
el("step").onclick=()=>{pause();step();};el("pause").onclick=pause;
el("seed").onchange=launch;el("policy").onchange=launch;el("reset").onclick=launch;el("fresh").onclick=fresh;
function move(delta){
  if(env.done||el("policy").value!=="manual")return;
  const p=env.player+delta;
  if(delta!==0&&!env.board.adjacent[env.player].includes(p))return;
  env.step(p);draw();
}
for(const button of el("pad").querySelectorAll("button")){
  button.onclick=()=>{move(Number(button.dataset.delta));el("board").focus({preventScroll:true});};
}
const keys={ArrowLeft:-1,ArrowRight:1,ArrowUp:-9,ArrowDown:9," ":0};
document.addEventListener("keydown",e=>{
  if(el("policy").value!=="manual"||!(e.key in keys)||e.altKey||e.ctrlKey||e.metaKey)return;
  if(e.target.matches("input, select, textarea, button, summary, a"))return;
  e.preventDefault();if(!e.repeat)move(keys[e.key]);
});
el("risk").onchange=()=>{draw();el("board").focus({preventScroll:true});};
fresh();
