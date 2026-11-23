"use strict";
const el = id => document.getElementById(id);
const canvas = el("field"), ctx = canvas.getContext("2d");
const colors = ["#aaf1ce","#f0bd7d","#b2cafe","#f5abd3","#dbcc89"];
let map, sim, timeline, timer, animation;
function pause(){ clearInterval(timer); timer=null; }
function draw(previous=[],fraction=1){
  const cell = canvas.width/map.rows[0].length;
  map.rows.forEach((row,y)=>[...row].forEach((tile,x)=>{
    ctx.fillStyle=tile==="#"?"#507078":"#1c3237";
    ctx.fillRect(x*cell+1,y*cell+1,cell-2,cell-2);
  }));
  const [gx,gy]=map.goal;
  ctx.fillStyle="#e9f6ed"; ctx.font="bold "+(cell*.58)+"px sans-serif";
  ctx.textAlign="center";ctx.textBaseline="middle";
  ctx.fillText("◎",(gx+.5)*cell,(gy+.5)*cell);
  sim.positions.forEach((position,i)=>{
    const from=previous[i] || position;
    if (!from || (!position && fraction===1)) return;
    const to=position || map.goal;
    const x=(from[0]+(to[0]-from[0])*fraction+.5)*cell;
    const y=(from[1]+(to[1]-from[1])*fraction+.5)*cell;
    ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x,y,cell*.36,0,2*Math.PI);ctx.fill();
    ctx.fillStyle="#142d32";ctx.font="bold "+cell*.36+"px sans-serif";ctx.fillText(i+1,x,y);
  });
}
function update(){
  el("start").textContent=sim.status==="Завершено" ? "Старт на новой карте" : "Старт";
  el("history").max=timeline.frames.length-1;
  el("history").value=timeline.index;
  el("frame").textContent=timeline.index+" / "+(timeline.frames.length-1);
  el("stats").textContent="Такт "+sim.tick+" · Прибыли "+sim.positions.filter(p=>!p).length+" / "+sim.positions.length;
  el("log").textContent=[sim.status,...sim.events].join(" · ");
}
function reset(){
  pause();cancelAnimationFrame(animation);
  timeline=new Swarm.Timeline(map,Number(el("count").value),el("policy").value);
  sim=timeline.current;
  el("comparison").textContent="";
  draw();update();
}
function loadMap(next, count){
  Swarm.validate(next);
  const selected=count ?? (Number(el("count").value)||5);
  el("guide").textContent="";
  map=next;
  el("count").replaceChildren(...map.starts.map((_,i)=>new Option(String(i+1),String(i+1))));
  el("count").value=Math.min(selected,map.starts.length);
  canvas.width=map.rows[0].length*48;canvas.height=map.rows.length*48;
  reset();
}
function advance(){
  if (!["Готово","Движение"].includes(sim.status)){pause();return;}
  cancelAnimationFrame(animation);
  const previous=sim.positions, begin=performance.now();
  sim=timeline.next();update();
  function frame(now){
    const f=Math.min(1,(now-begin)/180);
    draw(previous,f);
    if(f<1) animation=requestAnimationFrame(frame);
  }
  animation=requestAnimationFrame(frame);
  if(sim.status==="Завершено") pause();
}
function newMap(){
  let seed=crypto.getRandomValues(new Uint32Array(1))[0];
  let next=Swarm.generate(seed);
  while(map && next.rows.join("")===map.rows.join("")){
    seed=(seed+1)>>>0;
    next=Swarm.generate(seed);
  }
  el("seed").value=seed;
  loadMap(next);
}
el("fresh").onclick=newMap;
el("start").onclick=()=>{
  if(sim.status==="Завершено") newMap();
  if(!timer && ["Готово","Движение"].includes(sim.status)){
    timer=setInterval(advance,300);
    advance();
  }
};
el("step").onclick=()=>{pause();advance();};
el("stop").onclick=()=>{pause();if(sim.status==="Движение") el("log").textContent="Пауза · Такт "+sim.tick;};
el("reset").onclick=reset;el("count").onchange=reset;
el("gen").onclick=()=>{if(el("seed").reportValidity() && el("seed").value!=="")loadMap(Swarm.generate(Number(el("seed").value)));};
el("example").onclick=()=>loadMap(demo);
el("file").onchange=async e=>{
  const file=e.target.files[0];if(!file)return;
  pause();
  try{loadMap(JSON.parse(await file.text()));}
  catch(err){el("log").textContent="Ошибка карты. "+err.message;}
  e.target.value="";
};
canvas.onclick=e=>{
  const box=canvas.getBoundingClientRect();
  const p=[Math.floor((e.clientX-box.left)/box.width*map.rows[0].length),Math.floor((e.clientY-box.top)/box.height*map.rows.length)];
  if(map.rows[p[1]]?.[p[0]]==="." && !map.starts.some(s=>String(s)===String(p))) loadMap({...map,goal:p});
};
function seek(index){
  pause();cancelAnimationFrame(animation);
  sim=timeline.seek(index);draw();update();
}
el("history").oninput=()=>seek(Number(el("history").value));
el("replay").onclick=()=>seek(0);
el("policy").onchange=reset;
el("compare").onclick=()=>{
  const count=Number(el("count").value);
  const scores=["id","waiting"].map(policy=>{
    const run=new Swarm.Simulation(map,count,policy);
    while(["Готово","Движение"].includes(run.status)) run.step();
    if(run.status==="Пути нет") return run.status;
    return run.tick+" такт., "+run.waits+" ожид.";
  });
  el("comparison").textContent="По номеру — "+scores[0]+". Кто дольше ждёт — "+scores[1]+".";
};
const shows = {
  1: [5,"Скачайте JSON и найдите строки поля, координаты стартов и общий выход. Архитектура описана в README."],
  2: [1,"Нажмите «Старт» и покажите обход стены. Для проверки недостижимого выхода откройте maps/unreachable.json."],
  3: [3,"Нажмите «Шаг» и проследите за номерами трёх участников. У каждого своя стартовая клетка."],
  4: [5,"Нажмите «До первого ожидания», изучите причину и сравните правила очередности."]
};
function openShow(){
  const number=Number(el("showcase").value);
  const [count,guide]=shows[number];
  el("policy").value="id";
  if(number===4) el("seed").value=15;
  const next=number===4 ? Swarm.generate(15) : demo;
  // Для одиночного примера кратчайший путь должен огибать стену.
  const selected=number===2 ? {...next,starts:[[2,2]],goal:[17,2]} : next;
  loadMap(selected,count);
  el("guide").textContent=guide;
}
el("openShow").onclick=openShow;
el("saveMap").onclick=()=>{
  const blob=new Blob([JSON.stringify(map,null,2)],{type:"application/json"});
  const url=URL.createObjectURL(blob);
  const link=document.createElement("a");
  link.href=url;link.download="swarm-map.json";link.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
};
el("conflict").onclick=()=>{
  pause();cancelAnimationFrame(animation);
  timeline.seek(0);
  while (["Готово","Движение"].includes(timeline.current.status)){
    const next=timeline.next();
    if (next.events.some(message=>message.includes(" ждёт "))) break;
  }
  seek(timeline.index);
  if(sim.status==="Завершено" && sim.waits===0) el("log").textContent+=" · Никому не пришлось ждать.";
};
newMap();
