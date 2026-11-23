const fs=require("node:fs"),assert=require("node:assert/strict");
const {Simulation,Timeline,generate,validate}=require("./core");
const report=["seed,count,policy,ticks,waits,arrived"];
for(let seed=1;seed<=15;seed++){
  const map=generate(seed);assert.deepEqual(map,generate(seed));
  for(const count of [1,3,4,5]) for(const policy of ["id","waiting"]){
    const s=new Simulation(map,count,policy);
    const timeline=new Timeline(map,count,policy);
    const frames=[JSON.stringify(timeline.current)];
    while(s.tick<1000 && s.status!=="Завершено"){
      const before=s.positions;s.step();
      const frame=timeline.next();
      assert.deepEqual(frame.positions,s.positions);
      assert.deepEqual(frame.events,s.events);
      frames.push(JSON.stringify(frame));
      const destinations=[];
      before.forEach((p,i)=>{
        if(!p)return;
        const q=s.positions[i]||map.goal;
        assert.equal(map.rows[q[1]][q[0]],".");
        assert.ok(Math.abs(p[0]-q[0])+Math.abs(p[1]-q[1])<=1);
        assert.ok(!before.some((other,j)=>j!==i && String(other)===String(q)));
        destinations.push(String(q));
      });
      assert.equal(new Set(destinations).size,destinations.length);
    }
    assert.equal(s.status,"Завершено");timeline.seek(0);
    for (let i=0;i<frames.length;i++){
      const actual=i===0?timeline.current:timeline.next();
      assert.equal(JSON.stringify(actual),frames[i]);
    }
    assert.equal(timeline.next().tick,s.tick);
    report.push([seed,count,policy,s.tick,s.waits,count].join(","));
  }
}
const blocked={rows:[".#."],starts:[[0,0]],goal:[2,0]};
assert.equal(new Simulation(blocked,1).status,"Пути нет");
assert.throws(()=>validate({...blocked,goal:[1,0]}));
fs.writeFileSync(__dirname+"/results.csv",report.join("\n")+"\n");
console.log("120 successful runs and replays, collision checks passed; unreachable/invalid cases passed.");
