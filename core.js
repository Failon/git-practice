"use strict";
const Swarm = (() => {
  function validate(map) {
    const {rows, starts, goal} = map;
    if (!Array.isArray(rows) || !rows.length || rows.length > 80 ||
        typeof rows[0] !== "string" || !rows[0].length || rows[0].length > 80 ||
        !rows.every(row => typeof row === "string" && row.length === rows[0].length && /^[.#]+$/.test(row)))
      throw Error("Карта должна быть прямоугольником до 80×80 из точек и решёток.");
    const free = p => Array.isArray(p) && p.length === 2 && p.every(Number.isInteger) && rows[p[1]]?.[p[0]] === ".";
    if (!Array.isArray(starts) || starts.length < 1 || starts.length > 5 || !starts.every(free) || !free(goal))
      throw Error("Проверьте координаты финиша и 1–5 стартов.");
    if (new Set([...starts, goal].map(String)).size !== starts.length+1)
      throw Error("Стартовые клетки и финиш должны различаться.");
  }
  const neighbors = ([x,y]) => [[x+1,y],[x,y+1],[x-1,y],[x,y-1]];
  function distances(rows, goal) {
    const queue = [goal], result = new Map([[String(goal),0]]);
    for (let i=0; i<queue.length; i++) {
      for (const p of neighbors(queue[i])) {
        if (rows[p[1]]?.[p[0]] !== "." || result.has(String(p))) continue;
        result.set(String(p), result.get(String(queue[i]))+1);
        queue.push(p);
      }
    }
    return result;
  }
  class Simulation {
    constructor(map,count,policy="id") {
      validate(map);
      this.goal = map.goal;
      this.dist = distances(map.rows,map.goal);
      this.positions = map.starts.slice(0,count);
      this.policy = policy;
      this.streak = this.positions.map(() => 0);
      this.tick = 0;
      this.waits = 0;
      this.events = [];
      this.status = this.positions.every(p => this.dist.has(String(p))) ? "Готово" : "Пути нет";
    }
    step() {
      if (!["Готово","Движение"].includes(this.status)) return;
      const occupied = new Map();
      const order = [];
      this.positions.forEach((p,i) => {
        if (p) { occupied.set(String(p),i); order.push(i); }
      });
      if (this.policy === "waiting") order.sort((a,b) => this.streak[b]-this.streak[a] || a-b);
      const reserved = new Map();
      const after = this.positions.slice();
      this.events = [];
      for (const i of order) {
        const p = this.positions[i];
        const candidates = neighbors(p).filter(q => this.dist.get(String(q)) === this.dist.get(String(p))-1);
        const next = candidates.find(q => !occupied.has(String(q)) && !reserved.has(String(q)));
        if (!next) {
          const reasons = candidates.map(q => {
            const key = String(q);
            if (occupied.has(key)) return key+" занята дроном "+(occupied.get(key)+1);
            return key+" выбрана дроном "+(reserved.get(key)+1);
          });
          this.events.push("Дрон "+(i+1)+" ждёт ("+reasons.join(", ")+")");
          this.streak[i]++;
          this.waits++;
          continue;
        }
        this.streak[i] = 0;
        reserved.set(String(next),i);
        after[i] = next;
        if (String(next) === String(this.goal)) {
          this.events.push("Дрон "+(i+1)+" прибыл");
          after[i] = null;
        }
      }
      this.positions = after;
      this.tick++;
      this.status = this.positions.some(Boolean) ? "Движение" : "Завершено";
    }
  }
  class Timeline {
    constructor(map,count,policy) {
      this.engine = new Simulation(map,count,policy);
      this.frames = [this.snapshot()];
      this.index = 0;
    }
    snapshot() {
      const {positions,tick,waits,status,events} = this.engine;
      return {positions,tick,waits,status,events};
    }
    get current() { return this.frames[this.index]; }
    seek(index) {
      if (Number.isInteger(index) && index >= 0 && index < this.frames.length) this.index = index;
      return this.current;
    }
    next() {
      if (this.index+1 < this.frames.length) return this.seek(this.index+1);
      if (["Готово","Движение"].includes(this.engine.status)) {
        this.engine.step();
        this.frames.push(this.snapshot());
        this.index++;
      }
      return this.current;
    }
  }
  function generate(seed) {
    let state = seed >>> 0;
    const random = () => ((state=(Math.imul(state,1664525)+1013904223)>>>0)/4294967296);
    const starts = [1,3,5,7,9].map(y=>[1,y]), goal = [18,5];
    for (;;) {
      const grid = Array.from({length:11},()=>Array.from({length:20},()=>random()<0.22?"#":"."));
      for (const [x,y] of [...starts,goal]) grid[y][x]=".";
      const rows = grid.map(row=>row.join(""));
      const dist = distances(rows,goal);
      if (starts.every(p=>dist.has(String(p)))) return {rows,starts,goal};
    }
  }
  return {Simulation,Timeline,generate,validate};
})();
if (typeof module !== "undefined") module.exports = Swarm;
