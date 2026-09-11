import {test} from 'node:test';
import assert from 'node:assert/strict';
import {PatchTracker} from '../lib/external-tracker.ts';
function frame(dx=0,dy=0,duplicate=false){
 const f={width:160,height:120,data:new Uint8Array(160*120).fill(40)};
 let seed=123;const patch=Array.from({length:400},()=>{seed=(seed*1664525+1013904223)>>>0;return 70+seed%180;});
 for(const xx of duplicate?[50+dx,72+dx]:[50+dx])for(let y=0;y<20;y++)for(let x=0;x<20;x++)f.data[(40+dy+y)*160+xx+x]=patch[y*20+x];
 return f;
}
test('tracks a translated patch with signed image displacement',()=>{
 const t=new PatchTracker();assert.ok(t.select(frame(),{x:50,y:40,width:20,height:20}));
 const r=t.update(frame(8,-6));assert.equal(r.status,'tracking');assert.equal(r.dx,8);assert.equal(r.dy,-6);assert.ok(r.score>.99);
});
test('rejects blank selection and marks disappearance as lost',()=>{
 const t=new PatchTracker();const blank={...frame(),data:new Uint8Array(160*120).fill(40)};
 assert.equal(t.select(blank,{x:50,y:40,width:20,height:20}),false);
 t.select(frame(),{x:50,y:40,width:20,height:20});assert.equal(t.update(blank).status,'lost');assert.equal(t.update(frame()).box,null);
});
test('ambiguous duplicate does not claim a target lock',()=>{
 const t=new PatchTracker();t.select(frame(),{x:50,y:40,width:20,height:20});assert.equal(t.update(frame(0,0,true)).status,'lost');
});
