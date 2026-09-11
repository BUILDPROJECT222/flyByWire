// Independent full-graph graded prototype: dimensionless voltage and rectified release.
// Row-normalized signed contacts, recurrent gain 0.8, configurable tau, dt 5 ms.
// Parameters are engineering assumptions, not pretrained FlyVis parameters.
@group(0) @binding(0) var<storage,read> offsets:array<u32>;
@group(0) @binding(1) var<storage,read> sources:array<u32>;
@group(0) @binding(2) var<storage,read> weights:array<f32>;
@group(0) @binding(3) var<storage,read> old:array<f32>;
@group(0) @binding(4) var<storage,read_write> next:array<f32>;
@group(0) @binding(5) var<storage,read> drive:array<f32>;
@group(0) @binding(6) var<storage,read> bias:array<f32>;
@group(0) @binding(7) var<storage,read> alpha:array<f32>;
var<workgroup> partial:array<f32,64>;
@compute @workgroup_size(64)
fn advance(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32){
 let row=group.x+4096u*group.y;
 var sum=0.0;
 if(row<arrayLength(&old)){
  for(var e=offsets[row]+lane;e<offsets[row+1u];e+=64u){
   sum+=weights[e]*max(0.0,old[sources[e]]);
  }
 }
 partial[lane]=sum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(lane<stride){partial[lane]+=partial[lane+stride];}
  workgroupBarrier();
 }
 if(lane==0u&&row<arrayLength(&old)){
  next[row]=old[row]+alpha[row]*(-old[row]+bias[row]+0.8*partial[0]+drive[row]);
 }
}
