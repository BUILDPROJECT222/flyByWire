// Backend benchmark: same synthetic-drive leaky-rate equation as the CPU reference.
@group(0) @binding(0) var<storage,read> offsets: array<u32>;
@group(0) @binding(1) var<storage,read> sources: array<u32>;
@group(0) @binding(2) var<storage,read> weights: array<f32>;
@group(0) @binding(3) var<storage,read> old: array<f32>;
@group(0) @binding(4) var<storage,read_write> next: array<f32>;
fn update(row:u32, sum:f32) -> f32 {
 let drive=0.12+select(0.0,0.5,row%31u==0u);
 let activation=max(0.0,tanh(1.15*sum+drive));
 return old[row]+0.35*(activation-old[row]);
}
@compute @workgroup_size(64)
fn scalar(@builtin(global_invocation_id) id:vec3<u32>) {
 let row=id.x;
 if(row>=arrayLength(&old)){return;}
 var sum=0.0;
 for(var e=offsets[row];e<offsets[row+1u];e++){sum+=weights[e]*old[sources[e]];}
 next[row]=update(row,sum);
}
var<workgroup> partial:array<f32,64>;
@compute @workgroup_size(64)
fn cooperative(@builtin(workgroup_id) group:vec3<u32>,@builtin(local_invocation_index) lane:u32) {
 let row=group.x+4096u*group.y;
 var sum=0.0;
 if(row<arrayLength(&old)) {
  for(var e=offsets[row]+lane;e<offsets[row+1u];e+=64u){sum+=weights[e]*old[sources[e]];}
 }
 partial[lane]=sum;workgroupBarrier();
 for(var stride=32u;stride>0u;stride/=2u){
  if(lane<stride){partial[lane]+=partial[lane+stride];}
  workgroupBarrier();
 }
 if(lane==0u&&row<arrayLength(&old)){next[row]=update(row,partial[0]);}
}
