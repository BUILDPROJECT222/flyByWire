"""Loopback-only lab with an explicit, supervised flight API; brain outputs are observational."""
import io
import json
import subprocess
import threading
import time
from pathlib import Path
from contextlib import asynccontextmanager
import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse,Response
from pydantic import BaseModel,Field
from .core import Circuit,Arena,decode,initial_weights,ROOT,DATA,MODEL_VERSION
from .train import train

class Lab:
    def __init__(self):
        self.lock=threading.RLock(); self.circuit=Circuit(); self.weights=initial_weights(self.circuit.feature_size)
        self.running=False; self.status='paused'; self.mode='simulation'; self.condition='intact'; self.seed=11
        self.env=Arena(self.seed); self.action=[0.,0.]; self.trace=[]; self.events=[]; self.compute_ms=0.
        self.training={'status':'idle','history':[]}; self.cancel=threading.Event(); self.last_result=None
        self.comparison=[]; self.job=None; self.shutdown=threading.Event(); self.replay_index=0; self.replay_data=None
        self.recordings=[]; self.live_process=None; self.live_jpeg=None; self.live_light=np.zeros(8); self.live_last=0.
        for p in sorted((ROOT/'recordings').rglob('*.mp4')):
            if p.name in ['takeoff-land.mp4','camera-secondary.mp4','camera-front-restored.mp4'] and p.stat().st_size>0:
                self.recordings.append({'id':str(p.relative_to(ROOT/'recordings')),'name':p.parent.name+' / '+p.stem,'path':p})
        self.recording=self.recordings[0]['id'] if self.recordings else ''
        checkpoints=sorted(p for p in (ROOT/'experiments').glob('*/report.json') if (p.parent/'readout.npy').exists())
        if checkpoints:
            p=checkpoints[-1]; data=json.loads(p.read_text())
            w=np.load(p.parent/'readout.npy')
            if w.shape==self.weights.shape and data.get('model_version')==MODEL_VERSION:
                self.weights=w; self.last_result=data
                ablations=p.parent/'ablations.json'
                if ablations.exists(): self.comparison=json.loads(ablations.read_text())
        self.log('Ready. Hardware outputs are disconnected.')
    def log(self,message):
        self.events.append({'time':time.strftime('%H:%M:%S'),'message':message}); self.events=self.events[-30:]
    def reset(self):
        self.circuit.reset(); self.env=Arena(self.seed); self.trace=[]; self.replay_index=0; self.action=[0.,0.]
    def tick(self):
        start=time.perf_counter()
        if self.mode=='simulation': light=self.env.light
        elif self.mode=='replay':
            if self.replay_data is None or self.replay_index>=len(self.replay_data):
                self.running=False; self.status='replay complete'; return
            light=self.replay_data[self.replay_index]; self.replay_index+=1
        else:
            if time.monotonic()-self.live_last>2:
                self.status='waiting for camera'; return
            light=self.live_light
        f=self.circuit.step(light); self.action=decode(f,self.weights).tolist()
        if self.mode=='simulation':
            self.env.step(self.action)
            if self.env.t>=120:
                self.running=False; self.status='episode complete'; self.log('120 simulated seconds completed.')
        self.compute_ms=(time.perf_counter()-start)*1000
        self.trace.append({'time':self.env.t if self.mode=='simulation' else self.replay_index/10,
                           **self.circuit.activity(),'turn':self.action[0],'forward':self.action[1]})
        self.trace=self.trace[-120:]
    def loop(self):
        while not self.shutdown.is_set():
            then=time.monotonic()
            with self.lock:
                if self.running: self.tick()
            self.shutdown.wait(max(.005,.1-(time.monotonic()-then)))
    def state(self):
        with self.lock:
            return dict(status=self.status,running=self.running,mode=self.mode,condition=self.condition,
                        seed=self.seed,arena=self.env.state(),activity=self.circuit.activity(),
                        rates=np.round(self.circuit.rates,4).tolist(),actions=self.action,
                        trace=self.trace,compute_ms=self.compute_ms,training=self.training,events=self.events,
                        manifest=self.circuit.meta['manifest'],recording=self.recording,
                        replay_time=self.replay_index/10,replay_duration=len(self.replay_data)/10 if self.replay_data is not None else 0,
                        recordings=[{k:v for k,v in r.items() if k!='path'} for r in self.recordings],
                        last_result=self.last_result,comparison=self.comparison,hardware_enabled=False,
                        live_fresh=time.monotonic()-self.live_last<2,live_frame=int(self.live_last*1000))
    def set_replay(self,recording):
        selected=next((r for r in self.recordings if r['id']==recording),None)
        if selected is None: raise ValueError('Unknown recording')
        p=subprocess.run(['ffmpeg','-v','error','-i',str(selected['path']),'-vf','fps=10,scale=8:1',
                          '-pix_fmt','gray','-f','rawvideo','pipe:1'],capture_output=True,timeout=20)
        if p.returncode or not p.stdout: raise ValueError('Could not decode recording')
        self.replay_data=np.frombuffer(p.stdout,dtype=np.uint8).reshape(-1,8)/255.
        self.recording=recording
    def stop_live(self):
        if self.live_process is not None: self.live_process.set()
        self.live_process=None; self.live_last=0.
    def start_live(self):
        # Reuse the sole workspace camera decoder, including during flight recording.
        self.stop_live()
        cancelled=threading.Event(); self.live_process=cancelled
        feed=workspace().feed
        def frames():
            seen=-1
            while not cancelled.wait(.1):
                rgb,jpeg,seq,_,last=feed.snapshot()
                if rgb is None or seq==seen or feed.status()['source']!='live': continue
                seen=seq
                small=np.asarray(Image.fromarray(rgb).resize((160,120)))
                light=small.mean(axis=(0,2)).reshape(8,20).mean(axis=1)/255.
                with self.lock:
                    self.live_light=light; self.live_jpeg=jpeg; self.live_last=last
        threading.Thread(target=frames,daemon=True).start()
    def start_training(self,generations,seed):
        if self.job and self.job.is_alive(): raise ValueError('An experiment is already running')
        self.cancel.clear(); self.training={'status':'training','history':[],'generations':generations}
        self.log(f'Training readout for {generations} generations; circuit weights remain fixed.')
        def work():
            try:
                def update(item):
                    with self.lock: self.training['history'].append(item)
                weights,result=train(generations,seed,update,self.cancel.is_set)
                with self.lock:
                    self.weights=weights; self.last_result=result; self.comparison=[]
                    self.condition='intact'; self.circuit=Circuit()
                    self.training['status']='cancelled' if result['cancelled'] else 'complete'
                    self.reset(); self.log('Readout saved. Held-out evaluation and untrained baseline are available.')
            except Exception as exc:
                with self.lock: self.training['status']='error'; self.log(str(exc))
        self.job=threading.Thread(target=work,daemon=True); self.job.start()
    def compare(self):
        if self.job and self.job.is_alive(): raise ValueError('Wait for the current experiment')
        self.training['status']='comparing'; self.comparison=[]; weights=self.weights.copy()
        def work():
            from .core import evaluate
            try:
                for condition in ['intact','shuffled','disconnected']:
                    result=evaluate(Circuit(condition),weights)
                    with self.lock: self.comparison.append(dict(condition=condition,**result))
                result=evaluate(Circuit(),weights,baseline=True)
                with self.lock:
                    self.comparison.append(dict(condition='direct range baseline',**result)); self.training['status']='complete'
                    self.log('Ablations evaluated on seeds 101–103. Fixed readout; not matched retraining.')
                    if self.last_result:
                        p=ROOT/'experiments'/self.last_result['run_id']/'ablations.json'
                        p.write_text(json.dumps(self.comparison,indent=2))
            except Exception as exc:
                with self.lock: self.training['status']='error'; self.log(str(exc))
        self.job=threading.Thread(target=work,daemon=True); self.job.start()

lab=Lab()
@asynccontextmanager
async def lifespan(app):
    worker=threading.Thread(target=lab.loop,daemon=True); worker.start()
    yield
    flight.close()
    lab.shutdown.set(); lab.cancel.set(); lab.stop_live()
    if full_instance is not None: full_instance.stop.set(); full_instance.thread.join(timeout=5)
    if workspace_instance is not None: workspace_instance.close()
app=FastAPI(title='flyByWire local lab',lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
app.add_middleware(CORSMiddleware,allow_origin_regex=r'http://(127\.0\.0\.1|localhost)(:\d+)?',allow_methods=['GET','POST'],allow_headers=['Content-Type'])
@app.middleware('http')
async def origin_guard(request:Request,call_next):
    import re
    origin=request.headers.get('origin')
    if request.method=='POST' and origin and not re.fullmatch(r'http://(127\.0\.0\.1|localhost)(:\d+)?',origin):
        return Response('Local origins only',status_code=403)
    return await call_next(request)
@app.get('/api/state')
def state(): return lab.state()
@app.get('/api/graph')
def graph():
    return dict(nodes=lab.circuit.nodes,edges=[[int(a),int(b),float(w)] for a,b,w in zip(lab.circuit.source,lab.circuit.target,lab.circuit.contacts)])
class Command(BaseModel):
    action:str
    mode:str|None=None
    condition:str|None=None
    recording:str|None=None
    seed:int=Field(default=11,ge=0,le=1000000)
    generations:int=Field(default=8,ge=1,le=40)
@app.post('/api/command')
def command(c:Command):
    try:
        with lab.lock:
            if c.action=='run': lab.running=True; lab.status='running'
            elif c.action=='pause': lab.running=False; lab.status='paused'
            elif c.action=='step': lab.running=False; lab.status='paused'; lab.tick()
            elif c.action=='reset': lab.seed=c.seed; lab.reset(); lab.log(f'Reset to seed {lab.seed}.')
            elif c.action=='mode':
                if c.mode not in ['simulation','replay','live']: raise ValueError('Unknown source')
                lab.running=False; lab.stop_live(); lab.mode=c.mode; lab.status='paused'; lab.reset()
                if c.mode=='replay': lab.set_replay(c.recording or lab.recording)
                if c.mode=='live': lab.start_live()
            elif c.action=='condition':
                if c.condition not in ['intact','shuffled','disconnected']: raise ValueError('Unknown condition')
                lab.condition=c.condition; lab.circuit=Circuit(c.condition); lab.reset()
            elif c.action=='train': lab.start_training(c.generations,c.seed)
            elif c.action=='cancel': lab.cancel.set()
            elif c.action=='compare': lab.compare()
            elif c.action=='untrained': lab.weights=initial_weights(lab.circuit.feature_size,lab.seed); lab.reset(); lab.log('Using untrained readout.')
            elif c.action=='trained':
                if not lab.last_result: raise ValueError('Train a readout first')
                lab.weights=np.load(ROOT/'experiments'/lab.last_result['run_id']/'readout.npy'); lab.reset()
            else: raise ValueError('Unknown action')
        return lab.state()
    except (ValueError,subprocess.SubprocessError) as exc: raise HTTPException(400,str(exc))
@app.get('/api/media/{identifier:path}')
def media(identifier:str):
    selected=next((r for r in lab.recordings if r['id']==identifier),None)
    if not selected: raise HTTPException(404)
    return FileResponse(selected['path'],media_type='video/mp4')
@app.get('/api/live.jpg')
def live():
    if lab.live_jpeg is None or time.monotonic()-lab.live_last>2: raise HTTPException(503,'Camera offline')
    return Response(lab.live_jpeg,media_type='image/jpeg',headers={'Cache-Control':'no-store'})
@app.get('/api/export')
def export():
    return Response(json.dumps(dict(state=lab.state(),model_card=lab.circuit.meta['manifest']),indent=2),
                    media_type='application/json',headers={'Content-Disposition':'attachment; filename=fly-flight-experiment.json'})

# Independent full-graph worker; loading is lazy so compact experiments stay usable.
from typing import Literal
from .full_lab import FullLab
full_instance=None
full_init_lock=threading.Lock()
def full():
    global full_instance
    with full_init_lock:
        if full_instance is None: full_instance=FullLab()
    return full_instance
@app.get('/api/full/state')
def full_state(): return full().state()
@app.get('/api/full/positions')
def full_positions(): return FileResponse(DATA/'full-positions.bin',media_type='application/octet-stream')
@app.get('/api/full/neuron/{index}')
def full_neuron(index:int):
    try:return full().node(index)
    except ValueError as e:raise HTTPException(400,str(e))
class FullCommand(BaseModel):
    action:Literal['run','pause','step','reset','stimulus','neuron','recurrence']
    preset:Literal['LC4','LC9','DNa02','retina','off']='LC4'
    hz:float=Field(default=120,ge=0,le=400,allow_inf_nan=False)
    index:int=Field(default=0,ge=0,lt=166700)
    enabled:bool=True
@app.post('/api/full/command')
def full_command(c:FullCommand):
    worker=full()
    if not worker.state()['ready']:raise HTTPException(409,'Full brain is still loading')
    worker.send(c.model_dump());return {'queued':True}


# Camera-first graded workspace, independent of legacy experiment controls.
from .workspace import Workspace,validation_summary
workspace_instance=None
workspace_init_lock=threading.Lock()
def workspace():
    global workspace_instance
    with workspace_init_lock:
        if workspace_instance is None:workspace_instance=Workspace()
    return workspace_instance
@app.get('/api/workspace')
def workspace_state():return workspace().state()
@app.get('/api/workspace/frame.jpg')
def workspace_frame():
    feed=workspace().feed
    _,jpeg,_,_,last=feed.snapshot()
    if jpeg is None or time.monotonic()-last>1:raise HTTPException(503,'Video unavailable')
    return Response(jpeg,media_type='image/jpeg',headers={'Cache-Control':'no-store'})
class WorkspaceCommand(BaseModel):
    action:Literal['auto','recording','run','pause','reset']
@app.post('/api/workspace/command')
def workspace_command(c:WorkspaceCommand):
    try: workspace().command(c.action)
    except ValueError as exc: raise HTTPException(409,str(exc))
    return {'queued':True}
@app.get('/api/workspace/neuron/{index}')
def workspace_neuron(index:int):
    w=workspace()
    if not 0<=index<len(w.nodes):raise HTTPException(404,'Neuron unavailable')
    return w.nodes[index]
@app.get('/api/workspace/diagnostics')
def workspace_diagnostics():
    state=workspace().state()
    state.pop('activity',None)
    return dict(state=state,validation=validation_summary(),flight=flight.state())

@app.get('/api/workspace/telemetry')
def workspace_telemetry(after:int=0):return workspace().timing_after(after)


from .flight_runner import FlightRunner, PROFILES
flight=FlightRunner(ROOT)
class FlightCommand(BaseModel):
    action:Literal['prepare','start','lease','land','estop','stable','outcome']
    profile:str='stationary-v2'
    battery_id:str=Field(default='same-battery',max_length=60)
    charge:Literal['fresh','used','unknown']='unknown'
    run_id:str=Field(default='',max_length=100)
    outcome:str=Field(default='',max_length=30)
    motors_stopped:bool=False
    notes:str=Field(default='',max_length=1000)
    token:str|None=None
    confirmed:bool=False
@app.get('/api/flight')
def flight_state(): return flight.state()
@app.post('/api/flight/command')
def flight_command(c:FlightCommand):
    try:
        if c.action=='prepare':
            if c.profile not in PROFILES: raise ValueError('Unknown bounded test')
            return flight.launch(workspace(),c.profile,metadata=dict(battery_id=c.battery_id.strip(),charge=c.charge,notes=c.notes,battery_assumption='Same battery; operator reports no significant performance effect'))
        if c.action=='outcome': return flight.save_outcome(c.run_id,c.outcome,c.motors_stopped,c.notes)
        return flight.command(c.action,c.token,c.confirmed)
    except ValueError as exc: raise HTTPException(409,str(exc))

@app.get('/api/flight/baselines')
def flight_baselines(battery_id:str=''): return {'count':flight.baseline_count(battery_id)}

from .external_camera import router as external_camera_router
app.include_router(external_camera_router)
