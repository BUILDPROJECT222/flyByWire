"""Local external-camera artifacts. Observational only; no motor/control imports."""
import json
import re
import subprocess
import time
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field
from .core import ROOT

router=APIRouter(prefix='/api/external')
MAX_VIDEO=64*1024*1024

def capture_path(run_id, capture_id):
    if not re.fullmatch(r'trial-[0-9TZ]+',run_id) or not re.fullmatch(r'[a-f0-9]{32}',capture_id):
        raise HTTPException(400,'Invalid capture identifier')
    path=ROOT/'recordings'/run_id
    if not (path/'metadata.json').is_file(): raise HTTPException(404,'Trial not found')
    return path/('external-'+capture_id)

@router.get('/clock')
def clock():return dict(monotonic_s=time.monotonic(),utc_unix_s=time.time())

class Sample(BaseModel):
    elapsed_s:float=Field(ge=0,le=180,allow_inf_nan=False)
    received_monotonic_s:float=Field(ge=0,allow_inf_nan=False)
    media_time_s:float=Field(ge=0,allow_inf_nan=False)
    selection_id:int=Field(default=0,ge=0)
    status:str=Field(max_length=30)
    score:float=Field(ge=-1.01,le=1.01,allow_inf_nan=False)
    box:list[float]|None=Field(default=None,max_length=4)
    dx:float|None=Field(default=None,allow_inf_nan=False)
    dy:float|None=Field(default=None,allow_inf_nan=False)
class Tracking(BaseModel):
    started_monotonic_s:float=Field(ge=0,allow_inf_nan=False)
    media_time_at_start_s:float=Field(default=0,ge=0,allow_inf_nan=False)
    clock_uncertainty_ms:float=Field(ge=0,allow_inf_nan=False)
    width:int=Field(ge=1,le=1920)
    height:int=Field(ge=1,le=1080)
    camera:str=Field(max_length=200)
    samples:list[Sample]=Field(max_length=1800)

@router.post('/{run_id}/{capture_id}/tracking')
def tracking(run_id:str,capture_id:str,body:Tracking):
    base=capture_path(run_id,capture_id)
    meta=body.model_dump();samples=meta.pop('samples')
    meta.update(units='processing-image pixels; positive x right, positive y down',
        tracker='fixed template NCC; heuristic score, not probability',motor_authority=False,
        timing='browser frame callback mapped to server monotonic time; not camera exposure',
        trajectory_origin='most recent successful user selection; lost samples contain no position')
    path=Path(str(base)+'.tracking.json')
    try:
        with path.open('x') as f:json.dump(dict(metadata=meta,samples=samples),f)
    except FileExistsError:raise HTTPException(409,'Tracking already saved')
    return dict(saved=True,path=str(path),samples=len(samples))

def transcode(source,base):
    status=Path(str(base)+'.status.json')
    status.write_text(json.dumps(dict(status='converting')))
    try:
        p=subprocess.run(['ffmpeg','-nostdin','-v','error','-protocol_whitelist','file,pipe','-i',str(source),
            '-an','-c:v','libx264','-preset','ultrafast','-crf','23','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(base)+'.mp4'],capture_output=True,text=True,timeout=60)
        status.write_text(json.dumps(dict(status='ready' if p.returncode==0 else 'error',
            error=p.stderr[-1000:] if p.returncode else None,mp4=str(base)+'.mp4')))
    except Exception as exc:status.write_text(json.dumps(dict(status='error',error=str(exc))))

@router.post('/{run_id}/{capture_id}/video')
async def video(run_id:str,capture_id:str,request:Request,tasks:BackgroundTasks):
    base=capture_path(run_id,capture_id)
    mime=request.headers.get('content-type','').split(';')[0]
    if mime not in ('video/webm','video/mp4'):raise HTTPException(415,'Expected WebM or MP4')
    source=Path(str(base)+('.source.webm' if mime=='video/webm' else '.source.mp4'))
    partial=Path(str(source)+'.partial')
    if source.exists():raise HTTPException(409,'Video already saved')
    total=0
    try:
        with partial.open('xb') as f:
            async for chunk in request.stream():
                total+=len(chunk)
                if total>MAX_VIDEO:raise HTTPException(413,'External recording exceeds 64 MB')
                f.write(chunk)
        if total==0:raise HTTPException(400,'Empty video')
        partial.rename(source)
    except FileExistsError:raise HTTPException(409,'Upload already in progress')
    except BaseException:
        partial.unlink(missing_ok=True);raise
    Path(str(base)+'.status.json').write_text(json.dumps(dict(status='queued')))
    tasks.add_task(transcode,source,base)
    return dict(saved=True,bytes=total,path=str(source),status='queued')

@router.get('/{run_id}/{capture_id}/status')
def status(run_id:str,capture_id:str):
    path=Path(str(capture_path(run_id,capture_id))+'.status.json')
    return json.loads(path.read_text()) if path.exists() else dict(status='missing')
