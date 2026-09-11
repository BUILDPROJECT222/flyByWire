"""Measure the running workspace using recorded video; never claims camera exposure latency."""
import json,time,urllib.request
from datetime import datetime,timezone
from .core import ROOT

BASE='http://127.0.0.1:8766/api/workspace'
def state():return json.load(urllib.request.urlopen(BASE+'/diagnostics',timeout=5))['state']
def command(action):
    urllib.request.urlopen(urllib.request.Request(BASE+'/command',data=json.dumps({'action':action}).encode(),headers={'Content-Type':'application/json'}),timeout=5).close()
def stats(values):
    import numpy as np
    a=np.array(values)
    return dict(count=len(a),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),p99=float(np.percentile(a,99)),maximum=float(a.max())) if len(a) else None

def main():
    out=ROOT/'experiments'/('replay-latency-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir(parents=True)
    original=state();samples=[];frames={}
    report=dict(scope='30-second running recorded-video pipeline; no physical camera or browser display latency',
                samples=samples)
    try:
        command('recording');command('run')
        deadline=time.monotonic()+25
        while time.monotonic()<deadline:
            first=state()
            if first['error']:raise RuntimeError(first['error'])
            if first['ready'] and first['camera']['fresh'] and first['camera']['source']=='recording' and first['motion']:break
            time.sleep(.1)
        else:raise RuntimeError('Replay not ready')
        start=time.monotonic()
        while time.monotonic()-start<30:
            request=time.monotonic();s=state();end=time.monotonic()
            if s['error']:raise RuntimeError(s['error'])
            entry=dict(elapsed_s=end-start,http_ms=(end-request)*1000,source=s['camera']['source'],
                camera_sequence=s['camera']['sequence'],inference_sequence=s['inference_sequence'],
                frame_age_ms=s['camera']['age_ms'],compute_ms=s['compute_ms'],
                receipt_to_completion_ms=s['source_age_ms'],result_age_ms=s['result_age_ms'],
                input_age_at_poll_ms=s['observed_input_age_ms'],fps=s['camera']['fps'],
                motion_available=s['motion'] is not None,processed_frames=s['processed_frames'],skipped_frames=s['skipped_frames'])
            samples.append(entry)
            if s['result_age_ms'] is not None:frames.setdefault(s['inference_sequence'],entry)
            time.sleep(.08)
        telemetry=json.load(urllib.request.urlopen(BASE+'/telemetry?after='+str(first['inference_sequence']),timeout=5))
        if telemetry['possibly_truncated']:raise RuntimeError('Timing ring overflowed')
        last=samples[-1];unique=[x for x in telemetry['frames'] if x['sequence']<=last['inference_sequence']]
        report['inference_records']=unique
        report['timing_source']='Every completed inference after initial frame, from bounded worker telemetry ring'
        report['actual_history_span_ms']=stats([x['history_span_ms'] for x in unique if x['history_span_ms'] is not None])
        report.update(duration_s=last['elapsed_s'],unique_inferences=len(unique),
            camera_frames=last['camera_sequence']-first['camera']['sequence'],
            processed_frames=last['processed_frames']-first['processed_frames'],
            skipped_frames=last['skipped_frames']-first['skipped_frames'],
            timing_ms=dict(compute=stats([x['compute_ms'] for x in unique]),
                receipt_to_completion=stats([x['receipt_to_completion_ms'] for x in unique]),
                input_age_at_http_poll=stats([x['input_age_at_poll_ms'] for x in samples if x['input_age_at_poll_ms'] is not None]),
                http_roundtrip=stats([x['http_ms'] for x in samples])),
            nominal_history_span_ms=1100,
            history_note='At 10 fps, twelve samples span 1.1 seconds; this is separate from per-frame processing. Scene/source resets withhold output until history refills.',
            motion_unavailable_samples=sum(not x['motion_available'] for x in samples),
            model='Full graded graph, frozen T4/T5 quadratic decoder',
            limitation='Receipt timestamp follows decode on the Mac. Excludes camera exposure, RTSP transport, browser paint and motor response. Input-age-at-poll is computed by the server before HTTP transfer; HTTP roundtrip is reported separately.')
    except Exception as e:
        report['error']=str(e);raise
    finally:
        command('auto' if original['camera']['mode']=='auto' else 'recording')
        if not original['running']:command('pause')
        (out/'report.json').write_text(json.dumps(report,indent=2));print('Report:',out/'report.json',flush=True)
    print(json.dumps({k:v for k,v in report.items() if k not in ['samples','inference_records']},indent=2))

if __name__=='__main__':main()
