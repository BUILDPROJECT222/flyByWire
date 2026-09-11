import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from fastapi import BackgroundTasks, HTTPException
from . import external_camera as ext

class Request:
    headers={'content-type':'video/webm'}
    def __init__(self,data):self.data=data
    async def stream(self):yield self.data

class ExternalCameraTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run='trial-20260911T120000Z';self.cid='a'*32
        self.path=self.root/'recordings'/self.run;self.path.mkdir(parents=True)
        (self.path/'metadata.json').write_text('{}')
        patcher=patch.object(ext,'ROOT',self.root);patcher.start();self.addCleanup(patcher.stop)

    def test_no_path_traversal_or_unknown_trial(self):
        for run,cid in [('../outside',self.cid),(self.run,'../bad'),('trial-999Z',self.cid)]:
            with self.assertRaises(HTTPException):ext.capture_path(run,cid)

    def test_tracking_preserves_loss_and_clock_metadata(self):
        body=ext.Tracking(started_monotonic_s=100,clock_uncertainty_ms=4,width=320,height=240,camera='Synthetic',
            samples=[ext.Sample(elapsed_s=.1,received_monotonic_s=100.1,media_time_s=1,status='lost',score=.4,selection_id=1)])
        result=ext.tracking(self.run,self.cid,body)
        saved=json.loads(Path(result['path']).read_text())
        self.assertIsNone(saved['samples'][0]['box']);self.assertIsNone(saved['samples'][0]['dx'])
        self.assertFalse(saved['metadata']['motor_authority']);self.assertEqual(saved['metadata']['clock_uncertainty_ms'],4)
        with self.assertRaises(HTTPException):ext.tracking(self.run,self.cid,body)

    def test_size_limit_removes_partial_upload(self):
        with patch.object(ext,'MAX_VIDEO',10):
            with self.assertRaises(HTTPException) as err:asyncio.run(ext.video(self.run,self.cid,Request(b'x'*11),BackgroundTasks()))
        self.assertEqual(err.exception.status_code,413)
        self.assertEqual(list(self.path.glob('*.partial')),[])

    def test_uploaded_webm_converts_to_readable_mp4(self):
        source=self.root/'test.webm'
        subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','testsrc2=size=320x240:rate=10',
                        '-t','1','-c:v','libvpx',str(source)],check=True)
        tasks=BackgroundTasks()
        result=asyncio.run(ext.video(self.run,self.cid,Request(source.read_bytes()),tasks))
        self.assertTrue(result['saved']);self.assertEqual(ext.status(self.run,self.cid)['status'],'queued')
        asyncio.run(tasks())
        status=ext.status(self.run,self.cid);self.assertEqual(status['status'],'ready',status)
        probe=subprocess.run(['ffprobe','-v','error','-count_frames','-show_entries','stream=width,height,nb_read_frames',
                              '-of','json',status['mp4']],check=True,capture_output=True,text=True)
        stream=json.loads(probe.stdout)['streams'][0]
        self.assertEqual((stream['width'],stream['height'],stream['nb_read_frames']),(320,240,'10'))
        with self.assertRaises(HTTPException):asyncio.run(ext.video(self.run,self.cid,Request(b'x'),BackgroundTasks()))

if __name__=='__main__':unittest.main()
